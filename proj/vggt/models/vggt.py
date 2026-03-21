# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from typing import Tuple

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def _tokens_to_feature_map(self, tokens: torch.Tensor, patch_start_idx: int, image_hw: Tuple[int, int]) -> torch.Tensor:
        """Convert patch tokens [B,S,P,C] to feature map [B,S,C,Hp,Wp]."""
        H, W = int(image_hw[0]), int(image_hw[1])
        patch_tokens = tokens[:, :, patch_start_idx:]
        B, S, N, C = patch_tokens.shape
        patch_h = H // self.aggregator.patch_size
        patch_w = W // self.aggregator.patch_size
        expected_n = patch_h * patch_w
        if N != expected_n:
            raise ValueError(
                f"Patch token count mismatch: got {N}, expected {expected_n} "
                f"(H={H}, W={W}, patch_size={self.aggregator.patch_size})"
            )
        return patch_tokens.view(B, S, patch_h, patch_w, C).permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        return_intermediates: bool = False,
        intermediate_layer_idx: int = -1,
        return_all_aggregated_tokens: bool = False,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            return_intermediates (bool): If True, also return backbone intermediate variables
                for downstream networks.
            intermediate_layer_idx (int): Which aggregated-token layer to expose. Supports
                negative index (default -1 = last layer).
            return_all_aggregated_tokens (bool): If True and return_intermediates=True,
                include all aggregated token tensors in output dict.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]

                If return_intermediates=True, also includes:
                - intermediate_tokens (torch.Tensor): Selected aggregated tokens [B, S, P_all, 2C]
                - intermediate_patch_tokens (torch.Tensor): Patch-only tokens [B, S, P_patch, 2C]
                - intermediate_feature_map (torch.Tensor): Patch tokens reshaped as [B, S, 2C, Hp, Wp]
                - patch_start_idx (int): Start index of patch tokens.
                - intermediate_layer_idx (int): Resolved layer index used above.
                - aggregated_tokens_list (list[torch.Tensor], optional): Full list if requested.
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        layer_idx = None
        if return_intermediates:
            layer_idx = intermediate_layer_idx
            if layer_idx < 0:
                layer_idx = len(aggregated_tokens_list) + layer_idx
            if layer_idx < 0 or layer_idx >= len(aggregated_tokens_list):
                raise IndexError(
                    f"intermediate_layer_idx={intermediate_layer_idx} resolved to {layer_idx}, "
                    f"valid range is [0, {len(aggregated_tokens_list) - 1}]"
                )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if return_intermediates:
            tokens = aggregated_tokens_list[layer_idx]
            predictions["intermediate_tokens"] = tokens
            predictions["intermediate_patch_tokens"] = tokens[:, :, patch_start_idx:]
            predictions["intermediate_feature_map"] = self._tokens_to_feature_map(
                tokens,
                patch_start_idx=patch_start_idx,
                image_hw=images.shape[-2:],
            )
            predictions["patch_start_idx"] = patch_start_idx
            predictions["intermediate_layer_idx"] = layer_idx
            if return_all_aggregated_tokens:
                predictions["aggregated_tokens_list"] = aggregated_tokens_list

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
