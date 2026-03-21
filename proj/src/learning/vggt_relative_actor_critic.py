"""
Simplified VGGT relative-pose baseline.

This module implements a lightweight actor-critic that:
- uses frozen VGGT aggregator features from layers [4, 11, 17, 23]
- uses patch mean + max pooling only
- uses relative pose only
- summarizes sequence features with current + mean(history) + max(history)
- predicts relative actions in the environment's native action space

TODO: compare max-only vs mean-only vs mean+max patch pooling.
TODO: add a small transformer over frame tokens.
TODO: compare relative-only vs absolute+relative pose input.
TODO: compare separate actor/critic frame fusion vs shared frame fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from vggt.models.vggt import VGGT

try:
    from utils.math import matrix_from_quat
except ModuleNotFoundError:  # pragma: no cover - fallback for src.* imports
    from src.utils.math import matrix_from_quat


def _as_tuple_int(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    quat = quat / norm
    sign = torch.where(quat[..., :1] < 0.0, -1.0, 1.0)
    return quat * sign


def _pose7_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    # pose: [..., 7] = [tx, ty, tz, qw, qx, qy, qz]
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected pose [...,7], got shape {tuple(pose.shape)}")
    translation = pose[..., :3]
    quat = _normalize_quat_wxyz(pose[..., 3:7])
    rotation = matrix_from_quat(quat)
    out = torch.eye(4, device=pose.device, dtype=pose.dtype).expand(*pose.shape[:-1], 4, 4).clone()
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    return out


def _flatten_pose16_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 16:
        raise ValueError(f"Expected pose [...,16], got shape {tuple(pose.shape)}")
    return pose.reshape(*pose.shape[:-1], 4, 4)


def _invert_se3(mats: torch.Tensor) -> torch.Tensor:
    # mats: [..., 4, 4]
    rot = mats[..., :3, :3]
    trans = mats[..., :3, 3]
    rot_t = rot.transpose(-1, -2)
    out = torch.eye(4, device=mats.device, dtype=mats.dtype).expand_as(mats).clone()
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -(rot_t @ trans.unsqueeze(-1)).squeeze(-1)
    return out


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    # rotation: [..., 3, 3]
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation [...,3,3], got shape {tuple(rotation.shape)}")
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def _build_gelu_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class ObservationAdapterOutput:
    images: torch.Tensor
    pose_mats: torch.Tensor
    valid_mask: torch.Tensor | None
    lead_shape: tuple[int, ...]


class VGGTFeatureExtractor(nn.Module):
    """Frozen VGGT aggregator wrapper with patch mean/max pooling."""

    def __init__(
        self,
        *,
        selected_layers: Sequence[int] = (4, 11, 17, 23),
        visual_dim: int = 256,
        use_layernorm: bool = True,
        use_patch_mean_pool: bool = True,
        use_patch_max_pool: bool = True,
        freeze_aggregator: bool = True,
        model_id: str = "facebook/VGGT-1B",
        model_cache_dir: str = "",
        model_local_files_only: bool = True,
        vggt_input_size: int = 518,
    ) -> None:
        super().__init__()
        self.selected_layers = _as_tuple_int(selected_layers)
        if not self.selected_layers:
            raise ValueError("selected_layers must be non-empty.")
        self.visual_dim = int(visual_dim)
        self.use_patch_mean_pool = bool(use_patch_mean_pool)
        self.use_patch_max_pool = bool(use_patch_max_pool)
        if not self.use_patch_mean_pool and not self.use_patch_max_pool:
            raise ValueError("At least one of use_patch_mean_pool/use_patch_max_pool must be True.")
        self.freeze_aggregator = bool(freeze_aggregator)
        self.vggt_input_size = int(vggt_input_size)
        self.forward_call_count = 0

        if model_id:
            backbone = VGGT.from_pretrained(
                model_id,
                cache_dir=model_cache_dir or None,
                local_files_only=model_local_files_only,
            )
        else:
            backbone = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)

        self.aggregator = backbone.aggregator
        del backbone

        token_dim = int(self.aggregator.camera_token.shape[-1] * 2)
        pooled_factor = int(self.use_patch_mean_pool) + int(self.use_patch_max_pool)
        concat_dim = len(self.selected_layers) * pooled_factor * token_dim
        self.patch_norm = nn.LayerNorm(token_dim) if use_layernorm else nn.Identity()
        self.visual_proj = _build_gelu_mlp(concat_dim, max(512, self.visual_dim * 2), self.visual_dim)

        if self.freeze_aggregator:
            for param in self.aggregator.parameters():
                param.requires_grad_(False)
            self.aggregator.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_aggregator:
            self.aggregator.eval()
        return self

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, S, 3, H, W]
        if self.vggt_input_size <= 0:
            return images
        _, _, _, h, w = images.shape
        if h == self.vggt_input_size and w == self.vggt_input_size:
            return images
        bsz, seq = images.shape[:2]
        flat = images.reshape(bsz * seq, *images.shape[-3:])
        flat = F.interpolate(
            flat,
            size=(self.vggt_input_size, self.vggt_input_size),
            mode="bilinear",
            align_corners=False,
        )
        return flat.reshape(bsz, seq, 3, self.vggt_input_size, self.vggt_input_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, S, 3, H, W]
        if images.dim() != 5:
            raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {images.shape[2]}")
        if images.shape[1] < 1:
            raise ValueError("Sequence length S must be >= 1.")

        self.forward_call_count += 1
        images = self._resize_images(images)
        self.aggregator.eval()
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        num_layers = len(aggregated_tokens_list)
        for layer_idx in self.selected_layers:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise IndexError(f"Layer index {layer_idx} out of range for {num_layers} aggregator outputs.")

        pooled_layers = []
        for layer_idx in self.selected_layers:
            # x: [B, S, N_patch, D]
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:].float()
            x = self.patch_norm(x)
            pooled_parts = []
            if self.use_patch_mean_pool:
                mean_pool = x.mean(dim=2)  # [B, S, D]
                pooled_parts.append(mean_pool)
            if self.use_patch_max_pool:
                max_pool = x.max(dim=2).values  # [B, S, D]
                pooled_parts.append(max_pool)
            pooled_layers.append(torch.cat(pooled_parts, dim=-1))  # [B, S, 2D] or [B, S, D]

        visual_stack = torch.cat(pooled_layers, dim=-1)  # [B, S, L * pool_factor * D]
        visual_feats = self.visual_proj(visual_stack)  # [B, S, visual_dim]
        return visual_feats


class RelativePoseEmbedder(nn.Module):
    """Embed per-frame pose relative to the latest valid frame in the sequence."""

    def __init__(self, pose_dim: int = 64) -> None:
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.pose_mlp = _build_gelu_mlp(10, 128, self.pose_dim)

    def forward(self, pose_mats: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # pose_mats: [B, S, 4, 4]
        if pose_mats.dim() != 4 or pose_mats.shape[-2:] != (4, 4):
            raise ValueError(f"Expected pose_mats [B,S,4,4], got {tuple(pose_mats.shape)}")

        batch, seq_len = pose_mats.shape[:2]
        device = pose_mats.device
        dtype = pose_mats.dtype

        if valid_mask is None:
            valid_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        any_valid = valid_mask.any(dim=1)
        default_last = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)
        last_valid = valid_mask.long().argmax(dim=1)
        flipped = torch.flip(valid_mask.long(), dims=[1]).argmax(dim=1)
        last_valid = (seq_len - 1) - flipped
        last_valid = torch.where(any_valid, last_valid, default_last)

        current_pose = pose_mats[torch.arange(batch, device=device), last_valid]  # [B, 4, 4]
        current_pose_inv = _invert_se3(current_pose)  # [B, 4, 4]
        rel_pose = current_pose_inv[:, None] @ pose_mats  # [B, S, 4, 4]

        rel_translation = rel_pose[..., :3, 3]  # [B, S, 3]
        rel_rotation = rel_pose[..., :3, :3]  # [B, S, 3, 3]
        rel_rotation_6d = rotation_matrix_to_6d(rel_rotation)  # [B, S, 6]

        frame_indices = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
        denom = last_valid.to(dtype=dtype).clamp_min(1.0).unsqueeze(1)
        recency = (frame_indices - last_valid.to(dtype=dtype).unsqueeze(1)) / denom  # [B, S]
        recency = recency.unsqueeze(-1)  # [B, S, 1]

        raw_pose = torch.cat([rel_translation, rel_rotation_6d, recency], dim=-1)  # [B, S, 10]
        raw_pose = raw_pose * valid_mask.unsqueeze(-1).to(dtype)
        pose_feats = self.pose_mlp(raw_pose)  # [B, S, pose_dim]
        return pose_feats


class FrameFusion(nn.Module):
    """Fuse per-frame visual and relative-pose embeddings."""

    def __init__(self, visual_dim: int = 256, pose_dim: int = 64, fused_dim: int = 256) -> None:
        super().__init__()
        self.fused_dim = int(fused_dim)
        self.fuse = _build_gelu_mlp(int(visual_dim) + int(pose_dim), max(256, self.fused_dim), self.fused_dim)

    def forward(self, visual_feats: torch.Tensor, pose_feats: torch.Tensor) -> torch.Tensor:
        # visual_feats: [B, S, 256]
        # pose_feats:   [B, S, 64]
        if visual_feats.shape[:2] != pose_feats.shape[:2]:
            raise ValueError(
                f"visual/pose leading dims mismatch: {tuple(visual_feats.shape)} vs {tuple(pose_feats.shape)}"
            )
        fused_input = torch.cat([visual_feats, pose_feats], dim=-1)  # [B, S, 320]
        fused_feats = self.fuse(fused_input)  # [B, S, fused_dim]
        return fused_feats


class HistoryPooling(nn.Module):
    """Current + mean(history) + max(history) pooling with optional valid_mask."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, fused_feats: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # fused_feats: [B, S, C]
        if fused_feats.dim() != 3:
            raise ValueError(f"Expected fused_feats [B,S,C], got {tuple(fused_feats.shape)}")

        batch, seq_len, feat_dim = fused_feats.shape
        device = fused_feats.device
        dtype = fused_feats.dtype

        if valid_mask is None:
            valid_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        outputs = []
        frame_ids = torch.arange(seq_len, device=device)
        for batch_idx in range(batch):
            mask = valid_mask[batch_idx]
            if mask.any():
                current_idx = int(frame_ids[mask][-1].item())
            else:
                current_idx = seq_len - 1

            current = fused_feats[batch_idx, current_idx]  # [C]
            hist_mask = mask & (frame_ids < current_idx)
            if hist_mask.any():
                hist = fused_feats[batch_idx, hist_mask]  # [H, C]
                hist_mean = hist.mean(dim=0)
                hist_max = hist.max(dim=0).values
            else:
                hist_mean = torch.zeros(feat_dim, device=device, dtype=dtype)
                hist_max = torch.zeros(feat_dim, device=device, dtype=dtype)

            outputs.append(torch.cat([current, hist_mean, hist_max], dim=-1))  # [3C]

        return torch.stack(outputs, dim=0)  # [B, 3C]


class GaussianActorHead(nn.Module):
    """Small Gaussian policy head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        hidden_dims = list(int(v) for v in hidden_dims)
        if len(hidden_dims) != 2:
            raise ValueError(f"GaussianActorHead expects exactly 2 hidden dims, got {hidden_dims}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], int(action_dim) * 2),
        )

    def forward(self, state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state_feat)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return mean, log_std


class ValueHead(nn.Module):
    """Small critic head."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        hidden_dims = list(int(v) for v in hidden_dims)
        if len(hidden_dims) != 2:
            raise ValueError(f"ValueHead expects exactly 2 hidden dims, got {hidden_dims}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.net(state_feat)


class VGGTRelativeActorCritic(nn.Module):
    """Simplified actor-critic using frozen VGGT aggregator features and relative poses."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (256, 128),
        init_noise_std: float = 0.2,
        selected_layers: Sequence[int] = (4, 11, 17, 23),
        visual_dim: int = 256,
        pose_dim: int = 64,
        fused_dim: int = 256,
        freeze_aggregator: bool = True,
        use_patch_mean_pool: bool = True,
        use_patch_max_pool: bool = True,
        use_layernorm: bool = True,
        model_id: str = "facebook/VGGT-1B",
        model_cache_dir: str = "",
        model_local_files_only: bool = True,
        image_obs_key: str = "obs_rgb_tensor",
        pose_obs_key: str = "obs_pose",
        valid_mask_obs_key: str = "",
        vggt_input_size: int = 518,
        activation: str = "gelu",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "VGGTRelativeActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.num_actions = int(num_actions)
        self.image_obs_key = self._resolve_image_key(obs, obs_groups.get("policy", []), image_obs_key)
        self.pose_obs_key = self._resolve_pose_key(obs, obs_groups.get("policy", []), pose_obs_key)
        self.valid_mask_obs_key = self._resolve_valid_mask_key(obs, valid_mask_obs_key)
        self.activation = activation
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)

        self.feature_extractor = VGGTFeatureExtractor(
            selected_layers=selected_layers,
            visual_dim=visual_dim,
            use_layernorm=use_layernorm,
            use_patch_mean_pool=use_patch_mean_pool,
            use_patch_max_pool=use_patch_max_pool,
            freeze_aggregator=freeze_aggregator,
            model_id=model_id,
            model_cache_dir=model_cache_dir,
            model_local_files_only=model_local_files_only,
            vggt_input_size=vggt_input_size,
        )
        self.pose_embedder = RelativePoseEmbedder(pose_dim=pose_dim)
        self.frame_fusion = FrameFusion(visual_dim=visual_dim, pose_dim=pose_dim, fused_dim=fused_dim)
        self.history_pooling = HistoryPooling()

        state_dim = int(fused_dim) * 3
        self.actor_head = GaussianActorHead(state_dim, self.num_actions, actor_hidden_dims)
        self.value_head = ValueHead(state_dim, critic_hidden_dims)
        self.distribution: Normal | None = None

        # Initialize actor log_std bias near init_noise_std.
        final_linear = self.actor_head.net[-1]
        if isinstance(final_linear, nn.Linear):
            with torch.no_grad():
                final_linear.bias[self.num_actions :] = torch.log(
                    torch.full((self.num_actions,), float(init_noise_std), dtype=final_linear.bias.dtype)
                )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.feature_extractor.train(mode)
        return self

    @staticmethod
    def _resolve_image_key(obs, obs_groups: Sequence[str], preferred_key: str) -> str:
        if preferred_key in obs:
            return preferred_key
        for key in obs_groups:
            if key in obs and obs[key].dim() >= 5:
                return key
        raise KeyError(f"Could not find image observation key. Preferred={preferred_key}, groups={list(obs_groups)}")

    @staticmethod
    def _resolve_pose_key(obs, obs_groups: Sequence[str], preferred_key: str) -> str:
        if preferred_key in obs:
            return preferred_key
        for key in obs_groups:
            if key in obs and obs[key].shape[-1] in (7, 16):
                return key
        raise KeyError(f"Could not find pose observation key. Preferred={preferred_key}, groups={list(obs_groups)}")

    @staticmethod
    def _resolve_valid_mask_key(obs, preferred_key: str) -> str | None:
        if preferred_key and preferred_key in obs:
            return preferred_key
        for key in ("valid_mask", "obs_valid_mask", "frame_valid_mask"):
            if key in obs:
                return key
        return None

    @staticmethod
    def _prepare_image_sequence(images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        # Supported:
        # - [B, S, H, W, 3]
        # - [B, S, 3, H, W]
        # - [T, B, S, H, W, 3]
        # - [T, B, S, 3, H, W]
        if images.dim() == 5:
            if images.shape[-1] == 3:
                images = images.permute(0, 1, 4, 2, 3)  # [B, S, 3, H, W]
            elif images.shape[2] != 3:
                raise ValueError(f"Unsupported 5D image shape {tuple(images.shape)}")
        elif images.dim() == 6:
            if images.shape[-1] == 3:
                images = images.permute(0, 1, 2, 5, 3, 4)  # [T, B, S, 3, H, W]
            elif images.shape[3] != 3:
                raise ValueError(f"Unsupported 6D image shape {tuple(images.shape)}")
        else:
            raise ValueError(f"Expected image tensor rank 5 or 6, got {tuple(images.shape)}")

        lead_shape = tuple(images.shape[:-4])
        seq_len = int(images.shape[-4])
        if seq_len < 1:
            raise ValueError("Sequence length S must be >= 1.")
        flat = images.reshape(-1, seq_len, *images.shape[-3:])  # [N, S, 3, H, W]
        return flat, lead_shape

    @staticmethod
    def _prepare_valid_mask(valid_mask: torch.Tensor | None, lead_shape: tuple[int, ...], seq_len: int, device) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        expected_rank = len(lead_shape) + 1
        if valid_mask.dim() != expected_rank:
            raise ValueError(
                f"Expected valid_mask rank {expected_rank} for lead_shape={lead_shape}, got {tuple(valid_mask.shape)}"
            )
        if tuple(valid_mask.shape[:-1]) != lead_shape or int(valid_mask.shape[-1]) != seq_len:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} incompatible with lead_shape={lead_shape}, seq_len={seq_len}"
            )
        return valid_mask.reshape(-1, seq_len)

    @staticmethod
    def _prepare_pose_sequence(
        poses: torch.Tensor,
        lead_shape: tuple[int, ...],
        seq_len: int,
        device,
        dtype,
    ) -> torch.Tensor:
        poses = poses.to(device=device, dtype=dtype)

        has_matrix = poses.shape[-2:] == (4, 4)
        has_pose7 = poses.shape[-1] == 7
        has_pose16 = poses.shape[-1] == 16
        if not (has_matrix or has_pose7 or has_pose16):
            raise ValueError(f"Unsupported pose shape {tuple(poses.shape)}")

        if has_matrix:
            if poses.dim() == len(lead_shape) + 2:
                poses = poses.unsqueeze(-3).expand(*lead_shape, seq_len, 4, 4)
            elif poses.dim() != len(lead_shape) + 3:
                raise ValueError(f"Unexpected pose matrix rank for shape {tuple(poses.shape)}")
            mats = poses
        else:
            if poses.dim() == len(lead_shape) + 1:
                poses = poses.unsqueeze(-2).expand(*lead_shape, seq_len, poses.shape[-1])
            elif poses.dim() != len(lead_shape) + 2:
                raise ValueError(f"Unexpected pose vector rank for shape {tuple(poses.shape)}")
            if has_pose7:
                mats = _pose7_to_matrix(poses)
            else:
                mats = _flatten_pose16_to_matrix(poses)

        if tuple(mats.shape[:-3]) != lead_shape or int(mats.shape[-3]) != seq_len:
            raise ValueError(
                f"Pose sequence shape {tuple(mats.shape)} incompatible with lead_shape={lead_shape}, seq_len={seq_len}"
            )
        return mats.reshape(-1, seq_len, 4, 4)

    @staticmethod
    def _flatten_sequence_features(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
        if x.dim() < 3:
            raise ValueError(f"Expected sequence tensor with shape [...,S,C], got {tuple(x.shape)}")
        lead_shape = tuple(x.shape[:-2])
        seq_len = int(x.shape[-2])
        feat_dim = int(x.shape[-1])
        return x.reshape(-1, seq_len, feat_dim), lead_shape, seq_len

    @staticmethod
    def _flatten_state_features(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.dim() < 2:
            raise ValueError(f"Expected state tensor with shape [...,C], got {tuple(x.shape)}")
        lead_shape = tuple(x.shape[:-1])
        feat_dim = int(x.shape[-1])
        return x.reshape(-1, feat_dim), lead_shape

    def _adapt_visual_observations(self, obs) -> tuple[torch.Tensor, tuple[int, ...]]:
        return self._prepare_image_sequence(obs[self.image_obs_key])

    def _adapt_pose_observations(
        self,
        obs,
        *,
        lead_shape: tuple[int, ...],
        seq_len: int,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        valid_mask = None
        if self.valid_mask_obs_key is not None:
            valid_mask = self._prepare_valid_mask(obs[self.valid_mask_obs_key], lead_shape, seq_len, device)
        pose_mats = self._prepare_pose_sequence(
            obs[self.pose_obs_key],
            lead_shape,
            seq_len,
            device=device,
            dtype=dtype,
        )
        return pose_mats, valid_mask

    def _adapt_observations(self, obs) -> ObservationAdapterOutput:
        images, lead_shape = self._adapt_visual_observations(obs)
        seq_len = int(images.shape[1])
        pose_mats, valid_mask = self._adapt_pose_observations(
            obs,
            lead_shape=lead_shape,
            seq_len=seq_len,
            device=images.device,
            dtype=images.dtype,
        )
        return ObservationAdapterOutput(
            images=images,
            pose_mats=pose_mats,
            valid_mask=valid_mask,
            lead_shape=lead_shape,
        )

    @staticmethod
    def _restore_leading(x: torch.Tensor, lead_shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(*lead_shape, *x.shape[1:])

    def encode_visual(self, obs) -> torch.Tensor:
        images, lead_shape = self._adapt_visual_observations(obs)
        visual_feats = self.feature_extractor(images)
        return self._restore_leading(visual_feats, lead_shape)

    def encode_pose(self, obs) -> torch.Tensor:
        images, lead_shape = self._adapt_visual_observations(obs)
        seq_len = int(images.shape[1])
        pose_mats, valid_mask = self._adapt_pose_observations(
            obs,
            lead_shape=lead_shape,
            seq_len=seq_len,
            device=images.device,
            dtype=images.dtype,
        )
        pose_feats = self.pose_embedder(pose_mats, valid_mask)
        return self._restore_leading(pose_feats, lead_shape)

    def fuse_features(self, visual_feats: torch.Tensor, pose_feats: torch.Tensor) -> torch.Tensor:
        flat_visual, lead_shape, seq_len = self._flatten_sequence_features(visual_feats)
        flat_pose, pose_lead_shape, pose_seq_len = self._flatten_sequence_features(pose_feats)
        if lead_shape != pose_lead_shape or seq_len != pose_seq_len:
            raise ValueError(
                f"visual/pose cached features mismatch: {tuple(visual_feats.shape)} vs {tuple(pose_feats.shape)}"
            )
        fused_feats = self.frame_fusion(flat_visual, flat_pose)
        return self._restore_leading(fused_feats, lead_shape)

    def pool_history(self, fused_feats: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        flat_fused, lead_shape, seq_len = self._flatten_sequence_features(fused_feats)
        flat_valid_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=fused_feats.device, dtype=torch.bool)
            expected_shape = (*lead_shape, seq_len)
            if tuple(valid_mask.shape) != expected_shape:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} incompatible with fused_feats {tuple(fused_feats.shape)}"
                )
            flat_valid_mask = valid_mask.reshape(-1, seq_len)
        state_feat = self.history_pooling(flat_fused, flat_valid_mask)
        return self._restore_leading(state_feat, lead_shape)

    def actor_from_state(self, state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_state, lead_shape = self._flatten_state_features(state_feat)
        mean, log_std = self.actor_head(flat_state)
        return self._restore_leading(mean, lead_shape), self._restore_leading(log_std, lead_shape)

    def value_from_state(self, state_feat: torch.Tensor) -> torch.Tensor:
        flat_state, lead_shape = self._flatten_state_features(state_feat)
        value = self.value_head(flat_state)
        return self._restore_leading(value, lead_shape)

    def encode_trainable_state_from_cached_visual(self, visual_feats: torch.Tensor, obs) -> torch.Tensor:
        flat_visual, lead_shape, seq_len = self._flatten_sequence_features(visual_feats)
        pose_mats, valid_mask = self._adapt_pose_observations(
            obs,
            lead_shape=lead_shape,
            seq_len=seq_len,
            device=flat_visual.device,
            dtype=flat_visual.dtype,
        )
        pose_feats = self.pose_embedder(pose_mats, valid_mask)
        fused_feats = self.frame_fusion(flat_visual, pose_feats)
        state_feat = self.history_pooling(fused_feats, valid_mask)
        return self._restore_leading(state_feat, lead_shape)

    def _update_distribution_from_state(self, state_feat: torch.Tensor) -> None:
        mean, log_std = self.actor_from_state(state_feat)
        self.distribution = Normal(mean, torch.exp(log_std))

    def act_from_state(self, state_feat: torch.Tensor) -> torch.Tensor:
        self._update_distribution_from_state(state_feat)
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.sample()

    def evaluate_from_state(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.value_from_state(state_feat)

    def act_with_cached_visual(self, obs, visual_feats: torch.Tensor | None = None) -> torch.Tensor:
        if visual_feats is None:
            visual_feats = self.encode_visual(obs)
        state_feat = self.encode_trainable_state_from_cached_visual(visual_feats, obs)
        return self.act_from_state(state_feat)

    def get_visual_encode_call_count(self, reset: bool = False) -> int:
        count = int(self.feature_extractor.forward_call_count)
        if reset:
            self.feature_extractor.forward_call_count = 0
        return count

    def _encode_observation(self, obs, *, return_debug: bool = False):
        visual_feats = self.encode_visual(obs)
        flat_visual, lead_shape, seq_len = self._flatten_sequence_features(visual_feats)
        pose_mats, valid_mask = self._adapt_pose_observations(
            obs,
            lead_shape=lead_shape,
            seq_len=seq_len,
            device=flat_visual.device,
            dtype=flat_visual.dtype,
        )
        pose_feats = self.pose_embedder(pose_mats, valid_mask)
        fused_feats = self.frame_fusion(flat_visual, pose_feats)
        state_feat = self.history_pooling(fused_feats, valid_mask)

        if not return_debug:
            return state_feat, lead_shape

        debug = {
            "visual_feats": visual_feats,
            "pose_feats": self._restore_leading(pose_feats, lead_shape),
            "fused_feats": self._restore_leading(fused_feats, lead_shape),
            "state_feat": self._restore_leading(state_feat, lead_shape),
        }
        return state_feat, lead_shape, debug

    def forward(self, obs, return_debug: bool = False):
        if return_debug:
            state_feat, lead_shape, debug = self._encode_observation(obs, return_debug=True)
        else:
            state_feat, lead_shape = self._encode_observation(obs, return_debug=False)

        action_mean, action_log_std = self.actor_from_state(self._restore_leading(state_feat, lead_shape))
        value = self.value_from_state(self._restore_leading(state_feat, lead_shape))
        outputs = {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
        }
        if return_debug:
            outputs.update(debug)
        return outputs

    def _update_distribution(self, state_feat: torch.Tensor, lead_shape: tuple[int, ...]) -> None:
        self._update_distribution_from_state(self._restore_leading(state_feat, lead_shape))

    def act(self, obs, **kwargs) -> torch.Tensor:
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        self._update_distribution(state_feat, lead_shape)
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        mean, _ = self.actor_from_state(self._restore_leading(state_feat, lead_shape))
        return mean

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        return self.value_from_state(self._restore_leading(state_feat, lead_shape))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized. Call act() before get_actions_log_prob().")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs) -> None:
        return None

    def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
        return None

    def get_hidden_states(self):
        return None


def smoke_test() -> None:
    """Minimal smoke test with fake data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    seq_len = 3
    height = 64
    width = 64
    action_dim = 7

    fake_obs = {
        "obs_rgb_tensor": torch.rand(batch, seq_len, 3, height, width, device=device),
        "obs_pose": torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(batch, seq_len, 1, 1),
    }
    model = VGGTRelativeActorCritic(
        obs=fake_obs,
        obs_groups={"policy": ["obs_rgb_tensor", "obs_pose"], "critic": ["obs_rgb_tensor", "obs_pose"]},
        num_actions=action_dim,
        model_id="",
        selected_layers=(4, 11, 17, 23),
        visual_dim=256,
        pose_dim=64,
        fused_dim=256,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        freeze_aggregator=True,
        use_patch_mean_pool=True,
        use_patch_max_pool=True,
        use_layernorm=True,
    ).to(device)

    outputs = model(fake_obs, return_debug=True)
    print("visual_feats:", tuple(outputs["visual_feats"].shape))
    print("pose_feats:", tuple(outputs["pose_feats"].shape))
    print("fused_feats:", tuple(outputs["fused_feats"].shape))
    print("state_feat:", tuple(outputs["state_feat"].shape))
    print("action_mean:", tuple(outputs["action_mean"].shape))
    print("action_log_std:", tuple(outputs["action_log_std"].shape))
    print("value:", tuple(outputs["value"].shape))
