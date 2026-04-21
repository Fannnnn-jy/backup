"""Pluggable 3D reconstruction model interface.

Supports both **batch** models (e.g. VGGT — all frames at once) and
**incremental** models (e.g. SLAM-Former — one frame at a time).

To add a new backend:
  1. Subclass ``ReconstructionModel``
  2. Register it in ``MODEL_REGISTRY`` at the bottom of this file
  3. Set ``--pointcloud_eval.reconstruction_model <key>`` when launching training
"""

from __future__ import annotations

import os
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from learning.config import PointCloudEvalConfig
    from utils.pointcloud_eval import FrameData


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ReconstructionModel(ABC):
    """Abstract base for multi-view 3D reconstruction models.

    **Batch models** only need to implement ``predict_pointcloud``.

    **Incremental models** should also override:
      - ``on_frame``          — called every time a new frame is recorded
      - ``on_episode_reset``  — called when an episode ends (before the next begins)

    ``predict_pointcloud`` is always called at episode end.  For incremental
    models this is where you finalize and extract the accumulated map.

    The returned dict must contain:
      - ``"points"``:      np.ndarray (N, 3)
      - ``"extrinsics"``:  np.ndarray (K, 4, 4)   (K = number of keyframes)
      - ``"intrinsics"``:  np.ndarray (K, 3, 3)
      - ``"depth"``:       np.ndarray (K, H, W)
    """

    # -- incremental hooks (no-ops by default for batch models) -------------

    def on_frame(self, env_index: int, frame: FrameData) -> None:
        """Called each time a frame is recorded during rollout."""

    def on_episode_reset(self, env_index: int) -> None:
        """Called when an episode finishes, before the next one starts."""

    # -- required -----------------------------------------------------------

    @abstractmethod
    def predict_pointcloud(
        self,
        frames: List[FrameData],
        cfg: PointCloudEvalConfig,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        ...


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------

class PlaceholderReconstructionModel(ReconstructionModel):

    def predict_pointcloud(self, frames, cfg, device):
        raise NotImplementedError(
            "PlaceholderReconstructionModel.predict_pointcloud is not implemented. "
            "Subclass ReconstructionModel and register your model in MODEL_REGISTRY."
        )


# ---------------------------------------------------------------------------
# VGGT wrapper (original behaviour, batch)
# ---------------------------------------------------------------------------

def _resize_square(rgb: torch.Tensor, target: int) -> torch.Tensor:
    if target <= 0:
        return rgb
    if rgb.shape[-1] == target and rgb.shape[-2] == target:
        return rgb
    return F.interpolate(rgb, size=(target, target), mode="bilinear", align_corners=False)


class VGGTReconstructionModel(ReconstructionModel):

    def __init__(self, cfg: PointCloudEvalConfig, device: torch.device) -> None:
        from vggt.models.vggt import VGGT

        self._model = VGGT.from_pretrained(
            cfg.model_id,
            local_files_only=cfg.model_local_files_only,
            cache_dir=cfg.model_cache_dir or None,
        ).to(device)
        self._model.eval()

    def predict_pointcloud(self, frames, cfg, device):
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        rgbs = [f.rgb for f in frames]
        arr = np.clip(np.stack(rgbs, axis=0), 0.0, 1.0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        tensor = _resize_square(tensor, int(cfg.vggt_input_size))
        images = tensor.to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            amp_enabled = bool(cfg.use_amp and device.type == "cuda")
            dtype = torch.bfloat16 if str(cfg.amp_dtype).lower() == "bfloat16" else torch.float16
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=dtype):
                predictions = self._model(images)

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        depth = predictions["depth"].squeeze(0)
        depth_np = depth.detach().cpu().numpy()
        if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        extr_np = extrinsics.squeeze(0).detach().cpu().numpy()
        intr_np = intrinsics.squeeze(0).detach().cpu().numpy()

        points = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)
        points = points.reshape(-1, 3)
        valid = depth_np.reshape(-1) > 0.0
        points = points[valid]

        return {
            "points": points,
            "extrinsics": extr_np,
            "intrinsics": intr_np,
            "depth": depth_np,
        }


# ---------------------------------------------------------------------------
# SLAM-Former wrapper (incremental)
# ---------------------------------------------------------------------------

class SLAMFormerReconstructionModel(ReconstructionModel):
    """Incremental reconstruction using SLAM-Former.

    Frames are fed one-by-one via ``on_frame`` → ``slam.step()``.
    At episode end ``predict_pointcloud`` calls ``slam.terminate()``
    and extracts/aligns the accumulated map.

    Config fields used (all under ``PointCloudEvalConfig``):
      - slam_root           — path to SLAM-Former repo
      - slam_ckpt_path      — path to checkpoint
      - slam_target_size    — longer-side resize (default 518)
      - slam_kf_th          — keyframe threshold (default 0.1)
      - slam_retention_ratio— KV pruning ratio (default 0.5)
      - slam_bn_every       — backend every N KFs (default 10)
      - slam_conf_percentile— confidence filter percentile (default 15)
      - slam_save_gmem      — store token maps on CPU (default False)
    """

    def __init__(self, cfg: PointCloudEvalConfig, device: torch.device, shared_slam_model=None) -> None:
        self._cfg = cfg
        self._device = device
        self._verbose = bool(getattr(cfg, "verbose", True))
        self._shared_slam_model = shared_slam_model

        # Resolve paths
        self._slam_root = getattr(cfg, "slam_root", "")
        if not self._slam_root:
            raise ValueError("slam_root must be set for slamformer reconstruction model")
        self._slam_root = os.path.expanduser(self._slam_root)

        self._ckpt_path = getattr(cfg, "slam_ckpt_path", "")
        if not self._ckpt_path:
            raise ValueError("slam_ckpt_path must be set for slamformer reconstruction model")

        # Add SLAM-Former to sys.path (once)
        if self._slam_root not in sys.path:
            sys.path.insert(0, self._slam_root)
        croco = os.path.join(self._slam_root, "src", "croco")
        if os.path.isdir(croco) and croco not in sys.path:
            sys.path.insert(0, croco)

        # Disable flash SDP (required by SLAM-Former)
        torch.backends.cuda.enable_flash_sdp(False)

        # Hyperparams
        self._target_size = int(getattr(cfg, "slam_target_size", 518))
        self._kf_th = float(getattr(cfg, "slam_kf_th", 0.1))
        self._retention_ratio = float(getattr(cfg, "slam_retention_ratio", 0.5))
        self._bn_every = int(getattr(cfg, "slam_bn_every", 10))
        self._conf_percentile = float(getattr(cfg, "slam_conf_percentile", 15))
        self._save_gmem = bool(getattr(cfg, "slam_save_gmem", False))

        # Per-env SLAM instances
        self._slam_instances: Dict[int, object] = {}

    def _create_slam(self):
        """Create a fresh SLAM instance."""
        prev_cwd = os.getcwd()
        os.chdir(self._slam_root)
        try:
            from slam.demo import SLAM
            slam = SLAM(
                outdir=tempfile.mkdtemp(prefix="slamformer_train_"),
                ckpt_path=self._ckpt_path,
                target_size=self._target_size,
                kf_th=self._kf_th,
                retention_ratio=self._retention_ratio,
                bn_every=self._bn_every,
                save_gmem=self._save_gmem,
                vis=False,
                shared_model=self._shared_slam_model,
            )
        finally:
            os.chdir(prev_cwd)
        return slam

    def _get_slam(self, env_index: int):
        if env_index not in self._slam_instances:
            self._slam_instances[env_index] = self._create_slam()
        return self._slam_instances[env_index]

    def on_frame(self, env_index: int, frame: FrameData) -> None:
        """Feed a single frame to SLAM-Former incrementally."""
        slam = self._get_slam(env_index)

        # Convert RGB float [0,1] HxWx3 → BGR uint8 for cv2 (what SLAM-Former expects)
        rgb = np.asarray(frame.rgb, dtype=np.float32)
        if rgb.max() <= 1.0 + 1e-3:
            rgb = (rgb * 255.0).clip(0, 255)
        bgr = rgb[..., ::-1].astype(np.uint8)

        # Timestamp = frame index within this SLAM instance
        timestamp = float(slam.fid + 1)
        slam.step(timestamp, bgr)

    def on_episode_reset(self, env_index: int) -> None:
        """Discard the SLAM instance for this env so the next episode starts fresh."""
        slam = self._slam_instances.pop(env_index, None)
        if slam is not None:
            del slam
            torch.cuda.empty_cache()

    def predict_pointcloud(
        self,
        frames: List[FrameData],
        cfg: PointCloudEvalConfig,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Finalize the SLAM map and extract points + poses.

        ``frames`` are the buffered frames (same ones that were fed via
        ``on_frame``).  We use their GT poses for Umeyama alignment.
        The SLAM instance for env_index=0 is used (single-env training).
        """
        # Find the SLAM instance — we use the first (and typically only) one
        if not self._slam_instances:
            return self._empty_result()
        env_index = next(iter(self._slam_instances))
        slam = self._slam_instances[env_index]

        # Finalize
        prev_cwd = os.getcwd()
        os.chdir(self._slam_root)
        try:
            slam.terminate()
        finally:
            os.chdir(prev_cwd)

        # Extract from the best available map
        map_to_use = slam.map_opt if slam.map_opt is not None else slam.map
        if map_to_use is None:
            return self._empty_result()

        result = slam.model.extract(slam.maybe_to_cuda(map_to_use))
        pts_raw = result["points"].cpu().numpy()        # (1, S, H, W, 3)
        conf = result["conf"].cpu().numpy()              # (1, S, H, W)
        camera_poses = result["camera_poses"].cpu().numpy()[0]  # (S, 4, 4)

        S = pts_raw.shape[1]

        # Flatten and filter by confidence
        pts_flat = pts_raw.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        conf_threshold = np.percentile(conf_flat, self._conf_percentile)
        mask = conf_flat >= conf_threshold
        pred_pts = pts_flat[mask].astype(np.float32)

        # Align using Umeyama on keyframe camera centres vs GT poses
        from utils.pointcloud_eval import _umeyama_alignment

        pred_centres = camera_poses[:, :3, 3]  # (S, 3)
        gt_positions = np.stack([f.cam_pos for f in frames], axis=0)

        kfids = getattr(slam, "kfids", None)
        aligned = False

        if kfids is not None and len(kfids) >= S:
            kfids_trunc = kfids[:S]
            valid_kf = [i for i, fid in enumerate(kfids_trunc) if fid < len(gt_positions)]
            if len(valid_kf) >= 3:
                matched_pred = pred_centres[valid_kf]
                matched_gt = gt_positions[[kfids_trunc[i] for i in valid_kf]]
                R, t, scale = _umeyama_alignment(matched_pred, matched_gt)
                pred_pts = (scale * (pred_pts @ R.T) + t).astype(np.float32)
                camera_poses_aligned = camera_poses.copy()
                for i in range(S):
                    camera_poses_aligned[i, :3, 3] = scale * (camera_poses[i, :3, 3] @ R.T) + t
                camera_poses = camera_poses_aligned
                aligned = True
                if self._verbose:
                    print(f"[slamformer] Aligned via kfids: {len(valid_kf)} keyframes, scale={scale:.4f}")

        if not aligned:
            # Fallback: align on all camera centres (truncate to min(S, N))
            n_match = min(S, len(gt_positions))
            if n_match >= 3:
                R, t, scale = _umeyama_alignment(pred_centres[:n_match], gt_positions[:n_match])
                pred_pts = (scale * (pred_pts @ R.T) + t).astype(np.float32)
                if self._verbose:
                    print(f"[slamformer] Aligned via truncated centres: n={n_match}, scale={scale:.4f}")
            else:
                if self._verbose:
                    print(f"[slamformer] WARNING: only {n_match} centres, skipping alignment")

        # Build dummy depth/intrinsics arrays (SLAM-Former doesn't provide
        # per-frame depth in the same format as VGGT, but the evaluator
        # only uses them for debug dumps)
        H = pts_raw.shape[2] if pts_raw.ndim >= 4 else 1
        W = pts_raw.shape[3] if pts_raw.ndim >= 5 else 1
        dummy_depth = np.zeros((S, H, W), dtype=np.float32)
        dummy_intrinsics = np.tile(np.eye(3, dtype=np.float32), (S, 1, 1))

        if self._verbose:
            print(f"[slamformer] Predicted {pred_pts.shape[0]} points from {S} keyframes")

        return {
            "points": pred_pts,
            "extrinsics": camera_poses,
            "intrinsics": dummy_intrinsics,
            "depth": dummy_depth,
            "already_aligned": True,
        }

    @staticmethod
    def _empty_result():
        return {
            "points": np.zeros((0, 3), dtype=np.float32),
            "extrinsics": np.zeros((0, 4, 4), dtype=np.float32),
            "intrinsics": np.zeros((0, 3, 3), dtype=np.float32),
            "depth": np.zeros((0, 1, 1), dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "vggt": VGGTReconstructionModel,
    "slamformer": SLAMFormerReconstructionModel,
    "placeholder": PlaceholderReconstructionModel,
}


def get_reconstruction_model(
    cfg: PointCloudEvalConfig,
    device: torch.device,
    model_type: str | None = None,
    shared_slam_model=None,
) -> ReconstructionModel:
    if model_type is None:
        model_type = getattr(cfg, "reconstruction_model", "placeholder")
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown reconstruction model '{model_type}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    if shared_slam_model is not None and cls is SLAMFormerReconstructionModel:
        return cls(cfg, device, shared_slam_model=shared_slam_model)
    return cls(cfg, device)
