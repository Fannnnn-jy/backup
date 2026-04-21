from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None

from utils.reconstruction_model import ReconstructionModel, get_reconstruction_model

try:
    from envs.voxel_carving import intrinsics_from_fov
    from envs.maniskill_active_mapping_env import _camera_pose
    from envs.debug_camera import dump_camera_buffer_stats, print_camera_debug
except ModuleNotFoundError:  # pragma: no cover - fallback for src.* package layout
    from src.envs.voxel_carving import intrinsics_from_fov
    from src.envs.maniskill_active_mapping_env import _camera_pose
    from src.envs.debug_camera import dump_camera_buffer_stats, print_camera_debug

try:
    from utils.math import convert_camera_frame_orientation_convention, matrix_from_quat
except ModuleNotFoundError:  # pragma: no cover - fallback for src.* package layout
    from src.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat

try:
    from learning.config import PointCloudEvalConfig
except ModuleNotFoundError:  # pragma: no cover - fallback for src.* package layout
    from src.learning.config import PointCloudEvalConfig


@dataclass
class FrameData:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    cam_pos: np.ndarray
    cam_quat_wxyz: np.ndarray


@dataclass
class PointCloudCallbackOutput:
    reward_delta: Optional[torch.Tensor] = None
    scalars: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.scalars is None:
            self.scalars = {}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _maybe_normalize_rgb(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return arr
    out = np.asarray(arr)
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    max_val = float(np.nanmax(out)) if np.isfinite(out).any() else 1.0
    if max_val > 1.0 + 1e-3:
        out = out / 255.0
    return out


def _resize_square(rgb: torch.Tensor, target: int) -> torch.Tensor:
    if target <= 0:
        return rgb
    if rgb.shape[-1] == target and rgb.shape[-2] == target:
        return rgb
    return F.interpolate(rgb, size=(target, target), mode="bilinear", align_corners=False)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _camera_basis(forward_axis: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if forward_axis == "z+":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if forward_axis == "z-":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    if forward_axis == "x+":
        # Match project convention used by _depth_to_pointcloud: image-down maps to -Z.
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    if forward_axis == "x-":
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])
    if forward_axis == "y+":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    if forward_axis == "y-":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    raise ValueError(f"Unsupported forward axis: {forward_axis}")


def depth_to_world_points(
    depth: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_axis: str = "x+",
    depth_is_z: bool = True,
    depth_scale: float = 1.0,
    depth_max: float = 0.0,
    pose_is_camera_to_world: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth * float(depth_scale)
    if depth_max > 0:
        depth = np.minimum(depth, float(depth_max))
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    height, width = depth.shape[:2]
    right, down, forward = _camera_basis(forward_axis)
    right = right.astype(np.float32)
    down = down.astype(np.float32)
    forward = forward.astype(np.float32)

    u = np.arange(width, dtype=np.float32) + 0.5
    v = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(u, v)
    x = (float(cx) - xx) / float(fx)
    y = (yy - float(cy)) / float(fy)
    dirs = x[..., None] * right + y[..., None] * down + forward
    if not depth_is_z:
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        norms = np.where(norms <= 1e-6, 1.0, norms)
        dirs = dirs / norms

    rot = _quat_wxyz_to_matrix(np.asarray(camera_quat_wxyz, dtype=np.float32))
    if not pose_is_camera_to_world:
        rot = rot.T
    cam_pos = np.asarray(camera_position, dtype=np.float32).reshape(1, 1, 3)
    points_cam = dirs * depth[..., None]
    points_world = (points_cam.reshape(-1, 3) @ rot.T) + cam_pos.reshape(1, 3)
    return points_world, valid


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extr = np.asarray(extrinsics, dtype=np.float32)
    R = extr[:, :3, :3]
    t = extr[:, :3, 3]
    centers = -np.einsum("bij,bj->bi", R.transpose(0, 2, 1), t)
    return centers


def _align_with_first_camera_pose_notebook(pred_points: np.ndarray, first_frame: FrameData) -> np.ndarray:
    """Match notebook alignment: canonical/first-cam frame -> world via first GT camera pose."""
    pts = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    quat = torch.from_numpy(np.asarray(first_frame.cam_quat_wxyz, dtype=np.float32)).reshape(1, 4)
    quat = convert_camera_frame_orientation_convention(quat, "world", "ros")
    R_cam2world = matrix_from_quat(quat).squeeze(0).cpu().numpy().astype(np.float32)
    t = np.asarray(first_frame.cam_pos, dtype=np.float32).reshape(1, 3)
    return (pts @ R_cam2world) + t


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Expected src/dst as (N,3) arrays of the same shape.")
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var_src = np.sum(src_centered**2) / float(src.shape[0])
    scale = float(np.sum(S) / var_src) if var_src > 1e-6 else 1.0
    t = mean_dst - scale * (R @ mean_src)
    return R, t, scale


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
    return pts[idx]


def chamfer_distance(a: np.ndarray, b: np.ndarray, *, max_points: int = 20000) -> Optional[float]:
    a = _sample_points(a, max_points)
    b = _sample_points(b, max_points)
    if a.size == 0 or b.size == 0:
        return None
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_t = a_t.to(device=device, dtype=torch.float32)
    b_t = b_t.to(device=device, dtype=torch.float32)
    chunk = 2048
    mins_a = []
    for i in range(0, a_t.shape[0], chunk):
        dist = torch.cdist(a_t[i : i + chunk], b_t)
        mins_a.append(dist.min(dim=1).values)
    mins_a = torch.cat(mins_a, dim=0)
    mins_b = []
    for i in range(0, b_t.shape[0], chunk):
        dist = torch.cdist(b_t[i : i + chunk], a_t)
        mins_b.append(dist.min(dim=1).values)
    mins_b = torch.cat(mins_b, dim=0)
    return float(0.5 * (mins_a.mean() + mins_b.mean()).item())


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB image in HxWx3+, got shape {arr.shape}.")
    arr = arr[..., :3]
    max_val = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if max_val > 1.0 + 1e-3:
        arr = np.clip(arr, 0.0, 255.0)
    else:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    valid = np.isfinite(arr) & (arr > 0.0)
    vis = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(valid):
        return vis
    min_v = float(np.min(arr[valid]))
    max_v = float(np.max(arr[valid]))
    if max_v <= min_v + 1e-6:
        vis[valid] = 255
        return vis
    norm = (arr - min_v) / (max_v - min_v)
    vis = np.clip(norm, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def _write_depth_debug(depth: np.ndarray, npy_path: Path, vis_png_path: Path) -> None:
    arr = np.asarray(depth, dtype=np.float32)
    np.save(str(npy_path), arr)
    vis = _depth_to_vis(arr)
    if cv2 is not None:
        cv2.imwrite(str(vis_png_path), vis)
        return
    if Image is not None:
        Image.fromarray(vis, mode="L").save(str(vis_png_path))


def _write_rgb_png(rgb: np.ndarray, path: Path) -> None:
    rgb_u8 = _as_uint8_rgb(rgb)
    if cv2 is not None:
        cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        return
    if Image is not None:
        Image.fromarray(rgb_u8, mode="RGB").save(str(path))


def _write_xyz_ply(points: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _write_debug_readme(debug_root: Path) -> None:
    debug_root.mkdir(parents=True, exist_ok=True)
    readme_path = debug_root / "README.md"
    if readme_path.exists():
        return
    text = """# VGGT Debug Dump

This folder stores one episode dump generated by point-cloud evaluation debug mode.

## Layout

- `episode_XXXX_env_YY/`
  - `raw_frames/`: all captured frames before frame selection.
    - `frame_XXXX_rgb.png`: captured RGB image.
    - `frame_XXXX_gt_depth.npy`: GT depth array (float32).
    - `frame_XXXX_gt_depth_vis.png`: visualization of GT depth.
    - `frame_XXXX_intrinsics.npy`: camera intrinsics matrix.
    - `frame_XXXX_cam_pos.npy`: camera position in world coordinates.
    - `frame_XXXX_cam_quat_wxyz.npy`: camera orientation quaternion (wxyz).
  - `selected_frames/`: frames actually passed to VGGT after max_frames selection.
    - `sel_XXXX_src_XXXX_rgb.png`: selected RGB image.
    - `sel_XXXX_src_XXXX_gt_depth.npy`: matching GT depth.
    - `sel_XXXX_src_XXXX_gt_depth_vis.png`: matching GT depth visualization.
    - `predictions/`: VGGT outputs and derived point clouds.
    - `sel_XXXX_src_XXXX_pred_depth.npy`: predicted depth per selected frame.
    - `sel_XXXX_src_XXXX_pred_depth_vis.png`: predicted depth visualization.
    - `pred_points_raw.npy` / `pred_points_raw.ply`: raw predicted point cloud from VGGT depth + VGGT pose/intrinsics.
    - `pred_points_aligned.npy` / `pred_points_aligned.ply`: predicted point cloud after optional GT-based alignment.
    - `gt_points.npy` / `gt_points.ply`: GT point cloud from sensor depth + pose.
    - `pred_extrinsics.npy`: predicted camera extrinsics.
    - `pred_intrinsics.npy`: predicted camera intrinsics.
  - `eval_log.json`: metrics and config used for this debug dump.
"""
    readme_path.write_text(text, encoding="utf-8")


class EpisodePointCloudEvaluator:
    def __init__(self, cfg: PointCloudEvalConfig, shared_slam_model=None) -> None:
        self.cfg = cfg
        self._device = _resolve_device(cfg.device)
        self._model: Optional[ReconstructionModel] = None
        self._shared_slam_model = shared_slam_model
        self._buffers: Dict[int, List[FrameData]] = {}
        self._episode_counter = 0
        self._debug_dump_done = False
        self._buffer_debug_dumped_envs: set[int] = set()
        # Resolve debug dir early (before any model chdir)
        _dd = str(getattr(cfg, "debug_dump_dir", "") or "debug/pointcloud_eval")
        self._debug_dir_resolved = Path(_dd).resolve()

    def _ensure_model(self) -> ReconstructionModel:
        if self._model is not None:
            return self._model
        self._model = get_reconstruction_model(self.cfg, self._device, shared_slam_model=self._shared_slam_model)
        return self._model

    def _maybe_stride(self, step_idx: int) -> bool:
        stride = max(1, int(self.cfg.frame_stride))
        return (step_idx % stride) == 0

    def _append_frame(self, env_index: int, frame: FrameData) -> None:
        buf = self._buffers.setdefault(env_index, [])
        if self.cfg.max_buffer_frames > 0 and len(buf) >= int(self.cfg.max_buffer_frames):
            buf.pop(0)
        buf.append(frame)

    def _maybe_print_buffers_once(self, base_env, env_index: int, step_idx: int, cameras: List[object]) -> None:
        if not bool(getattr(self.cfg, "debug_print_buffers_once", False)):
            return
        if env_index in self._buffer_debug_dumped_envs:
            return
        print_camera_debug(base_env, step_idx, note=f"env={env_index}")
        for cam_idx, cam in enumerate(cameras):
            dump_camera_buffer_stats(cam, note=f"env={env_index}/cam={cam_idx}")
        self._buffer_debug_dumped_envs.add(int(env_index))

    def record_step(self, base_env, env_index: int, step_idx: int) -> None:
        if not self.cfg.enabled:
            return
        if not self._maybe_stride(step_idx):
            return
        if base_env is None:
            return
        cameras = getattr(base_env, "_cameras", None) or []
        if not cameras:
            return
        rgb_images = []
        depth_images = []
        if hasattr(base_env, "_render_rgb_images"):
            try:
                rgb_images = base_env._render_rgb_images()
            except Exception:
                rgb_images = []
        if hasattr(base_env, "_render_depth_images"):
            try:
                depth_images = base_env._render_depth_images()
            except Exception:
                depth_images = []
        self._maybe_print_buffers_once(base_env, env_index, step_idx, cameras)
        if not rgb_images or not depth_images:
            return
        count = min(len(rgb_images), len(depth_images), len(cameras))
        if count <= 0:
            return
        for idx in range(count):
            cam = cameras[idx]
            pose = _camera_pose(cam)
            if pose is None:
                continue
            cam_pos, cam_quat = pose
            if hasattr(cam, "get_intrinsic_matrix"):
                K = cam.get_intrinsic_matrix()
                K = np.asarray(K, dtype=np.float32)
            else:
                h, w = depth_images[idx].shape[:2]
                fov_rad = np.radians(float(getattr(base_env.cfg, "maniskill_camera_fov", 90.0)))
                fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_rad)
                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            frame = FrameData(
                rgb=_maybe_normalize_rgb(rgb_images[idx]),
                depth=np.asarray(depth_images[idx], dtype=np.float32),
                intrinsics=K,
                cam_pos=np.asarray(cam_pos, dtype=np.float32),
                cam_quat_wxyz=np.asarray(cam_quat, dtype=np.float32),
            )
            self._append_frame(env_index, frame)
            # Notify incremental models (e.g. SLAM-Former) about the new frame
            model = self._ensure_model()
            model.on_frame(env_index, frame)

    def _select_frames(self, frames: List[FrameData]) -> Tuple[List[FrameData], np.ndarray]:
        if self.cfg.max_frames <= 0 or len(frames) <= self.cfg.max_frames:
            idx = np.arange(len(frames), dtype=np.int64)
            return frames, idx
        idx = np.linspace(0, len(frames) - 1, int(self.cfg.max_frames))
        idx = np.round(idx).astype(np.int64)
        return [frames[int(i)] for i in idx], idx

    def _compute_predicted_pointcloud(self, frames: List[FrameData]) -> Dict[str, np.ndarray]:
        model = self._ensure_model()
        return model.predict_pointcloud(frames, self.cfg, self._device)

    def _resolve_debug_dir(self) -> Path:
        return self._debug_dir_resolved

    def _should_dump_debug(self) -> bool:
        return bool(getattr(self.cfg, "debug_dump_first_episode", False)) and not self._debug_dump_done

    def _dump_debug_artifacts(
        self,
        *,
        env_index: int,
        raw_frames: List[FrameData],
        selected_frames: List[FrameData],
        selected_indices: np.ndarray,
        pred_depth: np.ndarray,
        pred_points_raw: np.ndarray,
        pred_points_aligned: np.ndarray,
        gt_points: np.ndarray,
        pred_extr: np.ndarray,
        pred_intr: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        debug_root = self._resolve_debug_dir()
        _write_debug_readme(debug_root)
        episode_dir = debug_root / f"episode_{self._episode_counter:04d}_env_{int(env_index):02d}"
        raw_dir = episode_dir / "raw_frames"
        selected_dir = episode_dir / "selected_frames"
        pred_dir = episode_dir / "predictions"
        for d in (raw_dir, selected_dir, pred_dir):
            d.mkdir(parents=True, exist_ok=True)

        for raw_idx, frame in enumerate(raw_frames):
            _write_rgb_png(frame.rgb, raw_dir / f"frame_{raw_idx:04d}_rgb.png")
            _write_depth_debug(
                frame.depth,
                raw_dir / f"frame_{raw_idx:04d}_gt_depth.npy",
                raw_dir / f"frame_{raw_idx:04d}_gt_depth_vis.png",
            )
            np.save(str(raw_dir / f"frame_{raw_idx:04d}_intrinsics.npy"), frame.intrinsics.astype(np.float32))
            np.save(str(raw_dir / f"frame_{raw_idx:04d}_cam_pos.npy"), frame.cam_pos.astype(np.float32))
            np.save(str(raw_dir / f"frame_{raw_idx:04d}_cam_quat_wxyz.npy"), frame.cam_quat_wxyz.astype(np.float32))

        for local_idx, frame in enumerate(selected_frames):
            src_idx = int(selected_indices[local_idx])
            _write_rgb_png(frame.rgb, selected_dir / f"sel_{local_idx:04d}_src_{src_idx:04d}_rgb.png")
            _write_depth_debug(
                frame.depth,
                selected_dir / f"sel_{local_idx:04d}_src_{src_idx:04d}_gt_depth.npy",
                selected_dir / f"sel_{local_idx:04d}_src_{src_idx:04d}_gt_depth_vis.png",
            )

        for local_idx in range(min(len(selected_frames), int(pred_depth.shape[0]))):
            src_idx = int(selected_indices[local_idx])
            _write_depth_debug(
                pred_depth[local_idx],
                pred_dir / f"sel_{local_idx:04d}_src_{src_idx:04d}_pred_depth.npy",
                pred_dir / f"sel_{local_idx:04d}_src_{src_idx:04d}_pred_depth_vis.png",
            )

        np.save(str(pred_dir / "pred_points_raw.npy"), np.asarray(pred_points_raw, dtype=np.float32))
        np.save(str(pred_dir / "pred_points_aligned.npy"), np.asarray(pred_points_aligned, dtype=np.float32))
        np.save(str(pred_dir / "gt_points.npy"), np.asarray(gt_points, dtype=np.float32))
        np.save(str(pred_dir / "pred_extrinsics.npy"), np.asarray(pred_extr, dtype=np.float32))
        np.save(str(pred_dir / "pred_intrinsics.npy"), np.asarray(pred_intr, dtype=np.float32))
        _write_xyz_ply(pred_points_raw, pred_dir / "pred_points_raw.ply")
        _write_xyz_ply(pred_points_aligned, pred_dir / "pred_points_aligned.ply")
        _write_xyz_ply(gt_points, pred_dir / "gt_points.ply")
        for stale in ("pred_points.npy", "pred_points.ply"):
            stale_path = pred_dir / stale
            if stale_path.exists():
                stale_path.unlink()

        debug_log = {
            "env_index": int(env_index),
            "episode_counter": int(self._episode_counter),
            "raw_frame_count": int(len(raw_frames)),
            "selected_frame_count": int(len(selected_frames)),
            "selected_indices": np.asarray(selected_indices, dtype=np.int64),
            "metrics": metrics,
            "config": {
                "model_id": self.cfg.model_id,
                "vggt_input_size": int(self.cfg.vggt_input_size),
                "min_frames": int(self.cfg.min_frames),
                "frame_stride": int(self.cfg.frame_stride),
                "max_frames": int(self.cfg.max_frames),
                "align_mode": str(self.cfg.align_mode),
                "distance_mode": str(self.cfg.distance_mode),
                "reward_enabled": bool(self.cfg.reward_enabled),
                "reward_weight": float(self.cfg.reward_weight),
                "reward_clip": float(self.cfg.reward_clip),
            },
        }
        with (episode_dir / "eval_log.json").open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(debug_log), f, ensure_ascii=True, indent=2)
        print(f"[PointCloudEval] Dumped debug artifacts to {episode_dir}")

    def _compute_gt_pointcloud(self, frames: List[FrameData]) -> np.ndarray:
        points_list = []
        for frame in frames:
            K = frame.intrinsics
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            depth = frame.depth
            if self.cfg.depth_max > 0 and np.isfinite(depth).any():
                max_obs = float(np.nanmax(depth))
                if max_obs <= 1.0 + 1e-3:
                    depth = depth * float(self.cfg.depth_max)
            pts, valid = depth_to_world_points(
                depth,
                frame.cam_pos,
                frame.cam_quat_wxyz,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                forward_axis=str(self.cfg.forward_axis),
                depth_is_z=bool(self.cfg.depth_is_z),
                depth_scale=float(self.cfg.depth_scale),
                depth_max=float(self.cfg.depth_max),
                pose_is_camera_to_world=True,
            )
            if pts.size == 0:
                continue
            points_list.append(pts[valid.reshape(-1)])
        if not points_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(points_list, axis=0)

    def _align_pred_to_gt(self, pred_points: np.ndarray, pred_extr: np.ndarray, frames: List[FrameData]) -> np.ndarray:
        mode = str(self.cfg.align_mode).lower()
        if mode == "none":
            return pred_points
        if mode in {"notebook", "first_cam", "first_camera", "first_frame"}:
            if not frames:
                return pred_points
            return _align_with_first_camera_pose_notebook(pred_points, frames[0])
        pred_centers = _camera_centers_from_extrinsics(pred_extr)
        gt_centers = np.stack([f.cam_pos for f in frames], axis=0)
        if pred_centers.shape != gt_centers.shape or pred_centers.shape[0] < 2:
            return pred_points
        if mode == "translate" or pred_centers.shape[0] < 3:
            delta = gt_centers.mean(axis=0) - pred_centers.mean(axis=0)
            return pred_points + delta.reshape(1, 3)
        R, t, s = _umeyama_alignment(pred_centers, gt_centers)
        return (s * (pred_points @ R.T)) + t.reshape(1, 3)

    def evaluate_episode(self, env_index: int) -> Optional[Dict[str, float]]:
        raw_frames = list(self._buffers.get(env_index, []))
        if len(raw_frames) < int(self.cfg.min_frames):
            self._buffers[env_index] = []
            return None
        model = self._ensure_model()
        # Incremental models (e.g. SLAM-Former) already consumed all frames
        # via on_frame(); pass raw_frames so they can use GT poses for alignment.
        # Batch models (e.g. VGGT) use the selected subset.
        is_incremental = hasattr(model, "on_frame") and model.on_frame.__func__ is not ReconstructionModel.on_frame
        if is_incremental:
            frames = raw_frames
            selected_indices = np.arange(len(raw_frames), dtype=np.int64)
        else:
            frames, selected_indices = self._select_frames(raw_frames)
        pred = self._compute_predicted_pointcloud(frames)
        pred_points = np.asarray(pred["points"], dtype=np.float32)
        pred_extr = np.asarray(pred["extrinsics"], dtype=np.float32)
        pred_intr = np.asarray(pred["intrinsics"], dtype=np.float32)
        pred_depth = np.asarray(pred["depth"], dtype=np.float32)
        gt_points = self._compute_gt_pointcloud(frames)
        pred_points_raw = pred_points
        # Incremental models may handle alignment internally (e.g. SLAM-Former
        # does Umeyama on keyframe centres); skip double-alignment in that case.
        if pred.get("already_aligned", False):
            pred_points_aligned = pred_points_raw
        else:
            pred_points_aligned = self._align_pred_to_gt(pred_points_raw, pred_extr, frames)
        if str(self.cfg.distance_mode).lower() == "chamfer":
            dist = chamfer_distance(pred_points_aligned, gt_points, max_points=int(self.cfg.max_points))
        else:
            dist = chamfer_distance(pred_points_aligned, gt_points, max_points=int(self.cfg.max_points))
        loss = float(dist) if dist is not None else float("nan")
        metrics = {
            "pc_eval/loss": loss,
            "pc_eval/chamfer": loss,
            "pc_eval/pred_points": float(pred_points_aligned.shape[0]),
            "pc_eval/gt_points": float(gt_points.shape[0]),
            "pc_eval/frames": float(len(frames)),
        }
        if self._should_dump_debug():
            self._dump_debug_artifacts(
                env_index=env_index,
                raw_frames=raw_frames,
                selected_frames=frames,
                selected_indices=selected_indices,
                pred_depth=pred_depth,
                pred_points_raw=pred_points_raw,
                pred_points_aligned=pred_points_aligned,
                gt_points=gt_points,
                pred_extr=pred_extr,
                pred_intr=pred_intr,
                metrics=metrics,
            )
            self._debug_dump_done = True
        self._episode_counter += 1
        self._buffers[env_index] = []
        # Reset incremental model state for this env so next episode starts fresh
        model = self._ensure_model()
        model.on_episode_reset(env_index)
        return metrics


class PointCloudEpisodeCallback:
    """Episode-level point-cloud evaluation callback."""

    def __init__(
        self,
        cfg: PointCloudEvalConfig,
        *,
        num_envs: int,
        get_base_env: Callable[[int], object | None],
        shared_slam_model=None,
    ) -> None:
        self.cfg = cfg
        self._num_envs = max(1, int(num_envs))
        self._get_base_env = get_base_env
        self._global_step = 0
        self._evaluator = EpisodePointCloudEvaluator(cfg, shared_slam_model=shared_slam_model) if cfg.enabled else None

    @property
    def enabled(self) -> bool:
        return self._evaluator is not None

    def on_step_begin(self, *, iteration: int, rollout_step: int) -> None:
        if self._evaluator is None:
            return
        for env_index in range(self._num_envs):
            base_env = self._get_base_env(env_index)
            if base_env is None:
                continue
            self._evaluator.record_step(base_env, env_index, self._global_step)
        self._global_step += 1

    def on_step_end(
        self,
        *,
        iteration: int,
        rollout_step: int,
        rewards: torch.Tensor,
        dones: torch.Tensor | np.ndarray,
        extras: dict,
    ) -> PointCloudCallbackOutput:
        if self._evaluator is None:
            return PointCloudCallbackOutput()
        done_mask = dones.detach().cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)
        done_mask = done_mask.reshape(-1).astype(bool)
        metrics_list: List[Dict[str, float]] = []
        reward_delta = torch.zeros_like(rewards) if bool(self.cfg.reward_enabled) else None
        for env_index in range(min(done_mask.shape[0], self._num_envs)):
            if not done_mask[env_index]:
                continue
            metrics = self._evaluator.evaluate_episode(env_index)
            if metrics is None:
                continue
            item = dict(metrics)
            item["pc_eval/env_index"] = float(env_index)
            metrics_list.append(item)
            if reward_delta is not None:
                loss = float(item.get("pc_eval/loss", float("nan")))
                if not np.isfinite(loss):
                    loss = float(self.cfg.reward_nan_fallback)
                reward = -float(self.cfg.reward_weight) * loss
                clip_value = float(self.cfg.reward_clip)
                if clip_value > 0.0:
                    reward = float(np.clip(reward, -clip_value, clip_value))
                reward_delta[env_index] += reward
        return PointCloudCallbackOutput(
            reward_delta=reward_delta,
            scalars=self.aggregate(metrics_list),
        )

    @staticmethod
    def aggregate(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        out: Dict[str, float] = {"pc_eval/episodes_done": float(len(metrics_list))}
        keys = [key for key in metrics_list[0].keys() if key != "pc_eval/env_index"]
        for key in keys:
            vals = []
            for item in metrics_list:
                value = float(item.get(key, float("nan")))
                if np.isfinite(value):
                    vals.append(value)
            if vals:
                out[key] = float(np.mean(vals))
        return out


# Backward compatibility for previous naming.
PointCloudEpisodeHook = PointCloudEpisodeCallback
