"""Step 2: Incrementally reconstruct with VGGT and evaluate against GT.

For each prefix k = 1, 2, ..., N:
  - Runs VGGT on the first k rendered RGB images
  - Aligns predicted point cloud to GT world frame (Umeyama on camera centres)
  - Computes three metrics:
      1. Voxel coverage  – fraction of GT-occupied voxels covered by prediction
      2. Chamfer distance – bidirectional nearest-neighbour distance (metres)
      3. Depth error      – voxel-normalized L2 after scale alignment

Usage:
    # First run render_from_poses.py, then:
    python baselines/reconstruct_and_eval.py \\
        --pose_json baselines/poses/random_procthor_ProcTHOR-Test-21_run_000.json \\
        [--rendered_dir baselines/rendered] \\
        [--recon_dir   baselines/reconstructions] \\
        [--results_dir baselines/results] \\
        [--data_dir    /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data] \\
        [--model_cache_dir /home/ghr/fs/Junyi/data/proj/model_weights] \\
        [--voxel_size  0.05] \\
        [--max_points  20000] \\
        [--depth_max   10.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from src.envs.voxel_carving import depth_to_world_points
from src.utils.pointcloud_eval import (
    chamfer_distance,
    _camera_centers_from_extrinsics,
    _umeyama_alignment,
)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _write_ply(points: np.ndarray, path: Path) -> None:
    """Write (N, 3) float32 point cloud to an ASCII PLY file."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _write_xyzrgb_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Write (N, 3) XYZ + (N, 3) uint8 RGB point cloud to an ASCII PLY file."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors).reshape(-1, 3)
    if cols.dtype != np.uint8:
        cols = (np.clip(cols, 0.0, 1.0) * 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def _write_coverage_ply(
    covered_mask: np.ndarray,
    observable_mask: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    path: Path,
) -> None:
    """Write a coloured PLY showing covered vs uncovered surface voxels.

    Green  (0, 200, 0)   – surface voxel covered by the predicted point cloud
    Red    (200, 0, 0)   – surface voxel not yet covered
    Each voxel is represented by its centre point.
    """
    covered_surf   = covered_mask & observable_mask
    uncovered_surf = observable_mask & ~covered_mask

    # Convert voxel indices → world coordinates (voxel centre)
    def _vox_centers(mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask).astype(np.float32)   # (M, 3)
        return origin + (idx + 0.5) * pitch

    pts_cov   = _vox_centers(covered_surf)
    pts_uncov = _vox_centers(uncovered_surf)

    pts    = np.concatenate([pts_cov, pts_uncov], axis=0)
    colors = np.concatenate([
        np.tile(np.array([[0, 200, 0]], dtype=np.uint8), (len(pts_cov),   1)),
        np.tile(np.array([[200, 0, 0]], dtype=np.uint8), (len(pts_uncov), 1)),
    ], axis=0)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def _align(
    pred_points: np.ndarray,
    pred_extr: np.ndarray,
    gt_positions: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Umeyama alignment using predicted vs GT camera centres.

    Falls back to translation-only when fewer than 3 frames are available.
    Returns (aligned_points, scale).
    """
    pred_centres = _camera_centers_from_extrinsics(pred_extr)

    if pred_centres.shape[0] < 2 or pred_centres.shape != gt_positions.shape:
        delta = gt_positions.mean(0) - pred_centres.mean(0)
        return pred_points + delta.reshape(1, 3), 1.0

    if pred_centres.shape[0] < 3:
        delta = gt_positions.mean(0) - pred_centres.mean(0)
        return pred_points + delta.reshape(1, 3), 1.0

    R, t, s = _umeyama_alignment(pred_centres, gt_positions)
    aligned = (s * (pred_points @ R.T)) + t.reshape(1, 3)
    return aligned, float(s)


def _align_first_frame(
    pred_points: np.ndarray,
    pred_extr: np.ndarray,
    gt_position_0: np.ndarray,
    gt_quat_wxyz_0: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Align by matching VGGT frame-0 pose to GT frame-0 pose.

    T_align = T_gt0_c2w @ T_pred0_w2c
    i.e.  VGGT-world → VGGT-cam0 → GT-world.
    Scale is fixed to 1.0 (VGGT predicts metric depth).
    Works for any k ≥ 1.
    """
    from scipy.spatial.transform import Rotation

    # GT camera-0 → world (4×4)
    w, x, y, z = gt_quat_wxyz_0.tolist()
    R_c2w = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)  # xyzw
    T_gt_c2w = np.eye(4, dtype=np.float32)
    T_gt_c2w[:3, :3] = R_c2w
    T_gt_c2w[:3,  3] = gt_position_0

    # VGGT predicted frame-0: world-to-camera (4×4)
    T_pred_w2c = np.eye(4, dtype=np.float32)
    T_pred_w2c[:3, :] = pred_extr[0, :3, :]   # first row of extrinsics

    T_align = T_gt_c2w @ T_pred_w2c           # VGGT-world → GT-world
    R_a = T_align[:3, :3]
    t_a = T_align[:3,  3]
    aligned = (pred_points @ R_a.T) + t_a.reshape(1, 3)
    return aligned, 1.0


# ---------------------------------------------------------------------------
# GT point cloud
# ---------------------------------------------------------------------------

def _build_gt_pointcloud(
    depths: List[np.ndarray],
    positions: List[np.ndarray],
    quats: List[np.ndarray],
    intrinsics: np.ndarray,
    depth_max: float,
) -> np.ndarray:
    """Unproject GT depth maps to world-frame point cloud."""
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    parts = []
    for depth, pos, quat in zip(depths, positions, quats):
        pts, valid = depth_to_world_points(
            depth, pos, quat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            forward_axis="x+",
            depth_is_z=True,
            depth_max=depth_max,
            depth_max_is_no_hit=True,
            pose_is_camera_to_world=True,
        )
        if pts.shape[0] > 0:
            parts.append(pts)
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Voxel coverage
# ---------------------------------------------------------------------------

def _load_voxel_grid(h5_path: Path):
    """Load GT voxel grid from .voxels.h5. Returns (gt_grid, origin, pitch)."""
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        gt_grid = f["grid"][:].astype(bool)
        origin  = f["origin"][:].astype(np.float32)
        pitch   = float(f["pitch"][()])
    return gt_grid, origin, pitch


def _surface_voxels(gt_grid: np.ndarray) -> np.ndarray:
    """Return mask of occupied voxels adjacent to at least one free voxel (6-conn).

    Interior voxels (completely surrounded by occupied neighbours) are physically
    unobservable from any camera.  Only surface voxels should count toward the
    coverage denominator.
    """
    from scipy.ndimage import binary_dilation
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = True
    struct[1, 1, 0] = struct[1, 1, 2] = True  # ±x
    struct[1, 0, 1] = struct[1, 2, 1] = True  # ±y
    struct[0, 1, 1] = struct[2, 1, 1] = True  # ±z
    free_dilated = binary_dilation(~gt_grid, structure=struct)
    return gt_grid & free_dilated


def _mark_covered(
    pred_points: np.ndarray,
    covered_mask: np.ndarray,
    gt_grid: np.ndarray,
    origin: np.ndarray,
    pitch: float,
) -> None:
    """OR pred_points into covered_mask in-place (same shape as gt_grid)."""
    pts = np.asarray(pred_points, dtype=np.float32)
    idx = np.floor((pts - origin) / pitch).astype(np.int64)
    gx, gy, gz = gt_grid.shape
    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < gx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < gy) &
        (idx[:, 2] >= 0) & (idx[:, 2] < gz)
    )
    idx = idx[valid]
    if idx.shape[0] > 0:
        covered_mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _voxel_coverage_fallback(
    covered_pts_all: np.ndarray,
    gt_points_full: np.ndarray,
    voxel_size: float,
) -> float:
    """Fallback coverage: voxelise cumulative pred cloud vs full GT cloud."""
    if gt_points_full.shape[0] == 0 or covered_pts_all.shape[0] == 0:
        return 0.0
    min_xyz = np.minimum(covered_pts_all.min(0), gt_points_full.min(0))
    gt_set   = set(map(tuple, np.floor((gt_points_full  - min_xyz) / voxel_size).astype(np.int64)))
    pred_set = set(map(tuple, np.floor((covered_pts_all - min_xyz) / voxel_size).astype(np.int64)))
    return len(pred_set & gt_set) / max(len(gt_set), 1)


# ---------------------------------------------------------------------------
# Depth error
# ---------------------------------------------------------------------------

def _depth_error_l2_voxel(
    pred_depth: np.ndarray,           # (T, H_v, W_v) in VGGT scale
    gt_depths: List[np.ndarray],      # T × (H_gt, W_gt) in metres
    scale: float,
    gt_positions: List[np.ndarray],   # T × (3,) camera world positions
    gt_quats: List[np.ndarray],       # T × (4,) quaternion wxyz
    intrinsics: np.ndarray,           # (3, 3)
    vox_origin: Optional[np.ndarray], # (3,) voxel grid origin; None → return nan
    vox_pitch: Optional[float],       # voxel size in metres; None → return nan
    vox_shape: Optional[Tuple],       # (Gx, Gy, Gz); None → return nan
) -> float:
    """Voxel-normalized squared depth error.

    For each voxel j that is seen by at least one image:
      per_image_mean[i] = mean over pixels in frame i mapping to voxel j of (pred−gt)²
      voxel_error[j]    = mean of per_image_mean[i] over all frames covering voxel j

    Final metric = mean of voxel_error[j] over all covered voxels.

    Each image contributes equally per voxel (regardless of pixel count), and
    each voxel contributes equally to the final score.
    Returns nan when no voxel grid is available or no valid pixels exist.
    """
    if vox_origin is None or vox_pitch is None or vox_shape is None:
        return float("nan")

    from collections import defaultdict
    # voxel_1d_key → list of per-image mean squared errors
    voxel_image_errors: Dict[int, List[float]] = defaultdict(list)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    Gx, Gy, Gz = vox_shape

    T = min(pred_depth.shape[0], len(gt_depths))
    for i in range(T):
        gd = gt_depths[i].astype(np.float32)          # (H_gt, W_gt)
        H_gt, W_gt = gd.shape[:2]

        # Resize predicted depth to GT resolution and apply scale
        pd_t = torch.from_numpy(pred_depth[i].astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pd = F.interpolate(pd_t, size=(H_gt, W_gt), mode="bilinear",
                           align_corners=False).squeeze().numpy() * scale

        # Unproject valid GT depth pixels to world-frame 3-D points.
        # valid_out: bool mask (H_gt, W_gt) where gd > 0 and finite.
        pts_world, valid_out = depth_to_world_points(
            gd, gt_positions[i], gt_quats[i],
            fx=fx, fy=fy, cx=cx, cy=cy,
            forward_axis="x+",
            depth_is_z=True,
            pose_is_camera_to_world=True,
        )
        if pts_world.shape[0] == 0:
            continue

        # Keep only pixels where predicted depth is also valid.
        ys_gt, xs_gt = np.where(valid_out)
        pd_vals = pd[ys_gt, xs_gt]
        pd_ok = (pd_vals > 0) & np.isfinite(pd_vals)
        if pd_ok.sum() == 0:
            continue

        pts_world = pts_world[pd_ok]
        gt_vals   = gd[ys_gt[pd_ok], xs_gt[pd_ok]]
        pd_vals   = pd_vals[pd_ok]

        # Map world points → voxel indices
        vox_idx = np.floor((pts_world - vox_origin) / vox_pitch).astype(np.int64)
        in_bounds = (
            (vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < Gx) &
            (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < Gy) &
            (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < Gz)
        )
        vox_idx = vox_idx[in_bounds]
        sq_err  = (pd_vals[in_bounds] - gt_vals[in_bounds]) ** 2  # per-pixel L2²

        if vox_idx.shape[0] == 0:
            continue

        # Encode 3-D voxel index as a single int for grouping
        keys_1d = vox_idx[:, 0] * (Gy * Gz) + vox_idx[:, 1] * Gz + vox_idx[:, 2]
        unique_keys, inverse = np.unique(keys_1d, return_inverse=True)
        for k_idx, key in enumerate(unique_keys):
            mean_sq = float(sq_err[inverse == k_idx].mean())
            voxel_image_errors[int(key)].append(mean_sq)

    if not voxel_image_errors:
        return float("nan")

    # Per-voxel score = average across images that cover it
    voxel_errors = [float(np.mean(errs)) for errs in voxel_image_errors.values()]
    return float(np.mean(voxel_errors))


# ---------------------------------------------------------------------------
# VGGT inference
# ---------------------------------------------------------------------------

def _run_vggt(
    rgb_paths: List[str],
    model: VGGT,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    images = load_and_preprocess_images(rgb_paths).to(device)
    amp_on = device.type == "cuda"
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            preds = model(images)

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        preds["pose_enc"], images.shape[-2:]
    )
    depth = preds["depth"].squeeze(0)
    depth_np = depth.detach().float().cpu().numpy()
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]   # (T, H, W)

    extr_np = extrinsics.squeeze(0).detach().float().cpu().numpy()
    intr_np = intrinsics.squeeze(0).detach().float().cpu().numpy()

    pts = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)
    pts = pts.reshape(-1, 3)
    valid = depth_np.reshape(-1) > 0

    # Colors from preprocessed input images: (1, T, 3, H, W) → (T*H*W, 3) float [0,1]
    img_np = preds["images"].squeeze(0).detach().float().cpu().numpy()  # (T, 3, H, W)
    colors = img_np.transpose(0, 2, 3, 1).reshape(-1, 3)               # (T*H*W, 3)

    return {
        "points":     pts[valid].astype(np.float32),
        "colors":     np.clip(colors[valid], 0.0, 1.0).astype(np.float32),
        "depth":      depth_np,
        "extrinsics": extr_np,
        "intrinsics": intr_np,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental VGGT reconstruction + GT metric evaluation."
    )
    parser.add_argument("--pose_json",   required=True)
    parser.add_argument("--rendered_dir",  default="baselines/rendered")
    parser.add_argument("--recon_dir",     default="baselines/reconstructions")
    parser.add_argument("--results_dir",   default="baselines/results")
    parser.add_argument("--data_dir",
                        default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data")
    parser.add_argument("--model_id",      default="facebook/VGGT-1B")
    parser.add_argument("--model_cache_dir",
                        default="/home/ghr/fs/Junyi/data/proj/model_weights")
    parser.add_argument("--voxel_size",  type=float, default=0.05,
                        help="Fallback voxel size (m) when .voxels.h5 is absent.")
    parser.add_argument("--max_points",  type=int,   default=20000,
                        help="Subsampling cap for Chamfer distance.")
    parser.add_argument("--depth_max",   type=float, default=10.0,
                        help="Max valid GT depth (m); beyond this is treated as no-hit.")
    parser.add_argument("--align_method", default="umeyama",
                        choices=["umeyama", "first_frame"],
                        help="Point-cloud alignment method: "
                             "'umeyama' (Umeyama similarity on camera centres, ≥3 frames) or "
                             "'first_frame' (rigid align via frame-0 pose, works for k≥1).")
    parser.add_argument("--coverage_dilation", type=int, default=1,
                        help="Dilate covered_mask by this many voxels before computing coverage "
                             "(tolerance for sparse point clouds). 0 = exact hits only.")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    # --- load pose JSON ---
    with open(args.pose_json) as f:
        pose_data = json.load(f)

    scene_id: str = pose_data["scene_id"]
    method:   str = pose_data.get("method",  "unknown")
    run_id:   str = pose_data.get("run_id",  "run_000")
    max_views: int = int(pose_data["max_views"])

    safe_scene_id = scene_id.replace("/", "__")
    rendered_dir = Path(args.rendered_dir) / safe_scene_id / f"{method}_{run_id}"
    meta_path = rendered_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Rendered data not found: {meta_path}\n"
            "Run render_from_poses.py first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    intrinsics = np.array(meta["intrinsics"], dtype=np.float32)
    frames = meta["frames"][:max_views]
    N = len(frames)

    rgb_paths  = [str(rendered_dir / fr["rgb"])   for fr in frames]
    gt_depths  = [np.load(str(rendered_dir / fr["depth"])) for fr in frames]
    gt_positions = [np.array(fr["position"],        dtype=np.float32) for fr in frames]
    gt_quats     = [np.array(fr["quaternion_wxyz"], dtype=np.float32) for fr in frames]

    # GT voxel grid for coverage – load once
    h5_path: Optional[Path] = Path(args.data_dir) / f"{scene_id}.glb.voxels.h5"
    if h5_path.exists():
        gt_grid, vox_origin, vox_pitch = _load_voxel_grid(h5_path)
        observable_mask = _surface_voxels(gt_grid)
        n_gt_voxels      = int(gt_grid.sum())
        n_surface_voxels = int(observable_mask.sum())
        print(f"[eval] GT voxels: {n_gt_voxels} total, {n_surface_voxels} surface (observable)")
        covered_mask = np.zeros_like(gt_grid, dtype=bool)  # cumulative
        gt_points_full = None
    else:
        h5_path = None
        gt_grid = observable_mask = vox_origin = vox_pitch = covered_mask = None
        n_gt_voxels = n_surface_voxels = None
        gt_points_full = _build_gt_pointcloud(
            gt_depths, gt_positions, gt_quats, intrinsics, args.depth_max
        )
        covered_pts_list: List[np.ndarray] = []  # cumulative for fallback

    # Output directories
    recon_base  = Path(args.recon_dir)  / f"{method}_{safe_scene_id}_{run_id}"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load VGGT
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[eval] Loading VGGT on {device} …")
    model = VGGT.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir or None,
        local_files_only=True,
    ).to(device)
    model.eval()
    print(f"[eval] Starting incremental eval for {N} views …\n")

    all_metrics: List[Dict] = []

    for k in range(1, N + 1):
        print(f"[eval] k={k}/{N} …", end=" ", flush=True)
        recon_dir = recon_base / f"views_{k:04d}"
        recon_dir.mkdir(parents=True, exist_ok=True)

        # --- VGGT ---
        pred = _run_vggt(rgb_paths[:k], model, device)

        # --- Alignment ---
        gt_pos_k = np.stack(gt_positions[:k], axis=0)
        if args.align_method == "first_frame":
            pred_aligned, scale = _align_first_frame(
                pred["points"], pred["extrinsics"],
                gt_positions[0], gt_quats[0],
            )
        else:
            pred_aligned, scale = _align(pred["points"], pred["extrinsics"], gt_pos_k)

        # --- GT point cloud for this subset ---
        gt_pts_k = _build_gt_pointcloud(
            gt_depths[:k], gt_positions[:k], gt_quats[:k], intrinsics, args.depth_max
        )

        # --- Save reconstruction ---
        np.save(str(recon_dir / "pred_points_raw.npy"),     pred["points"])
        np.save(str(recon_dir / "pred_points_aligned.npy"), pred_aligned)
        np.save(str(recon_dir / "pred_depth.npy"),          pred["depth"])
        np.save(str(recon_dir / "pred_extrinsics.npy"),     pred["extrinsics"])
        np.save(str(recon_dir / "pred_intrinsics.npy"),     pred["intrinsics"])
        np.save(str(recon_dir / "gt_points_k.npy"),         gt_pts_k)
        # Align colors the same way as points (same valid mask was applied in _run_vggt)
        aligned_colors = (pred["colors"] * 255).astype(np.uint8)
        _write_xyzrgb_ply(pred_aligned, aligned_colors, recon_dir / "pred_points_aligned.ply")
        _write_ply(gt_pts_k, recon_dir / "gt_points_k.ply")

        # --- Metric 1: Voxel coverage (cumulative, surface voxels only) ---
        # Coverage measures exploration quality: use GT depth point cloud,
        # not VGGT predictions (whose alignment errors cause stripe artefacts).
        if gt_grid is not None:
            _mark_covered(gt_pts_k, covered_mask, gt_grid, vox_origin, vox_pitch)
            # Optional dilation: expand covered_mask by N voxels to tolerate
            # point-cloud sparsity at range (raw hits accumulate; dilation is
            # applied only when computing the metric and writing the PLY).
            if args.coverage_dilation > 0:
                from scipy.ndimage import binary_dilation
                struct = np.zeros((3, 3, 3), dtype=bool)
                struct[1, 1, 1] = True
                struct[1, 1, 0] = struct[1, 1, 2] = True
                struct[1, 0, 1] = struct[1, 2, 1] = True
                struct[0, 1, 1] = struct[2, 1, 1] = True
                covered_for_metric = covered_mask.copy()
                for _ in range(args.coverage_dilation):
                    covered_for_metric = binary_dilation(covered_for_metric, structure=struct)
            else:
                covered_for_metric = covered_mask
            vc = float((covered_for_metric & observable_mask).sum()) / n_surface_voxels if n_surface_voxels else 0.0
            _write_coverage_ply(
                covered_for_metric, observable_mask, vox_origin, vox_pitch,
                recon_dir / "voxel_coverage.ply",
            )
        else:
            covered_pts_list.append(pred_aligned)
            covered_pts_all = np.concatenate(covered_pts_list, axis=0)
            vc = _voxel_coverage_fallback(covered_pts_all, gt_points_full, args.voxel_size)

        # --- Metric 2: Chamfer distance ---
        cd = chamfer_distance(pred_aligned, gt_pts_k, max_points=args.max_points)

        # --- Metric 3: Depth error ---
        depth_l2_voxel = _depth_error_l2_voxel(
            pred["depth"],
            gt_depths[:k],
            scale,
            gt_positions[:k],
            gt_quats[:k],
            intrinsics,
            vox_origin,
            vox_pitch,
            gt_grid.shape if gt_grid is not None else None,
        )

        metrics_k = {
            "k":               k,
            "voxel_coverage":  round(vc, 6),
            "chamfer_distance": round(float(cd) if cd is not None else float("nan"), 6),
            "depth_l2_voxel":  round(depth_l2_voxel, 6),
            "scale":           round(scale, 6),
            "pred_points":     int(pred_aligned.shape[0]),
            "gt_points_k":     int(gt_pts_k.shape[0]),
            "n_surface_voxels": n_surface_voxels,
            "n_gt_voxels":      n_gt_voxels,
        }
        all_metrics.append(metrics_k)

        print(
            f"coverage={vc:.3f}  chamfer={cd:.4f}  "
            f"depth_l2_voxel={depth_l2_voxel:.6f}"
        )

    # --- Save results ---
    result_path = results_dir / f"{method}_{safe_scene_id}_{run_id}.json"
    result_out = {
        "scene_id":      scene_id,
        "method":        method,
        "run_id":        run_id,
        "max_views":     max_views,
        "metrics_by_k":  all_metrics,
    }
    with open(result_path, "w") as f:
        json.dump(result_out, f, indent=2)

    print(f"\n[eval] Results → {result_path}")


if __name__ == "__main__":
    main()