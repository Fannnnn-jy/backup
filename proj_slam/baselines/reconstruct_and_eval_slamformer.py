"""Incrementally reconstruct with SLAM-Former and evaluate against GT.

For each prefix k = 1, 2, ..., N:
  - Runs SLAM-Former on the first k rendered RGB images
  - Extracts the predicted point cloud from SLAM-Former's map
  - Aligns predicted point cloud to GT world frame with Umeyama on camera centres
  - Computes:
      1. Voxel coverage  – fraction of GT surface voxels covered by prediction
      2. Chamfer distance – bidirectional nearest-neighbour distance (metres)

Usage:
    python baselines/reconstruct_and_eval_slamformer.py \
        --pose_json baselines/poses/random/replicacad__apt_0_run_000.json \
        [--rendered_dir baselines/rendered] \
        [--recon_dir   baselines/reconstructions] \
        [--results_dir baselines/results] \
        [--data_dir    /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.envs.voxel_carving import depth_to_world_points
from src.utils.math import convert_camera_frame_orientation_convention
from src.utils.pointcloud_eval import (
    chamfer_distance,
    _camera_centers_from_extrinsics,
    _umeyama_alignment,
)


# ---------------------------------------------------------------------------
# SLAM-Former helpers
# ---------------------------------------------------------------------------

def _ensure_slamformer_importable(slam_root: Path) -> None:
    root = Path(slam_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SLAM-Former repo not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    croco = root / "src" / "croco"
    if croco.exists() and str(croco) not in sys.path:
        sys.path.insert(0, str(croco))


def _build_slamformer(slam_root: Path, ckpt_path: Path, target_size: int, device: torch.device):
    """Build a SLAM-Former SLAM instance."""
    prev_cwd = os.getcwd()
    os.chdir(str(slam_root))
    try:
        from slam.demo import SLAM
        slam = SLAM(
            outdir=tempfile.mkdtemp(prefix="slamformer_eval_"),
            ckpt_path=str(ckpt_path),
            target_size=target_size,
            retention_ratio=0.5,
            kf_th=0.1,
            save_gmem=False,
        )
    finally:
        os.chdir(prev_cwd)
    return slam


def _run_slamformer_incremental(slam, rgb_paths: List[str], k: int):
    """Feed first k images through SLAM-Former, extract points, colors, and poses.

    Returns (pred_points, pred_colors, pred_cam_poses) or None if extraction fails.
    pred_points: (N, 3)  pred_colors: (N, 3) uint8 RGB  pred_cam_poses: (K, 4, 4)
    """
    import cv2

    if k < 2:
        return None

    # Reset SLAM state for a fresh run
    prev_cwd = os.getcwd()
    slam_root = Path(slam.model.__class__.__module__).parent if hasattr(slam, 'outdir') else Path(".")
    # We need to be in SLAM-Former root for internal path resolution
    try:
        # Feed images
        for i in range(k):
            img = cv2.imread(rgb_paths[i])
            if img is None:
                print(f"[slamformer] failed to read image: {rgb_paths[i]}")
                return None
            slam.step(float(i), img)

        # Extract results from the current map
        if slam.map_opt is not None:
            map_to_use = slam.map_opt
        elif slam.map is not None:
            map_to_use = slam.map
        else:
            return None

        result = slam.model.extract(slam.maybe_to_cuda(map_to_use))
        pts = result['points'].cpu().numpy()  # (1, S, H, W, 3)
        conf = result['conf'].cpu().numpy()   # (1, S, H, W)
        camera_poses = result['camera_poses'].cpu().numpy()[0]  # (S, 4, 4)

        # Extract colors from stored frames (RGB float 0-1 tensor → RGB uint8)
        # Use [:S] not [-S:]: when map_opt has fewer keyframes than slam.frames,
        # map_opt corresponds to the first S keyframes, not the last S.
        S = pts.shape[1]
        colors_all = torch.stack(slam.frames[:S]).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()[:, ::-1]  # BGR→RGB
        colors_all = np.clip(colors_all * 255, 0, 255).astype(np.uint8)

        # Flatten and filter by confidence
        pts_flat = pts.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        conf_threshold = np.percentile(conf_flat, 15)
        mask = conf_flat >= conf_threshold
        pred_points = pts_flat[mask].astype(np.float32)
        pred_colors = np.ascontiguousarray(colors_all[mask])

        return pred_points, pred_colors, camera_poses
    except Exception as exc:
        print(f"[slamformer] extraction failed at k={k}: {exc}")
        return None
    finally:
        os.chdir(prev_cwd)


# ---------------------------------------------------------------------------
# GT pointcloud from depth
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
            pose_is_camera_to_world=True,
        )
        if pts.size > 0:
            pts = pts.reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            parts.append(pts)
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Voxel coverage (same as other eval scripts)
# ---------------------------------------------------------------------------

def _load_voxel_grid(h5_path: Path):
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        gt_grid = f["grid"][:].astype(bool)
        origin = f["origin"][:].astype(np.float32)
        pitch = float(f["pitch"][()])
    return gt_grid, origin, pitch


def _generate_voxel_grid(
    glb_path: Path,
    h5_path: Path,
    y_up: bool = True,
    scale: float = 1.0,
    voxel_max_dim: int = 128,
    inner_surface: bool = True,
    interior_seed: Tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> None:
    import math
    import trimesh

    scene = trimesh.load(str(glb_path), force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)

    if y_up:
        theta = math.radians(90.0)
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        mesh = mesh.copy()
        mesh.apply_transform(rot_x)
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(scale))

    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    max_len = float(max(size))
    pitch = max_len / voxel_max_dim if voxel_max_dim > 0 else 0.015625
    pitch = max(1e-4, float(pitch))

    vox = mesh.voxelized(pitch)
    grid = np.asarray(vox.matrix, dtype=bool)
    if hasattr(vox, "origin"):
        origin = np.asarray(vox.origin, dtype=np.float32)
    elif hasattr(vox, "transform"):
        origin = np.asarray(vox.transform, dtype=np.float32)[:3, 3]
    else:
        origin = np.zeros(3, dtype=np.float32)
    pitch_val = getattr(vox, "pitch", None)
    if pitch_val is None:
        pitch_val = getattr(vox, "scale", None)
    pitch_arr = np.asarray(pitch_val, dtype=np.float32).reshape(-1) if pitch_val is not None else np.array([pitch])
    pitch = float(np.mean(pitch_arr[:3])) if pitch_arr.size >= 3 else float(pitch_arr[0])

    if inner_surface:
        from src.envs.active_mapping_env import ActiveMappingEnv
        grid = ActiveMappingEnv._extract_inner_surface(grid, origin, pitch, interior_seed)

    import h5py
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("grid", data=grid.astype(np.uint8), compression="gzip", compression_opts=1)
        f.create_dataset("origin", data=origin)
        f.create_dataset("pitch", data=np.float32(pitch))
        f.attrs["inner_surface"] = bool(inner_surface)
    print(f"[voxel-gen] saved {h5_path} ({int(grid.sum())} voxels)")


def _surface_voxels(gt_grid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = True
    struct[1, 1, 0] = struct[1, 1, 2] = True
    struct[1, 0, 1] = struct[1, 2, 1] = True
    struct[0, 1, 1] = struct[2, 1, 1] = True
    free_dilated = binary_dilation(~gt_grid, structure=struct)
    return gt_grid & free_dilated


def _mark_covered(pred_points, covered_mask, gt_grid, origin, pitch):
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


def _voxel_coverage_fallback(covered_pts_all, gt_points_full, voxel_size):
    if gt_points_full.shape[0] == 0 or covered_pts_all.shape[0] == 0:
        return 0.0
    min_xyz = np.minimum(covered_pts_all.min(0), gt_points_full.min(0))
    gt_set = set(map(tuple, np.floor((gt_points_full - min_xyz) / voxel_size).astype(np.int64)))
    pred_set = set(map(tuple, np.floor((covered_pts_all - min_xyz) / voxel_size).astype(np.int64)))
    return len(pred_set & gt_set) / max(len(gt_set), 1)


def _write_ply(points: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.reshape(-1, 3).astype(np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def _write_xyzrgb_ply(points, colors, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.reshape(-1, 3).astype(np.float32)
    cols = (np.clip(colors.reshape(-1, 3), 0, 1) * 255).astype(np.uint8) if colors.max() <= 1.0 + 1e-3 else colors.reshape(-1, 3).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(pts, cols):
            f.write(p.tobytes())
            f.write(c.tobytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def main():
    torch.backends.cuda.enable_flash_sdp(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--rendered_dir", default="baselines/rendered")
    parser.add_argument("--recon_dir", default="baselines/reconstructions")
    parser.add_argument("--results_dir", default="baselines/results")
    parser.add_argument("--data_dir", default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data")
    parser.add_argument("--slam_root", default="/home/ghr/fs/Junyi/SLAM-Former")
    parser.add_argument("--ckpt_path", default="/home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model")
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coverage_dilation", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--depth_max", type=float, default=10.0)
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Fallback voxel size (m) when .voxels.h5 is absent.")
    parser.add_argument("--save_recon_every", type=int, default=0,
                        help="Save per-k reconstruction npy/ply every N steps. 0=never, 1=every step.")
    args = parser.parse_args()

    # --- load pose JSON ---
    with open(args.pose_json) as f:
        pose_data = json.load(f)

    scene_id = pose_data["scene_id"]
    method = pose_data.get("method", "unknown")
    run_id = pose_data.get("run_id", "run_000")
    max_views = int(pose_data["max_views"])

    safe_scene_id = scene_id.replace("/", "__")
    rendered_dir = Path(args.rendered_dir) / safe_scene_id / f"{method}_{run_id}"
    meta_path = rendered_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Rendered data not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    intrinsics = np.array(meta["intrinsics"], dtype=np.float32)
    frames = meta["frames"][:max_views]
    N = len(frames)

    rgb_paths = [str(rendered_dir / fr["rgb"]) for fr in frames]
    gt_depths = [np.load(str(rendered_dir / fr["depth"])) for fr in frames]
    gt_positions = [np.array(fr["position"], dtype=np.float32) for fr in frames]
    gt_quats = [np.array(fr["quaternion_wxyz"], dtype=np.float32) for fr in frames]

    # GT voxel grid
    h5_path: Optional[Path] = Path(args.data_dir) / f"{scene_id}.glb.voxels.h5"
    if not h5_path.exists():
        glb_path = Path(args.data_dir) / f"{scene_id}.glb"
        if glb_path.exists():
            print(f"[eval] .voxels.h5 not found, generating from {glb_path} …")
            _generate_voxel_grid(glb_path, h5_path)
    if h5_path.exists():
        gt_grid, vox_origin, vox_pitch = _load_voxel_grid(h5_path)
        observable_mask = _surface_voxels(gt_grid)
        n_gt_voxels = int(gt_grid.sum())
        n_surface_voxels = int(observable_mask.sum())
        print(f"[eval] GT voxels: {n_gt_voxels} total, {n_surface_voxels} surface (observable)")
        covered_mask = np.zeros_like(gt_grid, dtype=bool)
        gt_points_full = None
    else:
        h5_path = None
        gt_grid = observable_mask = vox_origin = vox_pitch = covered_mask = None
        n_surface_voxels = None
        gt_points_full = _build_gt_pointcloud(
            gt_depths, gt_positions, gt_quats, intrinsics, args.depth_max
        )
        covered_pts_list: List[np.ndarray] = []

    # Output directories
    recon_base = Path(args.recon_dir) / f"{method}_{safe_scene_id}_{run_id}"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build SLAM-Former
    slam_root = Path(args.slam_root).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    _ensure_slamformer_importable(slam_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[eval] Loading SLAM-Former on {device} …")

    print(f"[eval] Starting incremental eval for {N} views …")

    per_k_metrics: List[Dict] = []

    for k in range(1, N + 1):
        # Build fresh SLAM instance for each k
        os.chdir(str(slam_root))
        slam = _build_slamformer(slam_root, ckpt_path, args.target_size, device)

        result = _run_slamformer_incremental(slam, rgb_paths, k)

        if result is None:
            print(f"\n[eval] k={k}/{N} … skipped (SLAM-Former needs >= 2 images or extraction failed)")
            per_k_metrics.append({
                "k": k,
                "voxel_coverage": 0.0,
                "chamfer": float("nan"),
            })
            continue

        pred_points, pred_colors, pred_cam_poses = result

        # Build GT points for first k views
        gt_pts_k = _build_gt_pointcloud(
            gt_depths[:k], gt_positions[:k], gt_quats[:k], intrinsics, args.depth_max
        )

        # Align pred to GT via Umeyama on camera centres
        gt_centers = np.stack(gt_positions[:k], axis=0)
        pred_centers = pred_cam_poses[:, :3, 3]  # (k_kf, 3) - may be fewer than k
        n_align = min(gt_centers.shape[0], pred_centers.shape[0])
        if n_align >= 3:
            R_align, t_align, scale = _umeyama_alignment(
                pred_centers[:n_align], gt_centers[:n_align]
            )
            pred_aligned = (scale * (pred_points @ R_align.T) + t_align).astype(np.float32)
        elif n_align >= 1:
            offset = gt_centers[0] - pred_centers[0]
            pred_aligned = pred_points + offset
            scale = 1.0
        else:
            pred_aligned = pred_points
            scale = 1.0

        # Subsample for chamfer
        max_pts = args.max_points
        if pred_aligned.shape[0] > max_pts:
            idx = np.random.choice(pred_aligned.shape[0], max_pts, replace=False)
            pred_sub = pred_aligned[idx]
        else:
            pred_sub = pred_aligned
        if gt_pts_k.shape[0] > max_pts:
            idx = np.random.choice(gt_pts_k.shape[0], max_pts, replace=False)
            gt_sub = gt_pts_k[idx]
        else:
            gt_sub = gt_pts_k

        cd = chamfer_distance(pred_sub, gt_sub) if pred_sub.shape[0] > 0 and gt_sub.shape[0] > 0 else float("nan")

        # Coverage
        if gt_grid is not None:
            _mark_covered(gt_pts_k, covered_mask, gt_grid, vox_origin, vox_pitch)
            covered_for_metric = covered_mask.copy()
            if args.coverage_dilation > 0:
                from scipy.ndimage import binary_dilation
                struct = np.zeros((3, 3, 3), dtype=bool)
                struct[1, 1, 1] = True
                struct[1, 1, 0] = struct[1, 1, 2] = True
                struct[1, 0, 1] = struct[1, 2, 1] = True
                struct[0, 1, 1] = struct[2, 1, 1] = True
                for _ in range(args.coverage_dilation):
                    covered_for_metric = binary_dilation(covered_for_metric, structure=struct)
            vc = float((covered_for_metric & observable_mask).sum()) / n_surface_voxels if n_surface_voxels else 0.0
        else:
            covered_pts_list.append(gt_pts_k)
            covered_pts_all = np.concatenate(covered_pts_list, axis=0)
            vc = _voxel_coverage_fallback(covered_pts_all, gt_points_full, args.voxel_size)

        # Save artifacts
        _save_this_k = args.save_recon_every > 0 and k % args.save_recon_every == 0
        recon_dir: Optional[Path] = None
        if _save_this_k:
            recon_dir = recon_base / f"views_{k:04d}"
            recon_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(recon_dir / "pred_points_aligned_slamformer.npy"), pred_aligned)
            if not (recon_dir / "gt_points_k.npy").exists():
                np.save(str(recon_dir / "gt_points_k.npy"), gt_pts_k)
            _write_xyzrgb_ply(pred_aligned, pred_colors, recon_dir / "pred_points_aligned_slamformer.ply")
            if not (recon_dir / "gt_points_k.ply").exists():
                _write_ply(gt_pts_k, recon_dir / "gt_points_k.ply")

        print(f"\n[eval] k={k}/{N} … coverage={vc:.3f}  chamfer={cd:.4f}  scale={scale:.6f}")
        per_k_metrics.append({
            "k": k,
            "voxel_coverage": round(vc, 6),
            "chamfer": round(cd, 6) if np.isfinite(cd) else None,
            "scale": round(scale, 6) if isinstance(scale, float) else None,
        })

        # Cleanup SLAM state
        del slam
        torch.cuda.empty_cache()

    # Save results
    result_path = results_dir / f"{method}_{safe_scene_id}_{run_id}_slamformer.json"
    output = {
        "scene_id": scene_id,
        "method": method,
        "run_id": run_id,
        "model": "SLAM-Former",
        "num_views": N,
        "per_k": per_k_metrics,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[eval] Results saved to {result_path}")


if __name__ == "__main__":
    main()
