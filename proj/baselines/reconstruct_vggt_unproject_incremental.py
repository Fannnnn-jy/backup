"""Incrementally reconstruct rendered views with VGGT depth unprojection.

For each prefix k = 1, 2, ..., N:
  - Runs VGGT on the first k rendered RGB frames
  - Unprojects VGGT-predicted depth into a point cloud
  - Aligns the reconstructed cloud to world coordinates using the notebook method
  - Computes coverage from GT point clouds / GT voxel grid
  - Leaves depth loss as a TODO placeholder

Usage:
    python baselines/reconstruct_vggt_unproject_incremental.py \
        --pose_json baselines/poses/random_procthor_ProcTHOR-Test-21_run_000.json \
        [--rendered_dir baselines/rendered] \
        [--recon_dir baselines/reconstructions_unproject] \
        [--results_dir baselines/results] \
        [--data_dir /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data] \
        [--model_cache_dir /home/ghr/fs/Junyi/data/proj/model_weights]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from src.envs.voxel_carving import depth_to_world_points
from src.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat


def _write_ply(points: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _write_xyzrgb_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors).reshape(-1, 3)
    if cols.dtype != np.uint8:
        cols = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"Points/colors count mismatch: {pts.shape[0]} vs {cols.shape[0]}")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def _write_alignment_debug(debug: Dict[str, Any], out_dir: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    for key, value in debug.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value.astype(np.float32, copy=False)
        elif isinstance(value, np.generic):
            meta[key] = value.item()
        else:
            meta[key] = value
    with (out_dir / "alignment_debug.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if arrays:
        np.savez(str(out_dir / "alignment_debug_arrays.npz"), **arrays)


def _align_with_first_camera_pose_notebook(
    pred_points: np.ndarray,
    pred_extr_0: np.ndarray,
    gt_position_0: np.ndarray,
    gt_quat_wxyz_0: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    pred_extr_0 = np.asarray(pred_extr_0, dtype=np.float32).reshape(3, 4)
    pred_R_world_to_cam0 = pred_extr_0[:, :3]
    pred_t_world_to_cam0 = pred_extr_0[:, 3]

    quat = torch.from_numpy(np.asarray(gt_quat_wxyz_0, dtype=np.float32)).reshape(1, 4)
    quat = convert_camera_frame_orientation_convention(quat, "world", "ros")
    gt_R_cam0_to_world = matrix_from_quat(quat).squeeze(0).cpu().numpy().astype(np.float32)
    gt_t_cam0_to_world = np.asarray(gt_position_0, dtype=np.float32).reshape(1, 3)

    # `unproject_depth_map_to_point_map` returns points in VGGT's predicted world frame.
    # To place them in GT world coordinates:
    #   1) predicted world -> predicted frame-0 camera using pred extrinsic
    #   2) frame-0 camera -> GT world using the GT frame-0 pose
    points_cam0 = (pts @ pred_R_world_to_cam0.T) + pred_t_world_to_cam0.reshape(1, 3)
    aligned = (points_cam0 @ gt_R_cam0_to_world) + gt_t_cam0_to_world

    composed_rotation = pred_R_world_to_cam0.T @ gt_R_cam0_to_world
    composed_translation = (pred_t_world_to_cam0.reshape(1, 3) @ gt_R_cam0_to_world) + gt_t_cam0_to_world
    debug = {
        "method": "pred_world_to_gt_world_via_first_camera",
        "mode": "rigid_first_frame_composed",
        "scale": 1.0,
        "pred_rotation_world_to_cam0": pred_R_world_to_cam0,
        "pred_translation_world_to_cam0": pred_t_world_to_cam0,
        "gt_rotation_cam0_to_world": gt_R_cam0_to_world,
        "gt_translation_cam0_to_world": gt_t_cam0_to_world.reshape(-1),
        "rotation": composed_rotation,
        "translation": composed_translation.reshape(-1),
    }
    return aligned, debug


def _build_gt_pointcloud(
    depths: List[np.ndarray],
    positions: List[np.ndarray],
    quats: List[np.ndarray],
    intrinsics: np.ndarray,
    depth_max: float,
) -> np.ndarray:
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    parts = []
    for depth, pos, quat in zip(depths, positions, quats):
        pts, _ = depth_to_world_points(
            depth,
            pos,
            quat,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
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


def _load_voxel_grid(h5_path: Path):
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        gt_grid = f["grid"][:].astype(bool)
        origin = f["origin"][:].astype(np.float32)
        pitch = float(f["pitch"][()])
    return gt_grid, origin, pitch


def _surface_voxels(gt_grid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = True
    struct[1, 1, 0] = struct[1, 1, 2] = True
    struct[1, 0, 1] = struct[1, 2, 1] = True
    struct[0, 1, 1] = struct[2, 1, 1] = True
    free_dilated = binary_dilation(~gt_grid, structure=struct)
    return gt_grid & free_dilated


def _mark_covered(
    pred_points: np.ndarray,
    covered_mask: np.ndarray,
    gt_grid: np.ndarray,
    origin: np.ndarray,
    pitch: float,
) -> None:
    pts = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return
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


def _write_coverage_ply(
    covered_mask: np.ndarray,
    observable_mask: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    path: Path,
) -> None:
    covered_surface = covered_mask & observable_mask
    uncovered_surface = observable_mask & ~covered_mask

    def _vox_centers(mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask).astype(np.float32)
        return origin + (idx + 0.5) * pitch

    pts_cov = _vox_centers(covered_surface)
    pts_uncov = _vox_centers(uncovered_surface)
    pts = np.concatenate([pts_cov, pts_uncov], axis=0)
    colors = np.concatenate(
        [
            np.tile(np.array([[0, 200, 0]], dtype=np.uint8), (len(pts_cov), 1)),
            np.tile(np.array([[200, 0, 0]], dtype=np.uint8), (len(pts_uncov), 1)),
        ],
        axis=0,
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def _voxel_coverage_fallback(
    covered_pts: np.ndarray,
    gt_points_full: np.ndarray,
    voxel_size: float,
) -> float:
    if gt_points_full.shape[0] == 0 or covered_pts.shape[0] == 0:
        return 0.0
    min_xyz = np.minimum(covered_pts.min(0), gt_points_full.min(0))
    gt_set = set(map(tuple, np.floor((gt_points_full - min_xyz) / voxel_size).astype(np.int64)))
    pred_set = set(map(tuple, np.floor((covered_pts - min_xyz) / voxel_size).astype(np.int64)))
    return len(pred_set & gt_set) / max(len(gt_set), 1)


def _compute_depth_loss_todo(*_: Any, **__: Any) -> float:
    # TODO: implement depth loss once the target definition is finalized.
    return float("nan")


def _run_vggt_unproject(
    rgb_paths: List[str],
    model: VGGT,
    device: torch.device,
    *,
    preprocess_mode: str,
    depth_min: float,
    amp_dtype: str,
) -> Dict[str, np.ndarray]:
    images = load_and_preprocess_images(rgb_paths, mode=preprocess_mode).to(device)
    amp_enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=dtype):
            predictions = model(images)

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    depth = predictions["depth"].squeeze(0)
    depth_np = depth.detach().float().cpu().numpy()
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]

    extr_np = extrinsics.squeeze(0).detach().float().cpu().numpy()
    intr_np = intrinsics.squeeze(0).detach().float().cpu().numpy()
    img_np = predictions["images"].squeeze(0).detach().float().cpu().numpy()

    point_maps = unproject_depth_map_to_point_map(depth_np[..., None], extr_np, intr_np)
    points = point_maps.reshape(-1, 3)
    colors = img_np.transpose(0, 2, 3, 1).reshape(-1, 3)
    valid = np.isfinite(depth_np.reshape(-1)) & (depth_np.reshape(-1) > float(depth_min))

    return {
        "point_map": point_maps.astype(np.float32),
        "points": points[valid].astype(np.float32),
        "colors": np.clip(colors[valid], 0.0, 1.0).astype(np.float32),
        "depth": depth_np.astype(np.float32),
        "extrinsics": extr_np.astype(np.float32),
        "intrinsics": intr_np.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental VGGT reconstruction from rendered RGB/depth using depth unprojection."
    )
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--rendered_dir", default="baselines/rendered")
    parser.add_argument("--recon_dir", default="baselines/reconstructions_unproject")
    parser.add_argument("--results_dir", default="baselines/results")
    parser.add_argument(
        "--data_dir",
        default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data",
    )
    parser.add_argument("--model_id", default="facebook/VGGT-1B")
    parser.add_argument(
        "--model_cache_dir",
        default="/home/ghr/fs/Junyi/data/proj/model_weights",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=True,
        help="Load VGGT only from local cache.",
    )
    parser.add_argument(
        "--allow_download",
        dest="local_files_only",
        action="store_false",
        help="Allow downloading model weights if absent.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess_mode", default="crop", choices=["crop", "pad"])
    parser.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--depth_min", type=float, default=1e-8)
    parser.add_argument("--depth_max", type=float, default=10.0)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--coverage_dilation", type=int, default=1)
    args = parser.parse_args()

    with open(args.pose_json, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    scene_id: str = pose_data["scene_id"]
    method: str = pose_data.get("method", "unknown")
    run_id: str = pose_data.get("run_id", "run_000")
    max_views: int = int(pose_data["max_views"])

    safe_scene_id = scene_id.replace("/", "__")
    rendered_dir = Path(args.rendered_dir) / safe_scene_id / f"{method}_{run_id}"
    meta_path = rendered_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Rendered data not found: {meta_path}\nRun render_from_poses.py first."
        )

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    intrinsics = np.array(meta["intrinsics"], dtype=np.float32)
    frames = meta["frames"][:max_views]
    num_frames = len(frames)
    if num_frames == 0:
        raise RuntimeError(f"No rendered frames found in {meta_path}")

    rgb_paths = [str(rendered_dir / frame["rgb"]) for frame in frames]
    gt_depths = [np.load(str(rendered_dir / frame["depth"])) for frame in frames]
    gt_positions = [np.array(frame["position"], dtype=np.float32) for frame in frames]
    gt_quats = [np.array(frame["quaternion_wxyz"], dtype=np.float32) for frame in frames]

    h5_path = Path(args.data_dir) / f"{scene_id}.glb.voxels.h5"
    if h5_path.exists():
        gt_grid, vox_origin, vox_pitch = _load_voxel_grid(h5_path)
        observable_mask = _surface_voxels(gt_grid)
        n_gt_voxels = int(gt_grid.sum())
        n_surface_voxels = int(observable_mask.sum())
        covered_mask = np.zeros_like(gt_grid, dtype=bool)
        gt_points_full = None
        print(f"[recon] GT voxels: {n_gt_voxels} total, {n_surface_voxels} surface")
    else:
        gt_grid = None
        observable_mask = None
        vox_origin = None
        vox_pitch = None
        n_gt_voxels = None
        n_surface_voxels = None
        covered_mask = None
        gt_points_full = _build_gt_pointcloud(
            gt_depths,
            gt_positions,
            gt_quats,
            intrinsics,
            args.depth_max,
        )
        print("[recon] GT voxel grid not found, falling back to point-cloud voxelization for coverage")

    recon_base = Path(args.recon_dir) / f"{method}_{safe_scene_id}_{run_id}"
    results_dir = Path(args.results_dir)
    recon_base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[recon] Loading VGGT on {device} ...")
    model = VGGT.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir or None,
        local_files_only=bool(args.local_files_only),
    ).to(device)
    model.eval()

    metrics_by_k: List[Dict[str, Any]] = []
    print(f"[recon] Starting incremental reconstruction for {num_frames} views ...")

    for k in range(1, num_frames + 1):
        recon_dir = recon_base / f"views_{k:04d}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        print(f"[recon] k={k}/{num_frames} ...", end=" ", flush=True)

        pred = _run_vggt_unproject(
            rgb_paths[:k],
            model,
            device,
            preprocess_mode=args.preprocess_mode,
            depth_min=args.depth_min,
            amp_dtype=args.amp_dtype,
        )
        pred_aligned, align_debug = _align_with_first_camera_pose_notebook(
            pred["points"],
            pred["extrinsics"][0],
            gt_positions[0],
            gt_quats[0],
        )

        gt_pts_k = _build_gt_pointcloud(
            gt_depths[:k],
            gt_positions[:k],
            gt_quats[:k],
            intrinsics,
            args.depth_max,
        )

        np.save(str(recon_dir / "pred_point_map_raw.npy"), pred["point_map"])
        np.save(str(recon_dir / "pred_points_raw.npy"), pred["points"])
        np.save(str(recon_dir / "pred_points_aligned.npy"), pred_aligned)
        np.save(str(recon_dir / "pred_colors.npy"), pred["colors"])
        np.save(str(recon_dir / "pred_depth.npy"), pred["depth"])
        np.save(str(recon_dir / "pred_extrinsics.npy"), pred["extrinsics"])
        np.save(str(recon_dir / "pred_intrinsics.npy"), pred["intrinsics"])
        np.save(str(recon_dir / "gt_points_k.npy"), gt_pts_k)
        _write_xyzrgb_ply(pred["points"], pred["colors"], recon_dir / "pred_points_raw_rgb.ply")
        _write_xyzrgb_ply(pred_aligned, pred["colors"], recon_dir / "pred_points_aligned_rgb.ply")
        _write_ply(gt_pts_k, recon_dir / "gt_points_k.ply")
        _write_alignment_debug(align_debug, recon_dir)

        if gt_grid is not None:
            _mark_covered(gt_pts_k, covered_mask, gt_grid, vox_origin, vox_pitch)
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
            coverage = (
                float((covered_for_metric & observable_mask).sum()) / n_surface_voxels
                if n_surface_voxels
                else 0.0
            )
            _write_coverage_ply(
                covered_for_metric,
                observable_mask,
                vox_origin,
                vox_pitch,
                recon_dir / "voxel_coverage.ply",
            )
        else:
            coverage = _voxel_coverage_fallback(gt_pts_k, gt_points_full, args.voxel_size)

        depth_loss = _compute_depth_loss_todo(
            pred_depth=pred["depth"],
            gt_depths=gt_depths[:k],
            pred_intrinsics=pred["intrinsics"],
            pred_extrinsics=pred["extrinsics"],
        )

        metrics_k = {
            "k": k,
            "voxel_coverage": round(coverage, 6),
            "depth_loss": depth_loss,
            "depth_loss_status": "TODO",
            "pred_points_raw": int(pred["points"].shape[0]),
            "pred_points_aligned": int(pred_aligned.shape[0]),
            "gt_points_k": int(gt_pts_k.shape[0]),
            "n_surface_voxels": n_surface_voxels,
            "n_gt_voxels": n_gt_voxels,
            "recon_dir": str(recon_dir),
        }
        metrics_by_k.append(metrics_k)

        with (recon_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_k, f, indent=2)

        print(f"coverage={coverage:.3f}  pred_points={pred_aligned.shape[0]}  depth_loss=TODO")

    result_path = results_dir / f"{method}_{safe_scene_id}_{run_id}_vggt_unproject.json"
    result = {
        "scene_id": scene_id,
        "method": method,
        "run_id": run_id,
        "max_views": max_views,
        "coverage_source": "gt_pointcloud",
        "alignment_method": "pred_world_to_gt_world_via_first_camera",
        "depth_loss_status": "TODO",
        "metrics_by_k": metrics_by_k,
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[recon] Results saved to {result_path}")


if __name__ == "__main__":
    main()
