#!/usr/bin/env python3
"""Sample valid XYZ + yaw tuples for baseline batch experiments.

The sampled XYZ + yaw live in a Z-up experiment frame:
- x, y: horizontal plane
- z: vertical
- yaw: rotation around +Z

ActiveGS `planner.init_pose` currently follows the repo's existing OpenCV c2w
convention in a Y-up scene frame, so this script converts the full pose
(rotation + translation) from sampled Z-up coordinates into that Y-up init_pose
matrix. Pitch and roll remain fixed to 0.

Output TSV columns:
    sample_id    run_id    x    y    z    yaw_deg    init_pose_json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

STANDARD_GLB_VOXEL_PITCH = 2.0 / 128.0
STANDARD_GLB_VOXEL_MAX_DIM = 128
STANDARD_GLB_Y_UP = True
STANDARD_GLB_SCALE = 1.0

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.random_pose.sampler import VoxelGrid, _is_valid


def _axis_alignment_matrix(from_axis: str, to_axis: str) -> np.ndarray:
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    src = axes[from_axis.lower()]
    dst = axes[to_axis.lower()]
    if np.allclose(src, dst):
        rot = np.eye(3, dtype=np.float32)
    elif np.allclose(src, -dst):
        orth = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if np.allclose(np.abs(np.dot(orth, src)), 1.0):
            orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(src, orth)
        v = v / np.linalg.norm(v)
        rot = -np.eye(3, dtype=np.float32) + 2.0 * np.outer(v, v)
    else:
        v = np.cross(src, dst)
        s = np.linalg.norm(v)
        c = float(np.dot(src, dst))
        vx = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=np.float32,
        )
        rot = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = rot
    return tf


Z_UP_TO_Y_UP = _axis_alignment_matrix("z", "y")


def _compute_voxel_pitch(size: np.ndarray, pitch_arg: float, max_dim: int) -> float:
    if pitch_arg > 0:
        return float(pitch_arg)
    max_dim = max(8, int(max_dim))
    span = float(np.max(size)) if size.size else 1.0
    if span <= 0:
        span = 1.0
    return span / float(max_dim)


def _load_glb_mesh_for_voxels(glb_path: Path, y_up: bool, scale: float):
    import trimesh

    scene = trimesh.load(str(glb_path), force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        if len(scene.geometry) == 0:
            return None
        if hasattr(scene, "to_geometry"):
            mesh = scene.to_geometry()
        else:
            mesh = scene.dump(concatenate=True)

    if mesh.vertices.size == 0:
        return None

    if y_up:
        theta = math.radians(90.0)
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta), 0.0],
                [0.0, math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        mesh = mesh.copy()
        mesh.apply_transform(rot_x)
    if float(scale) != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(scale))
    return mesh


def _save_glb_voxel_cache(
    cache_path: Path,
    glb_path: Path,
    grid: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    *,
    y_up: bool,
    scale: float,
    voxel_pitch_arg: float,
    voxel_max_dim: int,
) -> None:
    import h5py

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    glb_mtime = os.path.getmtime(glb_path)
    glb_size = os.path.getsize(glb_path)

    with h5py.File(str(cache_path), "w") as f:
        f.create_dataset("grid", data=grid.astype(np.uint8), compression="gzip", compression_opts=1)
        f.create_dataset("origin", data=np.asarray(origin, dtype=np.float32))
        f.create_dataset("pitch", data=np.asarray(pitch, dtype=np.float32))
        f.attrs["glb_mtime"] = float(glb_mtime)
        f.attrs["glb_size"] = int(glb_size)
        f.attrs["glb_path"] = str(glb_path)
        f.attrs["y_up"] = bool(y_up)
        f.attrs["scale"] = float(scale)
        f.attrs["voxel_pitch_arg"] = float(voxel_pitch_arg)
        f.attrs["voxel_max_dim"] = int(voxel_max_dim)


def _build_glb_voxel_cache(
    glb_path: Path,
    cache_path: Path,
    *,
    y_up: bool,
    scale: float,
    voxel_pitch_arg: float,
    voxel_max_dim: int,
):
    mesh = _load_glb_mesh_for_voxels(glb_path, y_up=y_up, scale=scale)
    if mesh is None:
        raise ValueError(f"failed to load GLB mesh for voxelization: {glb_path}")

    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    pitch = _compute_voxel_pitch(size, voxel_pitch_arg, voxel_max_dim)
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
    pitch_arr = np.asarray(pitch_val, dtype=np.float32).reshape(-1) if pitch_val is not None else np.array([pitch], dtype=np.float32)
    if pitch_arr.size >= 3 and not np.allclose(pitch_arr[:3], pitch_arr[0]):
        print(f"[sample_xyz_pitch] non-uniform pitch detected: {pitch_arr[:3]}")
    pitch = float(np.mean(pitch_arr[:3])) if pitch_arr.size >= 3 else float(pitch_arr[0])

    if grid.size == 0:
        raise ValueError(f"empty voxel grid generated for {glb_path}")

    _save_glb_voxel_cache(
        cache_path,
        glb_path,
        grid,
        origin,
        pitch,
        y_up=y_up,
        scale=scale,
        voxel_pitch_arg=voxel_pitch_arg,
        voxel_max_dim=voxel_max_dim,
    )
    print(f"[sample_xyz_pitch] generated voxel cache: {cache_path}")
    return grid, origin, pitch


def _ensure_voxel_grid(scene_id: str, data_dir: str):
    h5_path = Path(data_dir) / f"{scene_id}.glb.voxels.h5"
    if h5_path.exists():
        return VoxelGrid.from_h5(h5_path), h5_path, False

    glb_path = Path(data_dir) / f"{scene_id}.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"neither voxel grid nor glb found for scene: {scene_id}")

    _build_glb_voxel_cache(
        glb_path,
        h5_path,
        y_up=STANDARD_GLB_Y_UP,
        scale=STANDARD_GLB_SCALE,
        voxel_pitch_arg=STANDARD_GLB_VOXEL_PITCH,
        voxel_max_dim=STANDARD_GLB_VOXEL_MAX_DIM,
    )
    return VoxelGrid.from_h5(h5_path), h5_path, True


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("zero-length vector cannot be normalized")
    return v / norm


def pose_from_xyz_yaw(x: float, y: float, z: float, yaw_deg: float) -> list[list[float]]:
    """Build an ActiveGS init_pose from sampled Z-up XYZ + yaw.

    The input pose is defined in a Z-up experiment frame.
    The returned matrix is converted into the Y-up OpenCV c2w convention used by
    the current ActiveGS init_pose configs.
    """
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    forward = _normalize(forward)

    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    right = _normalize(np.cross(world_down, forward))
    down = _normalize(np.cross(forward, right))

    pose_zup = np.eye(4, dtype=np.float32)
    pose_zup[:3, 0] = right
    pose_zup[:3, 1] = down
    pose_zup[:3, 2] = forward
    pose_zup[:3, 3] = np.array([x, y, z], dtype=np.float32)

    pose_yup = Z_UP_TO_Y_UP @ pose_zup
    return pose_yup.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample valid planner/world XYZ + yaw tuples from a scene voxel grid."
    )
    parser.add_argument("--scene_id", required=True, help="Scene ID without .glb suffix.")
    parser.add_argument(
        "--data_dir",
        default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data",
        help="Root containing <scene_id>.glb.voxels.h5",
    )
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x_min", type=float, default=-0.5)
    parser.add_argument("--x_max", type=float, default=0.5)
    parser.add_argument("--y_min", type=float, default=-0.5)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.3)
    parser.add_argument("--z_max", type=float, default=0.8)
    parser.add_argument("--xy_radius", type=int, default=6)
    parser.add_argument("--z_radius", type=int, default=3)
    parser.add_argument("--border_voxels", type=int, default=4)
    parser.add_argument(
        "--yaw_min_deg",
        type=float,
        default=0.0,
        help="Minimum sampled yaw angle in degrees around +Z.",
    )
    parser.add_argument(
        "--yaw_max_deg",
        type=float,
        default=360.0,
        help="Maximum sampled yaw angle in degrees around +Z.",
    )
    parser.add_argument("--max_attempts", type=int, default=2000)
    parser.add_argument("--run_prefix", default="run")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.add_argument("--scene_up_axis", default="y")
    parser.add_argument("--planner_up_axis", default="z")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vg, h5_path, generated_h5 = _ensure_voxel_grid(args.scene_id, args.data_dir)
    lo, hi = vg.bounds()
    x_lo = max(args.x_min, float(lo[0]))
    x_hi = min(args.x_max, float(hi[0] - vg.pitch))
    y_lo = max(args.y_min, float(lo[1]))
    y_hi = min(args.y_max, float(hi[1] - vg.pitch))
    z_lo = max(args.z_min, float(lo[2]))
    z_hi = min(args.z_max, float(hi[2] - vg.pitch))
    if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
        raise ValueError(
            "invalid sampling box after voxel-grid clamping: "
            f"x=[{x_lo:.4f}, {x_hi:.4f}] "
            f"y=[{y_lo:.4f}, {y_hi:.4f}] "
            f"z=[{z_lo:.4f}, {z_hi:.4f}]"
        )
    if generated_h5:
        print(f"[sample_xyz_pitch] using newly generated voxel-grid validity checks: {h5_path}")
    else:
        print(f"[sample_xyz_pitch] using voxel-grid validity checks: {h5_path}")

    rng = np.random.default_rng(args.seed)
    rows: list[tuple[int, str, float, float, float, float, str]] = []

    for sample_id in range(args.n_samples):
        placed = False
        for _ in range(args.max_attempts):
            pos = np.array(
                [
                    rng.uniform(x_lo, x_hi),
                    rng.uniform(y_lo, y_hi),
                    rng.uniform(z_lo, z_hi),
                ],
                dtype=np.float32,
            )
            if vg is not None and not _is_valid(
                pos,
                vg,
                args.xy_radius,
                args.z_radius,
                args.border_voxels,
            ):
                continue

            yaw_deg = float(rng.uniform(args.yaw_min_deg, args.yaw_max_deg))
            init_pose = pose_from_xyz_yaw(
                float(pos[0]), float(pos[1]), float(pos[2]), yaw_deg
            )
            run_id = f"{args.run_prefix}_{sample_id:03d}"
            rows.append(
                (
                    sample_id,
                    run_id,
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    yaw_deg,
                    json.dumps(init_pose, separators=(",", ":")),
                )
            )
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"failed to sample pose {sample_id} for {args.scene_id} after "
                f"{args.max_attempts} attempts"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("sample_id\trun_id\tx\ty\tz\tyaw_deg\tinit_pose_json\n")
        for row in rows:
            f.write(
                f"{row[0]}\t{row[1]}\t"
                f"{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}\t{row[6]}\n"
            )

    print(
        f"[sample_xyz_pitch] wrote {len(rows)} samples for {args.scene_id} to {out_path}"
    )


if __name__ == "__main__":
    main()
