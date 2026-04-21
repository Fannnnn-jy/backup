"""Entry script: generate random camera poses for a GLB scene and write to poses/<method>/.

Usage:
    python baselines/random_pose/generate_poses.py \
        --scene_id procthor/ProcTHOR-Test-21 \
        [--data_dir /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data] \
        [--poses_dir baselines/poses] \
        [--method random] \
        [--n_views 10] \
        [--run_id run_000] \
        [--seed 0] \
        [--init_pose_json '[[...],[...],[...],[...]]'] \
        [--fov_deg 90] --image_width 256 --image_height 256 \
        [--z_min 0.3] [--z_max 0.8]

Output:
    baselines/poses/<method>/{safe_scene_id}_{run_id}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.random_pose.sampler import (  # noqa: E402
    RandomSamplerConfig,
    VoxelGrid,
    _is_valid,
    sample_random_poses,
)


_Y_UP_WORLD_TO_Z_UP_WORLD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_JSON_CAMERA_FROM_OPENCV = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random camera poses within a voxel-validated GLB scene."
    )
    parser.add_argument(
        "--scene_id",
        required=True,
        help="Scene ID relative to data_dir, without .glb extension. "
        "Example: procthor/ProcTHOR-Test-21",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data",
        help="Root directory containing GLB and .voxels.h5 files.",
    )
    parser.add_argument(
        "--poses_dir",
        default="baselines/poses",
        help="Output root directory for pose JSON files.",
    )
    parser.add_argument(
        "--method",
        default="random",
        help="Method name used for output subdirectory and JSON metadata.",
    )
    parser.add_argument("--n_views", type=int, default=10)
    parser.add_argument("--run_id", default="run_000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init_pose_json",
        default=None,
        help="Optional ActiveGS planner.init_pose matrix encoded as a JSON string. "
        "When provided, it is converted into the first baseline view.",
    )
    parser.add_argument("--init_x", type=float, default=None)
    parser.add_argument("--init_y", type=float, default=None)
    parser.add_argument("--init_z", type=float, default=None)
    parser.add_argument("--init_yaw_deg", type=float, default=None)
    # Rendering parameters (written into the JSON for downstream scripts)
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--camera_near", type=float, default=0.01)
    parser.add_argument("--camera_far", type=float, default=10.0)
    # Sampling parameters
    parser.add_argument(
        "--z_min",
        type=float,
        default=0.3,
        help="Minimum camera height above scene floor (metres).",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=0.8,
        help="Maximum camera height above scene floor (metres).",
    )
    parser.add_argument(
        "--xy_radius",
        type=int,
        default=6,
        help="XY collision neighbourhood radius (voxels, ~9 cm at pitch=0.015625 m).",
    )
    parser.add_argument(
        "--z_radius",
        type=int,
        default=3,
        help="Z collision neighbourhood radius (voxels).",
    )
    parser.add_argument(
        "--border_voxels",
        type=int,
        default=4,
        help="Required clearance to voxel-grid borders.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=2000,
        help="Max resampling attempts per view.",
    )
    return parser.parse_args()


def _active_gs_init_pose_to_view(init_pose_json: str) -> dict:
    c2w_yup = np.asarray(json.loads(init_pose_json), dtype=np.float32)
    if c2w_yup.shape != (4, 4):
        raise ValueError(
            "--init_pose_json must decode to a 4x4 camera-to-world matrix; "
            f"got shape {tuple(c2w_yup.shape)}"
        )

    c2w_zup = _Y_UP_WORLD_TO_Z_UP_WORLD @ c2w_yup
    rotation = c2w_zup[:3, :3] @ _JSON_CAMERA_FROM_OPENCV
    translation = c2w_zup[:3, 3]

    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = [
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]
    return {
        "view_idx": 0,
        "position": translation.astype(np.float32).tolist(),
        "quaternion_wxyz": quat_wxyz,
    }


def main() -> None:
    args = _parse_args()

    # Locate .voxels.h5
    h5_path = Path(args.data_dir) / f"{args.scene_id}.glb.voxels.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f".voxels.h5 not found: {h5_path}\n"
            "Validity checking requires this file."
        )

    print(f"[random] Loading voxel grid: {h5_path}")
    vg = VoxelGrid.from_h5(h5_path)
    lo, hi = vg.bounds()
    print(
        f"  scene bounds  X [{lo[0]:.3f}, {hi[0]:.3f}]  "
        f"Y [{lo[1]:.3f}, {hi[1]:.3f}]  Z [{lo[2]:.3f}, {hi[2]:.3f}]"
    )
    print(
        f"  voxel pitch   {vg.pitch:.6f} m  "
        f"grid {vg.shape[0]}×{vg.shape[1]}×{vg.shape[2]}"
    )
    print(
        f"  occupied      {vg.grid.sum()} / {vg.grid.size} voxels "
        f"({100 * vg.grid.mean():.1f} %)"
    )

    cfg = RandomSamplerConfig(
        z_min=args.z_min,
        z_max=args.z_max,
        xy_radius=args.xy_radius,
        z_radius=args.z_radius,
        border_voxels=args.border_voxels,
        max_attempts=args.max_attempts,
    )
    rng = np.random.default_rng(args.seed)

    views = []
    has_init_xyz = all(v is not None for v in (args.init_x, args.init_y, args.init_z, args.init_yaw_deg))
    if has_init_xyz:
        init_pos = np.array([args.init_x, args.init_y, args.init_z], dtype=np.float32)
        if not _is_valid(init_pos, vg, cfg.xy_radius, cfg.z_radius, cfg.border_voxels):
            raise RuntimeError(
                "provided --init_x/y/z maps to an invalid first-view position "
                f"for scene {args.scene_id}: {init_pos.tolist()}"
            )
        yaw = np.radians(args.init_yaw_deg)
        views.append({
            "view_idx": 0,
            "position": init_pos.tolist(),
            "quaternion_wxyz": [
                float(np.cos(yaw / 2)),
                0.0,
                0.0,
                float(np.sin(yaw / 2)),
            ],
        })
        print(
            f"[random] Using init pose as view 0: "
            f"pos={np.round(init_pos, 4).tolist()} yaw_deg={args.init_yaw_deg:.1f}"
        )
    elif args.init_pose_json is not None:
        init_view = _active_gs_init_pose_to_view(args.init_pose_json)
        init_pos = np.asarray(init_view["position"], dtype=np.float32)
        if not _is_valid(init_pos, vg, cfg.xy_radius, cfg.z_radius, cfg.border_voxels):
            raise RuntimeError(
                "provided --init_pose_json maps to an invalid first-view position "
                f"for scene {args.scene_id}: {init_pos.tolist()}"
            )
        views.append(init_view)
        print(
            "[random] Using external init pose as view 0: "
            f"pos={np.round(init_pos, 4).tolist()}"
        )

    n_random_views = max(args.n_views - len(views), 0)
    print(
        f"\n[random] Sampling {n_random_views} additional random views "
        f"(seed={args.seed}) …"
    )
    random_views = sample_random_poses(vg, n_random_views, cfg, rng)
    for offset, view in enumerate(random_views, start=len(views)):
        view["view_idx"] = offset
    views.extend(random_views)
    print(f"[random] Generated {len(views)} / {args.n_views} total views.")

    if not views:
        raise RuntimeError("No valid positions found – check scene_id or relax constraints.")

    # Write JSON
    out_dir = Path(args.poses_dir)
    if out_dir.name != args.method:
        out_dir = out_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scene_id = args.scene_id.replace("/", "__")
    out_path = out_dir / f"{safe_scene_id}_{args.run_id}.json"

    pose_data = {
        "scene_id": args.scene_id,
        "method": args.method,
        "run_id": args.run_id,
        "max_views": len(views),
        "fov_deg": args.fov_deg,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "camera_near": args.camera_near,
        "camera_far": args.camera_far,
        "quaternion_convention": "wxyz",
        "coordinate_frame": "camera_to_world",
        "up_axis": "z",
        "views": views,
    }
    with open(out_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    print(f"[random] Saved → {out_path}")


if __name__ == "__main__":
    main()
