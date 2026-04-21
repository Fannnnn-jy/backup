"""Step 1: Render RGB + depth from a GLB scene at camera poses specified in a pose JSON file.

Usage:
    python baselines/render_from_poses.py \\
        --pose_json baselines/poses/random_procthor_ProcTHOR-Test-21_run_000.json \\
        [--data_dir /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data] \\
        [--output_dir baselines/rendered] \\
        [--invalid_depth_skip_ratio 0.10] \\
        [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.envs.glb_camera_env  # noqa: F401 – registers "GLBCameraAgent-v1"
import gymnasium as gym

from src.envs.voxel_carving import intrinsics_from_fov


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_rgb_depth(env, camera_uid: str):
    """One render pass → (H,W,3) uint8 RGB and (H,W) float32 depth."""
    sensor = env.unwrapped._sensors[camera_uid]
    env.unwrapped.scene.update_render(
        update_sensors=True, update_human_render_cameras=False
    )
    sensor.capture()
    obs = sensor.get_obs(
        rgb=True, depth=True, position=False,
        segmentation=False, normal=False, albedo=False,
    )
    rgb = obs["rgb"]
    depth = obs["depth"]

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]

    depth = np.asarray(depth, dtype=np.float32)
    # Normalise to (H, W): strip leading batch dims and trailing channel dim.
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]          # (1,...) → drop batch
    if depth.ndim == 3:
        depth = depth[..., 0]     # (H, W, 1) → (H, W)
    depth = depth / 1000.0        # SAPIEN sensor returns mm → convert to metres

    return rgb, depth


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(str(path))
    except ImportError:
        import cv2
        cv2.imwrite(str(path), rgb[..., ::-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RGB+depth from a GLB scene at camera poses from a JSON file."
    )
    parser.add_argument("--pose_json", required=True,
                        help="Path to pose JSON file (see BASELINE.md for schema).")
    parser.add_argument("--data_dir",
                        default="/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data",
                        help="Root directory that contains the GLB files.")
    parser.add_argument("--output_dir", default="baselines/rendered",
                        help="Root output directory for rendered frames.")
    parser.add_argument("--render_backend", default="gpu",
                        choices=["gpu", "cpu"],
                        help="SAPIEN render backend.")
    parser.add_argument(
        "--invalid_depth_skip_ratio",
        type=float,
        default=0.10,
        help="Skip a rendered view if invalid depth pixel ratio is greater than this threshold.",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render even if output directory already exists.")
    args = parser.parse_args()
    if not (0.0 <= args.invalid_depth_skip_ratio <= 1.0):
        raise ValueError(
            f"--invalid_depth_skip_ratio must be within [0, 1], got {args.invalid_depth_skip_ratio}"
        )

    # --- load pose JSON ---
    pose_json_path = Path(args.pose_json)
    with open(pose_json_path) as f:
        pose_data = json.load(f)

    scene_id: str = pose_data["scene_id"]
    method: str = pose_data.get("method", "unknown")
    run_id: str = pose_data.get("run_id", "run_000")
    max_views: int = int(pose_data["max_views"])
    views = pose_data["views"][:max_views]

    fov_deg: float = float(pose_data.get("fov_deg", 90.0))
    image_width: int = int(pose_data.get("image_width", 512))
    image_height: int = int(pose_data.get("image_height", 512))
    camera_near: float = float(pose_data.get("camera_near", 0.01))
    camera_far: float = float(pose_data.get("camera_far", 10.0))

    # --- resolve GLB path ---
    glb_path = Path(args.data_dir) / f"{scene_id}.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    # --- output directory ---
    safe_scene_id = scene_id.replace("/", "__")
    out_dir = Path(args.output_dir) / safe_scene_id / f"{method}_{run_id}"
    if out_dir.exists() and not args.overwrite:
        print(f"[render] Already exists, skipping (use --overwrite): {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- compute intrinsics ---
    fov_rad = np.deg2rad(fov_deg)
    fx, fy, cx, cy = intrinsics_from_fov(image_width, image_height, fov_rad, fov_axis="vertical")
    intrinsics_mat = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)

    # --- build env ---
    first_view = views[0]
    env = gym.make(
        "GLBCameraAgent-v1",
        glb_path=str(glb_path),
        y_up=True,
        add_collision=False,
        obs_mode="sensor_data",
        reward_mode=None,
        control_mode=None,
        num_envs=1,
        render_backend=args.render_backend,
        sim_backend="physx_cpu",
        camera_width=image_width,
        camera_height=image_height,
        camera_fov=fov_rad,
        camera_near=camera_near,
        camera_far=camera_far,
        init_position=first_view["position"],
        init_quat=first_view["quaternion_wxyz"],
        action_mode="absolute_quat",
    )
    env.reset()

    camera_uid = env.unwrapped.camera_uid
    frame_meta = []

    for view in views:
        idx = int(view["view_idx"])
        pos = view["position"]
        quat = view["quaternion_wxyz"]

        env.unwrapped.set_agent_pose(position=pos, quat=quat)
        rgb, depth = _render_rgb_depth(env, camera_uid)

        valid_d = depth[depth > 0]
        invalid_ratio = 1 - valid_d.size / depth.size
        if invalid_ratio > args.invalid_depth_skip_ratio:
            print(
                f"  [render] view {idx:04d}  SKIP – "
                f"{depth.size - valid_d.size} invalid depth pixels ({100*invalid_ratio:.1f}%), "
                f"threshold={100*args.invalid_depth_skip_ratio:.1f}%"
            )
            continue

        rgb_path = out_dir / f"view_{idx:04d}_rgb.png"
        depth_path = out_dir / f"view_{idx:04d}_depth.npy"

        _save_rgb(rgb, rgb_path)
        np.save(str(depth_path), depth)

        print(f"  [render] view {idx:04d}  depth [{valid_d.min():.3f}, {valid_d.max():.3f}] m")
        frame_meta.append({
            "view_idx": idx,
            "position": list(pos),
            "quaternion_wxyz": list(quat),
            "rgb": rgb_path.name,
            "depth": depth_path.name,
        })

    env.close()

    # --- save meta.json ---
    meta = {
        "scene_id": scene_id,
        "method": method,
        "run_id": run_id,
        "fov_deg": fov_deg,
        "image_width": image_width,
        "image_height": image_height,
        "camera_near": camera_near,
        "camera_far": camera_far,
        "intrinsics": intrinsics_mat.tolist(),
        "frames": frame_meta,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[render] Done. {len(views)} frames → {out_dir}")


if __name__ == "__main__":
    main()
