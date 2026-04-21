from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.active_mapping_env import ActiveMappingConfig
from envs.debug_camera import print_camera_debug
from envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ManiSkill camera pose and buffers.")
    parser.add_argument("--gleam-data-dir", default="data/eval_128", help="Path to GLEAM eval_128 dir.")
    parser.add_argument("--gleam-dataset-name", default="", help="Dataset prefix (default: dir name).")
    parser.add_argument("--scene-index", type=int, default=-1, help="Fixed scene index.")
    parser.add_argument("--render-device", default="", help="SAPIEN render device alias.")
    parser.add_argument("--width", type=int, default=256, help="Depth image width.")
    parser.add_argument("--height", type=int, default=256, help="Depth image height.")
    parser.add_argument("--max-frames", type=int, default=4, help="Depth stack length.")
    parser.add_argument("--env-depth-max", type=float, default=8.0, help="Depth max in env.")
    parser.add_argument(
        "--depth-buffer",
        default=ActiveMappingConfig().maniskill_depth_buffer,
        help="Camera depth buffer name (e.g. DepthLinear, Depth, GbufferDepth).",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between steps (seconds).")
    parser.add_argument("--random-action", action="store_true", help="Use random actions.")
    parser.add_argument(
        "--dump-buffers",
        action="store_true",
        help="Print raw camera buffer stats each interval.",
    )
    parser.add_argument(
        "--dump-interval",
        type=int,
        default=1,
        help="Dump buffer stats every N steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gleam_dir = Path(args.gleam_data_dir)
    dataset_name = args.gleam_dataset_name or gleam_dir.name

    cfg = ActiveMappingConfig(
        sim_backend="maniskill",
        gleam_data_dir=str(gleam_dir),
        gleam_dataset_name=dataset_name,
        gleam_scene_index=args.scene_index,
        maniskill_render_device=args.render_device,
        maniskill_depth_buffer=args.depth_buffer,
        max_depth_frames=args.max_frames,
        depth_image_shape=(args.height, args.width),
        depth_data_max_value=args.env_depth_max,
    )

    env = ManiSkillActiveMappingEnv(cfg)
    obs, info = env.reset(seed=args.seed)
    print_camera_debug(env, 0, note="reset")
    if args.dump_buffers:
        from envs.debug_camera import dump_camera_buffer_stats

        cams = getattr(env, "_cameras", None)
        cam = cams[0] if cams else None
        dump_camera_buffer_stats(cam, note="reset")

    try:
        for step_idx in range(1, args.steps + 1):
            if args.random_action:
                action = env.action_space.sample()
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            print_camera_debug(env, step_idx)
            if args.dump_buffers and args.dump_interval > 0 and step_idx % args.dump_interval == 0:
                from envs.debug_camera import dump_camera_buffer_stats

                cams = getattr(env, "_cameras", None)
                cam = cams[0] if cams else None
                dump_camera_buffer_stats(cam, note=f"step={step_idx}")
            if args.sleep > 0:
                time.sleep(args.sleep)
            if terminated or truncated:
                obs, info = env.reset()
                print_camera_debug(env, step_idx, note="reset")
                if args.dump_buffers:
                    from envs.debug_camera import dump_camera_buffer_stats

                    cams = getattr(env, "_cameras", None)
                    cam = cams[0] if cams else None
                    dump_camera_buffer_stats(cam, note="reset")
    finally:
        env.close()


if __name__ == "__main__":
    main()
