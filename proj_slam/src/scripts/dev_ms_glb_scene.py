from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np


def _resolve_sim_backend(sim_backend: str) -> str:
    if sim_backend == "maniskill":
        return "physx_cpu"
    return sim_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GLB scene with GLBCameraAgent-v1.")
    parser.add_argument("glb_path", help="Path to the GLB file.")
    parser.add_argument("--save-path", default="glb_scene.png", help="Optional path to save an RGB frame.")
    parser.add_argument("--y-up", action="store_true", default=True, help="Rotate scene from Y-up to Z-up.")
    parser.add_argument("--no-y-up", dest="y_up", action="store_false", help="Disable Y-up rotation.")
    parser.add_argument("--add-collision", action="store_true", help="Add non-convex collision for the scene.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale for the scene.")
    parser.add_argument("--render-backend", default="gpu", help="Render backend (gpu/cpu/cuda).")
    parser.add_argument("--sim-backend", default="maniskill", help="Sim backend (maniskill/physx_cpu/physx_cuda).")
    parser.add_argument("--width", type=int, default=512, help="Camera width.")
    parser.add_argument("--height", type=int, default=512, help="Camera height.")
    parser.add_argument("--fov", type=float, default=1.0, help="Camera FOV (radians).")
    parser.add_argument("--near", type=float, default=0.01, help="Camera near plane.")
    parser.add_argument("--far", type=float, default=100.0, help="Camera far plane.")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2 wait time per frame.")
    parser.add_argument("--live", action="store_true", help="Render continuously in a window.")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Disable live rendering.")
    parser.set_defaults(live=True)
    return parser.parse_args()


def _render_rgb(env) -> "np.ndarray":
    rgb = env.unwrapped.render_agent_rgb()
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    return rgb


def main() -> None:
    args = _parse_args()
    sim_backend = _resolve_sim_backend(args.sim_backend)

    env_kwargs = dict(
        obs_mode="sensor_data",
        reward_mode=None,
        control_mode=None,
        num_envs=1,
        render_backend=args.render_backend,
        sim_backend=sim_backend,
        camera_width=args.width,
        camera_height=args.height,
        camera_fov=args.fov,
        camera_near=args.near,
        camera_far=args.far,
        init_position=(0.0, 10.0, 1),
        init_quat=(0.0, 0.0, 0.0, 1.0),
        action_mode="absolute_quat",
    )

    env = gym.make(
        "GLBCameraAgent-v1",
        glb_path=args.glb_path,
        y_up=args.y_up,
        add_collision=args.add_collision,
        scale=args.scale,
        **env_kwargs,
    )

    env.reset()

    pos, quat = env.unwrapped.get_agent_pose()
    print("agent pose:", pos, quat)

    rgb = _render_rgb(env)
    if args.save_path:
        import matplotlib.pyplot as plt

        plt.imsave(args.save_path, rgb)
        print(f"Saved RGB to: {args.save_path}")

    if not args.live:
        env.close()
        return

    # try:
    #     import cv2
    # except ImportError:
    #     print("cv2 not available; disable --live or install opencv-python.")
    #     env.close()
    #     return

    # window_name = "glb_scene"
    # while True:
    #     rgb = _render_rgb(env)
    #     bgr = rgb[..., ::-1]
    #     cv2.imshow(window_name, bgr)
    #     key = cv2.waitKey(args.wait_ms) & 0xFF
    #     if key in (ord("q"), 27):
    #         break
    #     time.sleep(0.001)

    # env.close()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
