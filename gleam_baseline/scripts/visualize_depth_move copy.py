from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import sys

import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.active_mapping_env import ActiveMappingConfig
from envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv
from envs.voxel_carving import carve_depth_to_voxels, intrinsics_from_fov
from utils.rendering import init_render_state, reset_render_state, render_depth_and_topdown


def _resize_to_height(
    image: np.ndarray, target_height: int, *, interpolation: int | None = None
) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / float(height)
    new_width = max(1, int(round(width * scale)))
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, target_height), interpolation=interpolation)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ManiSkill depth observations.")
    parser.add_argument("--gleam-data-dir", default="data/eval_128", help="Path to GLEAM eval_128 dir.")
    parser.add_argument("--gleam-dataset-name", default="", help="Dataset prefix (default: dir name).")
    parser.add_argument("--scene-index", type=int, default=-1, help="Fixed scene index.")
    parser.add_argument("--render-device", default="", help="SAPIEN render device alias.")
    parser.add_argument("--width", type=int, default=256, help="Depth image width.")
    parser.add_argument("--height", type=int, default=256, help="Depth image height.")
    parser.add_argument("--max-frames", type=int, default=4, help="Depth stack length.")
    parser.add_argument(
        "--forward-axis",
        choices=("x+", "y+", "z+", "x-", "y-", "z-"),
        default="",
        help="Force a single camera to face the given axis (test helper).",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=8.0,
        help="Depth max for visualization (0 to auto-scale).",
    )
    parser.add_argument(
        "--env_depth_max",
        type=float,
        default=ActiveMappingConfig().depth_data_max_value,
        help="Depth max used to clip raw depth images in the environment.",
    )
    parser.add_argument(
        "--depth-buffer",
        default=ActiveMappingConfig().maniskill_depth_buffer,
        help="Camera depth buffer name (e.g. DepthLinear, PointDepth).",
    )
    parser.add_argument(
        "--carve-depth-buffer",
        default="DepthLinear",
        help="Depth buffer used for ray carving (e.g. DepthLinear).",
    )
    parser.add_argument(
        "--view",
        choices=("depth", "topdown", "both"),
        default="both",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--topdown-scale", type=int, default=4, help="Scale factor for topdown map."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2 wait time per frame.")
    parser.add_argument("--random-action", action="store_true", help="Use random actions.")
    parser.add_argument("--load-model", default="", help="Path to a trained model checkpoint.")
    parser.add_argument(
        "--no-gleam-use-init-pose",
        action="store_true",
        help="Disable dataset-provided initial poses.",
    )
    parser.add_argument(
        "--policy",
        choices=("cnn", "rnn"),
        default="cnn",
        help="Policy architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Policy device for inference (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=-1,
        help="Print per-frame depth stats every N steps (0 to disable).",
    )
    parser.add_argument(
        "--first-camera-only",
        action="store_true",
        help="Render only the first depth frame instead of tiling all cameras.",
    )
    parser.add_argument(
        "--reward-by",
        choices=("location", "sight"),
        default="location",
        help="Reward mode: location (default) or sight (carving-based).",
    )
    return parser.parse_args()


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_policy(
    cfg: ActiveMappingConfig, model_path: str, device: str, seed: int, policy_kind: str
):
    import torch

    from learning.config import TrainConfig, build_train_cfg
    from learning.policy import register_custom_models
    from learning.utils import GymnasiumVecEnv
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    policy_device = _resolve_device(device)
    vec_env = GymnasiumVecEnv(cfg, 1, seed, policy_device)
    train_cfg = build_train_cfg(TrainConfig(device=device))
    if policy_kind == "rnn":
        train_cfg["policy"]["class_name"] = "DepthActorCriticCNNRecurrent"
    else:
        train_cfg["policy"]["class_name"] = "DepthActorCriticCNN"
    register_custom_models()
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir="", device=policy_device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=policy_device)
    return vec_env, policy, policy_device, runner


def _camera_pose(cam: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for getter in ("get_pose", "get_local_pose"):
        if not hasattr(cam, getter):
            continue
        try:
            pose = getattr(cam, getter)()
        except Exception:
            continue
        if pose is None:
            continue
        if not hasattr(pose, "p") or not hasattr(pose, "q"):
            continue
        try:
            pos = np.asarray(pose.p, dtype=np.float32)
            quat = np.asarray(pose.q, dtype=np.float32)
        except Exception:
            continue
        if pos.shape[0] < 3 or quat.shape[0] < 4:
            continue
        return pos[:3], quat[:4]
    return None


def _depth_is_z_for_buffer(buffer_name: str) -> bool:
    name = buffer_name.lower()
    if "pointdepth" in name or "linedepth" in name:
        return False
    return True


def _prepare_depth_for_carving(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, neginf=0.0, posinf=0.0)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.size > 0 and np.nanmax(depth) <= 0.0 and np.nanmin(depth) < 0.0:
        depth = -depth
    target_h, target_w = target_shape
    if depth.shape != (target_h, target_w):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return depth


def _carve_from_camera(
    env: Any,
    cfg: ActiveMappingConfig,
    cam: Any,
    *,
    grid_size: int,
    depth_buffer: str,
    forward_axis: str,
) -> Any:
    cam_pose = _camera_pose(cam)
    if cam_pose is None:
        return None
    range_gt = getattr(env, "_range_gt", None)
    voxel_size = getattr(env, "_voxel_size", None)
    if range_gt is None or voxel_size is None:
        return None
    depth_raw = env._read_camera_depth(cam, target_name=depth_buffer or None)
    depth = _prepare_depth_for_carving(depth_raw, cfg.depth_image_shape)
    depth_is_z = _depth_is_z_for_buffer(depth_buffer or "")
    h, w = depth.shape[:2]
    if hasattr(cam, "get_intrinsic_matrix"):
        K = cam.get_intrinsic_matrix()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        fov_rad = math.radians(float(cfg.maniskill_camera_fov))
        fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_rad)

    cam_pos, cam_quat = cam_pose
    depth_scale = 1.0
    depth_max = float(cfg.maniskill_camera_far or cfg.depth_data_max_value)
    max_obs = float(np.nanmax(depth)) if np.isfinite(depth).any() else 0.0
    if depth_max > 1.0 and max_obs <= 1.0 + 1e-3:
        depth_scale = depth_max

    masks = carve_depth_to_voxels(
        depth,
        cam_pos,
        cam_quat,
        range_gt,
        voxel_size,
        grid_size,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        forward_axis=forward_axis or None,
        depth_is_z=depth_is_z,
        depth_scale=depth_scale,
        depth_max=depth_max,
        depth_max_epsilon=depth_max * 0.01,
        depth_max_is_no_hit=True,
        ground_z=cfg.carve_min_z,
    )
    return masks


def _reset_reward_visited_full(env: Any) -> None:
    reward_full = getattr(env, "_reward_visited_full", None)
    if reward_full is not None:
        reward_full.fill(False)


def _action_from_key(key: int) -> np.ndarray | None:
    if key < 0:
        return None
    try:
        ch = chr(key).lower()
    except ValueError:
        return None
    key_map = {
        "w": (0.0, 1.0, 0.0),
        "s": (0.0, -1.0, 0.0),
        "a": (-1.0, 0.0, 0.0),
        "d": (1.0, 0.0, 0.0),
        "k": (0.0, 0.0, 1.0),
        "l": (0.0, 0.0, -1.0),
    }
    if ch not in key_map:
        return None
    return np.array(key_map[ch], dtype=np.float32)


def _compute_carving_error(
    carved_occupied: np.ndarray,
    carved_free: np.ndarray,
    gt_occupied: np.ndarray,
) -> tuple[float, int, int, int, int]:
    if (
        carved_occupied.shape != carved_free.shape
        or carved_occupied.shape != gt_occupied.shape
    ):
        raise ValueError("carved/free/gt shapes must match.")
    known = carved_occupied | carved_free
    known_count = int(known.sum())
    if known_count == 0:
        return 0.0, 0, 0, 0, 0
    false_pos = int((carved_occupied & ~gt_occupied).sum())
    false_neg = int((carved_free & gt_occupied).sum())
    errors = false_pos + false_neg
    error_rate = float(errors) / float(known_count)
    return error_rate, known_count, errors, false_pos, false_neg


def _summarize_mask_indices(mask: np.ndarray) -> tuple[int, tuple[float, float, float] | None, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    idx = np.argwhere(mask)
    count = int(idx.shape[0])
    if count == 0:
        return 0, None, None, None
    mins = tuple(int(v) for v in idx.min(axis=0))
    maxs = tuple(int(v) for v in idx.max(axis=0))
    means = tuple(float(v) for v in idx.mean(axis=0))
    return count, means, mins, maxs


def main() -> None:
    args = _parse_args()
    gleam_dir = Path(args.gleam_data_dir)
    dataset_name = args.gleam_dataset_name or gleam_dir.name

    cfg = ActiveMappingConfig(
        sim_backend="maniskill",
        gleam_data_dir=str(gleam_dir),
        gleam_dataset_name=dataset_name,
        gleam_scene_index=args.scene_index,
        gleam_use_init_pose=not args.no_gleam_use_init_pose,
        maniskill_render_device=args.render_device,
        maniskill_depth_buffer=args.depth_buffer,
        max_depth_frames=args.max_frames,
        depth_image_shape=(args.height, args.width),
        depth_data_max_value=args.env_depth_max,
        reward_by=args.reward_by,
    )
    if args.forward_axis:
        axis_map = {
            "x+": (1.0, 0.0, 0.0),
            "y+": (0.0, 1.0, 0.0),
            "z+": (0.0, 0.0, 1.0),
            "x-": (-1.0, 0.0, 0.0),
            "y-": (0.0, -1.0, 0.0),
            "z-": (0.0, 0.0, -1.0),
        }
        cfg.maniskill_camera_axis_forward = axis_map[args.forward_axis]

    use_policy = bool(args.load_model)
    vec_env = None
    policy = None
    policy_device = None
    runner = None
    if use_policy:
        vec_env, policy, policy_device, runner = _load_policy(
            cfg, args.load_model, args.device, args.seed, args.policy
        )
        env = vec_env.env.envs[0]
        obs = vec_env.reset()
    else:
        env = ManiSkillActiveMappingEnv(cfg)
        obs, info = env.reset(seed=args.seed)


    state = init_render_state(cfg)
    grid_size = int(cfg.gleam_grid_size)
    step_idx = 0
    carve_cam_idx = 0


    try:
        while True:
            render_images: list[tuple[np.ndarray, str]] = []
            depth_img, topdown_img, current_position = render_depth_and_topdown(
                obs,
                env,
                state=state,
                depth_max=args.depth_max,
                topdown_scale=args.topdown_scale,
                view=args.view,
                first_camera_only=args.first_camera_only,
                stats_interval=args.stats_interval,
                step_idx=step_idx,
            )
            if depth_img is not None:
                render_images.append((depth_img, "depth"))
            if topdown_img is not None:
                render_images.append((topdown_img, "topdown"))
            if render_images:
                display, _ = render_images[0]
                if len(render_images) > 1:
                    target_height = display.shape[0]
                    resized = [display]
                    for img, kind in render_images[1:]:
                        if kind == "topdown":
                            img = _resize_to_height(
                                img, target_height, interpolation=cv2.INTER_NEAREST
                            )
                        else:
                            img = _resize_to_height(img, target_height)
                        ###
                        resized.append(img)
                    display = np.concatenate(resized, axis=1)
                cv2.imshow("depth", display)
            key = cv2.waitKey(args.wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                gt_occupied = getattr(env, "_obstacle_mask_3d", None)
                if gt_occupied is None:
                    print("[carving] ground truth unavailable.")
                    continue
                try:
                    err_rate, known, errors, fp, fn = _compute_carving_error(
                        state.carved_occupied, state.carved_free, gt_occupied
                    )
                except ValueError as exc:
                    print(f"[carving] error: {exc}")
                    continue
                carved_occ_count = int(state.carved_occupied.sum())
                carved_free_count = int(state.carved_free.sum())
                known_mask = state.carved_occupied | state.carved_free
                known_occ = int((known_mask & gt_occupied).sum())
                known_free = int((known_mask & ~gt_occupied).sum())
                occ_error_rate = (float(fp) / float(carved_occ_count)) if carved_occ_count > 0 else 0.0
                free_error_rate = (float(fn) / float(carved_free_count)) if carved_free_count > 0 else 0.0
                if known == 0:
                    print("[carving] no carved voxels to evaluate.")
                else:
                    print(
                        f"[carving] known={known} occ={carved_occ_count} free={carved_free_count} "
                        f"gt_occ_in_known={known_occ} gt_free_in_known={known_free}"
                    )
                    print(
                        f"[carving] error_rate={err_rate:.4f} errors={errors} "
                        f"fp={fp} fn={fn} occ_err={occ_error_rate:.4f} free_err={free_error_rate:.4f}"
                    )
                occ_stats = _summarize_mask_indices(state.carved_occupied)
                free_stats = _summarize_mask_indices(state.carved_free)
                if occ_stats[0] > 0:
                    print(
                        f"[carving] occ_idx count={occ_stats[0]} mean={occ_stats[1]} "
                        f"min={occ_stats[2]} max={occ_stats[3]}"
                    )
                if free_stats[0] > 0:
                    print(
                        f"[carving] free_idx count={free_stats[0]} mean={free_stats[1]} "
                        f"min={free_stats[2]} max={free_stats[3]}"
                    )
                state.show_error_overlay = True
                continue
            if key == ord("r"):
                if args.reward_by == "sight":
                    cameras = getattr(env, "_cameras", None) or []
                    if not cameras:
                        print("[sight] no cameras available.")
                        continue
                    if carve_cam_idx >= len(cameras):
                        print(f"[sight] camera index {carve_cam_idx} out of range.")
                        continue
                    masks = _carve_from_camera(
                        env,
                        cfg,
                        cameras[carve_cam_idx],
                        grid_size=grid_size,
                        depth_buffer=args.carve_depth_buffer or cfg.maniskill_depth_buffer,
                        forward_axis=args.forward_axis or "",
                    )
                    if masks is None:
                        print("[sight] mask unavailable.")
                        continue
                    state.carved_occupied |= masks.occupied
                    state.carved_free |= masks.free
                    state.carved_free[state.carved_occupied] = False
                    seen_xy = np.any(masks.occupied | masks.free, axis=2)
                    seen_rc = np.flip(seen_xy.T, 0)
                    if hasattr(env, "_reward_visited"):
                        reward_mask = seen_rc
                        if (
                            reward_mask.shape != env._reward_visited.shape
                            and hasattr(env, "_downsample_mask")
                        ):
                            reward_mask = env._downsample_mask(
                                reward_mask, env._reward_visited.shape
                            )
                        if getattr(env, "_reward_valid_mask", None) is not None:
                            reward_mask = reward_mask & env._reward_valid_mask
                        env._reward_visited |= reward_mask
                        print(f"[sight] marked={int(reward_mask.sum())}")
                    if hasattr(env, "_valid_mask") and env._valid_mask is not None:
                        if env._valid_mask.shape == seen_rc.shape:
                            seen_rc = seen_rc & env._valid_mask
                    if getattr(env, "_reward_visited_full", None) is None:
                        env._reward_visited_full = np.zeros_like(seen_rc, dtype=bool)
                    env._reward_visited_full |= seen_rc
                    continue
                if use_policy and vec_env is not None:
                    obs = vec_env.reset()
                    if runner is not None and runner.alg.policy.is_recurrent:
                        runner.alg.policy.reset()
                else:
                    obs, info = env.reset()
                reset_render_state(state)
                _reset_reward_visited_full(env)
                carve_cam_idx = 0
                continue
            if key in (ord("["), ord("]")):
                cameras = getattr(env, "_cameras", None) or []
                if cameras:
                    delta = 1 if key == ord("]") else -1
                    carve_cam_idx = (carve_cam_idx + delta) % len(cameras)
                    print(f"[carving] set camera={carve_cam_idx}")
                continue
            if ord("1") <= key <= ord("9"):
                cameras = getattr(env, "_cameras", None) or []
                cam_idx = int(key - ord("1"))
                if cameras and cam_idx < len(cameras):
                    carve_cam_idx = cam_idx
                    print(f"[carving] set camera={carve_cam_idx}")
                continue
            if key == ord("c"):
                cameras = getattr(env, "_cameras", None) or []
                assert cameras, "No cameras available for carving."
                if carve_cam_idx >= len(cameras):
                    print(f"Selected camera index {carve_cam_idx} out of range.")
                    continue
                cam_idx = carve_cam_idx
                masks = _carve_from_camera(
                    env,
                    cfg,
                    cameras[cam_idx],
                    grid_size=grid_size,
                    depth_buffer=args.carve_depth_buffer or cfg.maniskill_depth_buffer,
                    forward_axis=args.forward_axis or "",
                )
                if masks is None:
                    print(f"[carving] camera={cam_idx} mask unavailable.")
                    continue
                state.carved_occupied |= masks.occupied
                state.carved_free |= masks.free
                state.carved_free[state.carved_occupied] = False
                print(
                    f"[carving] camera={cam_idx} occ={int(masks.occupied.sum())} "
                    f"free={int(masks.free.sum())}"
                )
                continue
            if not use_policy and not args.random_action and not args.forward_axis:
                manual_action = _action_from_key(key)
                if manual_action is not None:
                    obs, reward, terminated, truncated, info = env.step(manual_action)
                    step_idx += 1
                    if current_position is not None:
                        state.prev_position = current_position
                    if terminated or truncated:
                        obs, info = env.reset()
                        reset_render_state(state)
                        _reset_reward_visited_full(env)
                        carve_cam_idx = 0
                    continue

            if args.random_action or args.forward_axis:
                assert not use_policy, "在使用预训练 policy 的时候不能使用 random 或者 forward axis！"
                if args.random_action:
                    action = env.action_space.sample()
                    time.sleep(0.1)
                elif args.forward_axis:
                    action = cfg.maniskill_camera_axis_forward
                    time.sleep(0.1)
                else:
                    assert False, "random_action 和 forwar_axis 不能同时配置！"

                if use_policy and vec_env is not None:
                    import torch

                    actions = torch.tensor(action, dtype=torch.float32, device=policy_device).unsqueeze(0)
                    obs, reward, dones, info = vec_env.step(actions)
                    if runner is not None and runner.alg.policy.is_recurrent:
                        runner.alg.policy.reset(dones)
                    terminated = bool(dones[0].item())
                    truncated = False
                else:
                    obs, reward, terminated, truncated, info = env.step(action)
                step_idx += 1
                if current_position is not None:
                    state.prev_position = current_position
                if terminated or truncated:
                    if use_policy and vec_env is not None:
                        obs = vec_env.reset()
                    else:
                        obs, info = env.reset()
                    reset_render_state(state)
                    _reset_reward_visited_full(env)
                    carve_cam_idx = 0
                
            elif use_policy and policy is not None and vec_env is not None:
                import torch

                with torch.no_grad():
                    actions = policy(obs)
                    # actions = runner.alg.policy.act(obs)
                obs, reward, dones, info = vec_env.step(actions)

                time.sleep(0.1)
                
                # print("action", actions)
                # print("agent_state", obs["agent_state"])
                # print("obs_depth_tensor", obs["obs_depth_tensor"])
                # print("obs_depth_tensor shape", obs["obs_depth_tensor"].shape)
                
                # print("action_mean", runner.alg.policy.action_mean)
                # print("action_std", runner.alg.policy.action_std)

                # assert False, "debug"
                if runner is not None and runner.alg.policy.is_recurrent:
                    runner.alg.policy.reset(dones)
                terminated = bool(dones[0].item())
                truncated = False
                step_idx += 1
                if current_position is not None:
                    state.prev_position = current_position
                if terminated or truncated:
                    obs = vec_env.reset()
                    reset_render_state(state)
                    _reset_reward_visited_full(env)
                    carve_cam_idx = 0
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)  # shape (3,)

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
