"""Interactive trajectory sampler.

Launch the same active-mapping viewer as ``visualize_active.py`` but add a
**pose-recording** workflow:

* Navigate with the usual WASD / K-L / 1-6 keys.
* Press **P** to save the current camera pose as a new view.
* Press **Q** / Esc to quit.  On exit the collected poses are written as a
JSON file (same schema as ``proj/baselines/poses/example_pose.json``) into
``<output_dir>/raw_poses/``.

Usage example::

    python -m src.traj_sampling.sample_trajectory \
    /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/replicacad/apt_0.glb \
    --output-dir ./src/traj_sampling/my_trajectories \
    --scene-id replicacad/apt_0 \
    --run-id run_000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (mirrored from visualize_active.py so we don't import it)
# ---------------------------------------------------------------------------

STANDARD_GLB_BOUNDS_Y_UP = {"x": (-1.0, 1.0), "y": (0.0, 1.0), "z": (-1.0, 1.0)}
STANDARD_GLB_INTERNAL_BOUNDS_Z_UP = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.0)}
STANDARD_GLB_VOXEL_GRID = (128, 128, 64)
STANDARD_GLB_VOXEL_PITCH = 2.0 / 128.0
STANDARD_GLB_AGENT_HEIGHT = 0.5
DEFAULT_DEPTH_BUFFER = "DepthLinear"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float_list(text: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def _parse_vec3(text: str) -> Tuple[float, ...]:
    vals = _parse_float_list(text)
    if len(vals) not in (2, 3):
        raise argparse.ArgumentTypeError("Expected 2 or 3 floats: x,y[,z]")
    return tuple(float(v) for v in vals)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively sample a camera trajectory and save poses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("glb_path", help="Path to a GLB file.")
    parser.add_argument("--output-dir", default=".", help="Root output directory. Poses are saved under <output-dir>/raw_poses/.")
    parser.add_argument("--scene-id", default="unknown_scene", help="Scene identifier written to JSON.")
    parser.add_argument("--run-id", default="run_000", help="Run identifier written to JSON.")
    parser.add_argument("--method", default="manual", help="Method name written to JSON.")
    parser.add_argument("--max-views", type=int, default=-1, help="Max views to record (-1 = unlimited).")

    # Env / rendering
    parser.add_argument("--y-up", action="store_true", default=True)
    parser.add_argument("--no-y-up", dest="y_up", action="store_false")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--move-step", type=float, default=0.1)
    parser.add_argument("--rot-step-deg", type=float, default=5.0)
    parser.add_argument("--topdown", action="store_true", default=True)
    parser.add_argument("--no-topdown", dest="topdown", action="store_false")
    parser.add_argument("--init-pos", type=_parse_vec3, default=None)
    parser.add_argument("--glb-dir", default="")
    parser.add_argument("--glb-glob", default="*.glb")
    parser.add_argument("--glb-scene-index", type=int, default=-1)
    parser.add_argument("--am-action-mode", default="xyz_quat")
    parser.add_argument("--am-camera-angles", default="0")
    parser.add_argument("--am-single-camera", action="store_true", default=True)
    parser.add_argument("--am-multi-camera", dest="am_single_camera", action="store_false")
    parser.add_argument("--am-topdown-scale", type=int, default=4)
    parser.add_argument("--am-depth-max", type=float, default=-1.0)
    parser.add_argument("--am-depth-buffer", default=DEFAULT_DEPTH_BUFFER)
    parser.add_argument("--am-image-mode", choices=("rgb", "depth"), default="rgb")
    parser.add_argument("--am-camera-fov", type=float, default=-1.0)
    parser.add_argument("--am-camera-height", type=float, default=-1.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Env construction (same logic as visualize_active.py)
# ---------------------------------------------------------------------------

def _build_cfg(args, glb_dir, glb_glob, scene_index):
    from src.envs.active_mapping_env import ActiveMappingConfig

    cfg = ActiveMappingConfig()
    cfg.glb_data_dir = str(glb_dir)
    cfg.glb_scene_glob = glb_glob
    cfg.glb_scene_index = int(scene_index)
    cfg.glb_y_up = bool(args.y_up)
    cfg.glb_scale = float(args.scale)

    cfg.glb_voxel_pitch = float(STANDARD_GLB_VOXEL_PITCH)
    cfg.glb_voxel_max_dim = int(STANDARD_GLB_VOXEL_GRID[0])
    cfg.agent_init_height = float(STANDARD_GLB_AGENT_HEIGHT)
    cfg.maniskill_camera_height = float(STANDARD_GLB_AGENT_HEIGHT)
    cfg.maniskill_depth_buffer = str(args.am_depth_buffer).strip() or DEFAULT_DEPTH_BUFFER

    if args.am_action_mode:
        cfg.action_mode = args.am_action_mode
    if args.am_camera_angles:
        cfg.maniskill_camera_angles = tuple(_parse_float_list(args.am_camera_angles))
    if args.am_single_camera:
        cfg.single_camera_obs = True
    if args.am_depth_max > 0:
        cfg.depth_data_max_value = float(args.am_depth_max)
        cfg.maniskill_camera_far = float(args.am_depth_max)
    if args.am_camera_fov > 0:
        cfg.maniskill_camera_fov = float(args.am_camera_fov)
    if args.am_camera_height > 0:
        cfg.maniskill_camera_height = float(args.am_camera_height)
        cfg.agent_init_height = float(args.am_camera_height)
    if args.width > 0 and args.height > 0:
        cfg.depth_image_shape = (int(args.height), int(args.width))
    if args.init_pos is not None:
        cfg.init_position = tuple(float(v) for v in args.init_pos)
    return cfg


def _find_glb_scene_index(glb_dir: Path, glb_glob: str, glb_path: Path) -> Optional[int]:
    if not glb_dir.exists():
        return None
    paths = sorted([p for p in glb_dir.glob(glb_glob) if p.suffix.lower() == ".glb"])
    target = glb_path.resolve()
    for idx, path in enumerate(paths):
        try:
            if path.resolve() == target:
                return idx
        except Exception:
            continue
    return None


def _resolve_glb_scene_path(glb_dir: Path, glb_glob: str, scene_index: int, fallback: Path) -> Path:
    if not glb_dir.exists():
        return fallback
    paths = sorted([p for p in glb_dir.glob(glb_glob) if p.suffix.lower() == ".glb"])
    if 0 <= scene_index < len(paths):
        return paths[scene_index]
    return fallback


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    s = float(math.sin(half))
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Rendering helpers (thin wrappers)
# ---------------------------------------------------------------------------

def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame
    if frame.dtype == np.uint8:
        return frame
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    max_val = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if max_val <= 1.0 + 1e-3:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _stack_bgr_row(frames: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    import cv2
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    base = _ensure_uint8(valid[0])
    base_h, base_w = base.shape[:2]
    out = []
    for frame in frames:
        if frame is None:
            out.append(np.zeros((base_h, base_w, 3), dtype=np.uint8))
            continue
        frame_u8 = _ensure_uint8(frame)
        if frame_u8.shape[0] != base_h:
            new_w = max(1, int(round(frame_u8.shape[1] * base_h / float(frame_u8.shape[0]))))
            frame_u8 = cv2.resize(frame_u8, (new_w, base_h), interpolation=cv2.INTER_AREA)
        out.append(frame_u8)
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# Pose saving
# ---------------------------------------------------------------------------

def _save_poses(
    views: List[dict],
    output_dir: str,
    scene_id: str,
    method: str,
    run_id: str,
    fov_deg: float,
    image_width: int,
    image_height: int,
) -> str:
    raw_dir = Path(output_dir) / "raw_poses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: scene_id with '/' replaced by '__'
    safe_scene = scene_id.replace("/", "__").replace("\\", "__")
    filename = f"{safe_scene}_{run_id}.json"
    out_path = raw_dir / filename

    data = {
        "scene_id": scene_id,
        "method": method,
        "run_id": run_id,
        "max_views": len(views),
        "fov_deg": fov_deg,
        "image_width": image_width,
        "image_height": image_height,
        "camera_near": 0.01,
        "camera_far": 10.0,
        "quaternion_convention": "wxyz",
        "coordinate_frame": "camera_to_world",
        "up_axis": "z",
        "views": views,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(out_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _get_active_pose(env) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(getattr(env, "_position", np.zeros(3)), dtype=np.float32)
    quat = getattr(env, "_camera_quat_wxyz", None)
    if quat is None:
        yaw = float(getattr(env, "_yaw", 0.0))
        quat = np.array(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            dtype=np.float32,
        )
    else:
        quat = np.asarray(quat, dtype=np.float32)
    return pos, quat


def main() -> None:
    args = _parse_args()

    from src.envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv
    from src.utils.rendering import init_render_state, render_depth_and_topdown

    glb_path = Path(args.glb_path)
    glb_dir = Path(args.glb_dir) if args.glb_dir else glb_path.parent
    glb_glob = args.glb_glob or "*.glb"
    scene_index = int(args.glb_scene_index)
    if scene_index < 0:
        idx = _find_glb_scene_index(glb_dir, glb_glob, glb_path)
        if idx is None:
            print("[traj_sampling] glb path not found in dir; defaulting to index 0.")
            scene_index = 0
        else:
            scene_index = idx
    active_glb_path = _resolve_glb_scene_path(glb_dir, glb_glob, scene_index, glb_path)
    print(f"[traj_sampling] glb: {active_glb_path}")

    cfg = _build_cfg(args, glb_dir, glb_glob, scene_index)
    env = ManiSkillActiveMappingEnv(cfg)

    obs, _ = env.reset()
    render_state = init_render_state(cfg)

    mode = getattr(cfg, "action_mode", "xyz")
    move_step = float(args.move_step)
    rot_step = math.radians(float(args.rot_step_deg))
    yaw = float(getattr(env, "_yaw", 0.0))
    action_scale = float(cfg.action_scale) if float(cfg.action_scale) != 0.0 else 1.0
    quat_wxyz = np.asarray(
        getattr(env, "_camera_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])),
        dtype=np.float32,
    )

    depth_max = float(args.am_depth_max) if args.am_depth_max > 0 else float(cfg.depth_data_max_value)
    topdown_scale = max(1, int(args.am_topdown_scale))
    show_topdown = bool(args.topdown)

    fov_deg = float(cfg.maniskill_camera_fov) if hasattr(cfg, "maniskill_camera_fov") else 90.0

    # Recorded views
    saved_views: List[dict] = []

    try:
        import cv2
    except ImportError:
        print("cv2 not available; install opencv-python.")
        env.close()
        return

    print("=" * 60)
    print("Controls: WASD = move XY, K/L = +Z/-Z")
    if mode == "xyz_quat":
        print("          1/2 = yaw (left/right), 3/4 = pitch (up/down)")
    else:
        print("          1/2 = yaw")
    print("          P   = record current pose")
    print("          U   = undo last recorded pose")
    print("          Q/Esc = save & quit")
    print("=" * 60)

    window_name = "traj_sampling"
    while True:
        image_bgr, topdown_bgr, _ = render_depth_and_topdown(
            obs, env, state=render_state,
            depth_max=depth_max,
            topdown_scale=topdown_scale,
            view="both" if show_topdown else "depth",
            image_mode=args.am_image_mode,
            first_camera_only=args.am_single_camera,
        )

        # Draw pose count overlay
        info_frame = image_bgr
        if info_frame is not None:
            info_frame = info_frame.copy()
            cv2.putText(
                info_frame,
                f"Poses: {len(saved_views)}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        frame = _stack_bgr_row([info_frame, topdown_bgr])
        if frame is None:
            print("[traj_sampling] no frame; exiting.")
            break
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(args.wait_ms) & 0xFF

        # --- Quit ---
        if key in (ord("q"), ord("Q"), 27):
            break

        # --- Record pose ---
        if key in (ord("p"), ord("P")):
            pos, quat = _get_active_pose(env)
            view_idx = len(saved_views)
            if 0 < args.max_views <= view_idx:
                print(f"[traj_sampling] max_views ({args.max_views}) reached, cannot add more.")
            else:
                view = {
                    "view_idx": view_idx,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "quaternion_wxyz": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                }
                saved_views.append(view)
                print(
                    f"[traj_sampling] recorded view {view_idx}: "
                    f"pos={np.round(pos, 4).tolist()} quat={np.round(quat, 4).tolist()}"
                )
            continue

        # --- Undo last pose ---
        if key in (ord("u"), ord("U")):
            if saved_views:
                removed = saved_views.pop()
                print(f"[traj_sampling] undone view {removed['view_idx']}")
            else:
                print("[traj_sampling] no poses to undo.")
            continue

        # --- Movement ---
        delta = np.zeros(3, dtype=np.float32)
        moved = False
        quat_updated = False

        if key in (ord("w"), ord("W")):
            delta[1] += move_step; moved = True
        elif key in (ord("s"), ord("S")):
            delta[1] -= move_step; moved = True
        elif key in (ord("a"), ord("A")):
            delta[0] -= move_step; moved = True
        elif key in (ord("d"), ord("D")):
            delta[0] += move_step; moved = True
        elif key in (ord("k"), ord("K"), ord("l"), ord("L")) and mode not in ("xyz", "xyz_quat"):
            dz = move_step if key in (ord("k"), ord("K")) else -move_step
            prev_pos = np.asarray(getattr(env, "_position", np.zeros(3)), dtype=np.float32)
            new_pos = prev_pos.copy()
            new_pos[2] += dz
            bounds = getattr(env, "_bounds", None)
            if bounds is not None and len(bounds) == 6:
                new_pos[2] = float(np.clip(new_pos[2], float(bounds[4]), float(bounds[5])))
            try:
                env._position = new_pos
                if hasattr(env, "_update_camera_poses"):
                    env._update_camera_poses()
                if hasattr(env, "_build_observation"):
                    obs = env._build_observation()
                render_state.prev_position = prev_pos
            except Exception as exc:
                print(f"[traj_sampling] manual Z update failed: {exc}")
            continue
        elif key in (ord("k"), ord("K")) and mode in ("xyz", "xyz_quat"):
            delta[2] += move_step; moved = True
        elif key in (ord("l"), ord("L")) and mode in ("xyz", "xyz_quat"):
            delta[2] -= move_step; moved = True
        elif key == ord("1"):
            if mode == "xyz_quat":
                quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 0.0, 1.0], rot_step), quat_wxyz)
                quat_updated = True
            else:
                yaw += rot_step; moved = True
        elif key == ord("2"):
            if mode == "xyz_quat":
                quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 0.0, 1.0], -rot_step), quat_wxyz)
                quat_updated = True
            else:
                yaw -= rot_step; moved = True
        elif key == ord("3") and mode == "xyz_quat":
            # Pitch up: rotate around camera-local Y axis
            quat_wxyz = _quat_mul(quat_wxyz, _quat_from_axis_angle([0.0, 1.0, 0.0], rot_step))
            quat_updated = True
        elif key == ord("4") and mode == "xyz_quat":
            # Pitch down: rotate around camera-local Y axis
            quat_wxyz = _quat_mul(quat_wxyz, _quat_from_axis_angle([0.0, 1.0, 0.0], -rot_step))
            quat_updated = True

        if not moved and not quat_updated:
            continue
        if quat_updated:
            quat_wxyz = _quat_normalize(quat_wxyz)

        # Build action
        if mode == "xy_cos_sin":
            action = np.array([
                delta[0] / action_scale, delta[1] / action_scale,
                math.cos(yaw), math.sin(yaw),
            ], dtype=np.float32)
        elif mode == "xyz_quat":
            action = np.array([
                delta[0] / action_scale, delta[1] / action_scale, delta[2] / action_scale,
                quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
            ], dtype=np.float32)
        elif mode == "xy":
            action = np.array([delta[0] / action_scale, delta[1] / action_scale], dtype=np.float32)
        else:
            action = delta / action_scale

        prev_pos = np.asarray(getattr(env, "_position", np.zeros(3)), dtype=np.float32)
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as exc:
            print(f"[traj_sampling] step failed: {exc}")
            continue
        render_state.prev_position = prev_pos

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"[traj_sampling] episode {reason}; resetting.")
            obs, _ = env.reset()
            yaw = float(getattr(env, "_yaw", 0.0))
            quat_wxyz = np.asarray(
                getattr(env, "_camera_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])),
                dtype=np.float32,
            )

        time.sleep(0.001)

    # --- Save on exit ---
    env.close()
    cv2.destroyAllWindows()

    if saved_views:
        out_path = _save_poses(
            views=saved_views,
            output_dir=args.output_dir,
            scene_id=args.scene_id,
            method=args.method,
            run_id=args.run_id,
            fov_deg=fov_deg,
            image_width=args.width,
            image_height=args.height,
        )
        print(f"[traj_sampling] saved {len(saved_views)} poses to {out_path}")
    else:
        print("[traj_sampling] no poses recorded; nothing saved.")


if __name__ == "__main__":
    main()
