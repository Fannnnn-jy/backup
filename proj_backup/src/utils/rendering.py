from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass
class RenderState:
    prev_position: np.ndarray | None = None
    last_heading: float | None = None
    carved_occupied: np.ndarray | None = None
    carved_free: np.ndarray | None = None
    show_error_overlay: bool = False


def init_render_state(cfg: Any) -> RenderState:
    grid_size = int(cfg.gleam_grid_size)
    carved_occupied = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    carved_free = np.zeros_like(carved_occupied)
    return RenderState(
        prev_position=None,
        last_heading=None,
        carved_occupied=carved_occupied,
        carved_free=carved_free,
        show_error_overlay=False,
    )


def reset_render_state(state: RenderState) -> None:
    if state.carved_occupied is not None:
        state.carved_occupied.fill(False)
    if state.carved_free is not None:
        state.carved_free.fill(False)
    state.show_error_overlay = False
    state.prev_position = None
    state.last_heading = None


def _normalize_depth(frames: np.ndarray, max_value: float) -> np.ndarray:
    frames = np.nan_to_num(frames, nan=0.0, neginf=0.0, posinf=0.0)
    max_val = float(max_value)
    max_obs = float(np.nanmax(frames)) if np.isfinite(frames).any() else 1.0
    if max_val <= 0.0:
        max_val = max_obs if max_obs > 0.0 else 1.0
    if max_val > 1.0 and max_obs <= 1.0 + 1e-3:
        max_val = 1.0
    frames = np.clip(frames / max_val, 0.0, 1.0)
    return (frames * 255.0).astype(np.uint8)


def _tile_frames(frames: np.ndarray) -> np.ndarray:
    num_frames, height, width = frames.shape
    grid = int(math.ceil(math.sqrt(num_frames)))
    tiled = np.zeros((grid * height, grid * width), dtype=frames.dtype)
    for idx in range(num_frames):
        row = idx // grid
        col = idx % grid
        tiled[row * height : (row + 1) * height, col * width : (col + 1) * width] = frames[idx]
    return tiled


def _tile_color_frames(frames: np.ndarray) -> np.ndarray:
    num_frames, height, width, channels = frames.shape
    grid = int(math.ceil(math.sqrt(num_frames)))
    tiled = np.zeros((grid * height, grid * width, channels), dtype=frames.dtype)
    for idx in range(num_frames):
        row = idx // grid
        col = idx % grid
        tiled[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = frames[idx]
    return tiled


def _print_depth_stats(frames: np.ndarray, step_idx: int, eps: float = 1e-6) -> None:
    if frames.ndim != 3:
        return
    means = frames.mean(axis=(1, 2))
    stds = frames.std(axis=(1, 2))
    mins = frames.min(axis=(1, 2))
    maxs = frames.max(axis=(1, 2))
    stats = [
        f"{idx}:mean={means[idx]:.4f},std={stds[idx]:.4f},min={mins[idx]:.4f},max={maxs[idx]:.4f}"
        for idx in range(frames.shape[0])
    ]
    print(f"[depth-stats step={step_idx}] {' | '.join(stats)}")


def _extract_depth(obs: Any) -> np.ndarray:
    depth = obs["obs_depth_tensor"]
    try:
        import torch

        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
    except Exception:
        pass
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[0]
    return depth


def _read_camera_color(cam: Any) -> np.ndarray | None:
    if hasattr(cam, "get_picture"):
        names = []
        if hasattr(cam, "get_picture_names"):
            try:
                candidate_names = list(cam.get_picture_names())
            except Exception:
                candidate_names = None
            if candidate_names:
                preferred = [
                    name
                    for name in candidate_names
                    if any(tag in name.lower() for tag in ("color", "rgb", "albedo", "basecolor"))
                ]
                names = preferred or candidate_names
        if not names:
            names = [
                "Color",
                "color",
                "RGB",
                "rgb",
                "Albedo",
                "albedo",
                "BaseColor",
                "basecolor",
            ]
        for name in names:
            try:
                texture = cam.get_picture(name)
            except Exception:
                continue
            if texture is None:
                continue
            arr = np.asarray(texture)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                return arr[..., :3]
    if hasattr(cam, "get_color"):
        try:
            arr = np.asarray(cam.get_color())
        except Exception:
            arr = None
        if arr is not None and arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3]
    return None


def _prepare_rgb_frame(frame: np.ndarray) -> np.ndarray | None:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return None
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.nan_to_num(rgb, nan=0.0, neginf=0.0, posinf=0.0)
        max_val = float(np.nanmax(rgb)) if np.isfinite(rgb).any() else 1.0
        if max_val <= 1.0 + 1e-3:
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        else:
            rgb = np.clip(rgb, 0.0, 255.0)
        rgb = rgb.astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _infer_heading(env: Any, state: RenderState) -> tuple[float | None, np.ndarray | None]:
    heading: float | None = None
    current_position: np.ndarray | None = None
    if hasattr(env, "_position"):
        current_position = np.asarray(env._position, dtype=np.float32)
        if hasattr(env, "_heading"):
            heading = float(env._heading)
        elif hasattr(env, "_yaw"):
            heading = float(env._yaw)
        elif state.prev_position is not None:
            delta = current_position - state.prev_position
            if float(np.linalg.norm(delta)) > 1e-5:
                heading = math.atan2(float(delta[1]), float(delta[0]))
                state.last_heading = heading
            else:
                heading = state.last_heading
    if heading is None and state.last_heading is None:
        cam_angles = getattr(env, "_camera_angles", None)
        if cam_angles:
            heading = math.radians(float(cam_angles[0]))
            state.last_heading = heading
    elif heading is not None:
        state.last_heading = heading
    return heading, current_position


def _render_depth_image(
    obs: Any,
    *,
    depth_max: float,
    first_camera_only: bool,
    stats_interval: int,
    step_idx: int,
) -> np.ndarray | None:
    if obs is None or not isinstance(obs, dict) or "obs_depth_tensor" not in obs:
        return None
    depth = _extract_depth(obs)
    if depth.size == 0:
        return None
    depth_u8 = _normalize_depth(depth, depth_max)
    if depth_u8.ndim == 3 and depth_u8.shape[-1] == 1:
        depth_u8 = depth_u8[..., 0]
    if depth_u8.ndim == 2:
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    elif first_camera_only:
        colored = cv2.applyColorMap(depth_u8[0], cv2.COLORMAP_TURBO)
    else:
        tiled = _tile_frames(depth_u8)
        colored = cv2.applyColorMap(tiled, cv2.COLORMAP_TURBO)
    if stats_interval > 0 and step_idx % stats_interval == 0:
        _print_depth_stats(depth, step_idx)
    return colored


def _render_rgb_image(env: Any, *, first_camera_only: bool) -> np.ndarray | None:
    cameras = getattr(env, "_cameras", None) or []
    if not cameras:
        return None
    scene = getattr(env, "_scene", None)
    if scene is not None and hasattr(scene, "update_render"):
        scene.update_render()
    frames: list[np.ndarray] = []
    for cam in cameras:
        if hasattr(cam, "take_picture"):
            cam.take_picture()
        rgb = _read_camera_color(cam)
        if rgb is None:
            continue
        frame = _prepare_rgb_frame(rgb)
        if frame is None:
            continue
        frames.append(frame)
        if first_camera_only:
            break
    if not frames:
        return None
    if first_camera_only or len(frames) == 1:
        return frames[0]
    target_h, target_w = frames[0].shape[:2]
    aligned = []
    for frame in frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        aligned.append(frame)
    stacked = np.stack(aligned, axis=0)
    return _tile_color_frames(stacked)


def render_depth_and_topdown(
    obs: Any,
    env: Any,
    *,
    state: RenderState,
    depth_max: float = 8.0,
    topdown_scale: int = 4,
    view: str = "both",
    image_mode: str = "depth",
    first_camera_only: bool = False,
    stats_interval: int = -1,
    step_idx: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Returns (image_bgr, topdown_bgr, current_position).
    For stable heading estimation, set state.prev_position after stepping the env.
    """
    heading, current_position = _infer_heading(env, state)
    depth_img = None
    topdown_img = None
    if view in ("depth", "both"):
        if image_mode == "rgb":
            depth_img = _render_rgb_image(env, first_camera_only=first_camera_only)
        else:
            depth_img = _render_depth_image(
                obs,
                depth_max=depth_max,
                first_camera_only=first_camera_only,
                stats_interval=stats_interval,
                step_idx=step_idx,
            )
    if view in ("topdown", "both"):
        topdown_img = _render_topdown(
            env,
            topdown_scale,
            heading,
            carved_occupied=state.carved_occupied,
            carved_free=state.carved_free,
            show_error_overlay=state.show_error_overlay,
        )
    return depth_img, topdown_img, current_position


def _expand_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    rows_s, cols_s = mask.shape
    rows_t, cols_t = target_shape
    out = np.zeros(target_shape, dtype=bool)
    for r in range(rows_s):
        r0 = int(math.floor(r * rows_t / rows_s))
        r1 = int(math.floor((r + 1) * rows_t / rows_s))
        if r1 <= r0:
            r1 = min(r0 + 1, rows_t)
        for c in range(cols_s):
            if not mask[r, c]:
                continue
            c0 = int(math.floor(c * cols_t / cols_s))
            c1 = int(math.floor((c + 1) * cols_t / cols_s))
            if c1 <= c0:
                c1 = min(c0 + 1, cols_t)
            out[r0:r1, c0:c1] = True
    return out


def _render_topdown(
    env: Any,
    scale: int,
    heading: float | None = None,
    *,
    carved_occupied: np.ndarray | None = None,
    carved_free: np.ndarray | None = None,
    show_error_overlay: bool = False,
) -> np.ndarray | None:
    # OpenCV stores arrays as [H, W, C], which swaps row/col vs world coords.
    visited = getattr(env, "_visited", None)
    if visited is None:
        return None
    reward_visited = getattr(env, "_reward_visited_full", None)
    if reward_visited is None:
        reward_visited = getattr(env, "_reward_visited", None)
    if reward_visited is not None and reward_visited.shape != visited.shape:
        # assert False, "reward_visited shape mismatch"
        reward_visited = _expand_mask(reward_visited, visited.shape)
    obstacle = getattr(env, "_obstacle_mask", None)
    if obstacle is None:
        obstacle = np.zeros_like(visited, dtype=bool)
    obstacle_3d = getattr(env, "_obstacle_mask_3d", None)
    full_block = None
    partial_block = None
    if obstacle_3d is not None:
        try:
            min_ratio = 0.1
            occ_ratio = np.mean(obstacle_3d, axis=2)
            full_xy = np.all(obstacle_3d, axis=2)
            partial_xy = np.logical_and(occ_ratio >= min_ratio, ~full_xy)
            full_rc = np.flip(full_xy.T, 0)
            full_block = full_rc
            partial_block = np.flip(partial_xy.T, 0)
        except Exception:
            full_block = None
            partial_block = None
    if full_block is None:
        full_block = obstacle
        partial_block = np.zeros_like(visited, dtype=bool)
    img = np.full((*visited.shape, 3), 255, dtype=np.uint8)
    if reward_visited is not None:
        img[reward_visited] = (180, 180, 180)
    img[partial_block] = (100, 100, 100)
    img[full_block] = (0, 0, 0)
    img[visited] = (0, 255, 0)

    if carved_free is not None:
        try:
            free_xy = np.any(carved_free, axis=2)
            free_rc = np.flip(free_xy.T, 0)
            img[free_rc] = (180, 80, 255)
        except Exception:
            pass
    if carved_occupied is not None:
        try:
            occ_xy = np.any(carved_occupied, axis=2)
            occ_rc = np.flip(occ_xy.T, 0)
            img[occ_rc] = (180, 0, 180)
        except Exception:
            pass
    if show_error_overlay and carved_occupied is not None and carved_free is not None and obstacle_3d is not None:
        try:
            false_pos = carved_occupied & ~obstacle_3d
            false_neg = carved_free & obstacle_3d
            fp_xy = np.any(false_pos, axis=2)
            fn_xy = np.any(false_neg, axis=2)
            fp_rc = np.flip(fp_xy.T, 0)
            fn_rc = np.flip(fn_xy.T, 0)
            img[fp_rc] = (0, 128, 255)
        except Exception:
            pass

    try:
        row, col = env._position_to_patch(tuple(env._position))
        img[row, col] = (0, 0, 255)
    except Exception:
        pass

    scale = max(1, int(scale))
    height, width = img.shape[:2]
    scaled = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    return scaled
