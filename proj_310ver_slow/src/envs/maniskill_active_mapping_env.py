from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

from .active_mapping_env import ActiveMappingConfig, ActiveMappingEnv
from .patchify import mark_patch_visited
from .voxel_carving import VoxelMasks, carve_depth_to_voxels, depth_to_world_points, intrinsics_from_fov


def _import_sapien():
    try:
        import sapien
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sapien is required for ManiSkillActiveMappingEnv. Install maniskill3 and its sapien dependency."
        ) from exc
    return sapien


def _quat_from_yaw(yaw_rad: float) -> Tuple[float, float, float, float]:
    """
    直接构造绕 Z 轴旋转的四元数 (W-XYZ 格式)
    Yaw 0 对应于相机初始朝向
    """
    w = math.cos(yaw_rad / 2.0)
    z = math.sin(yaw_rad / 2.0)
    if w < 0:
        w, z = -w, -z
    # 返回 (w, x, y, z)
    return (w, 0.0, 0.0, z)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q[:4]
    norm = float(np.linalg.norm(q))
    if norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q.astype(np.float32)


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # Hamilton product in (w, x, y, z) format.
    lw, lx, ly, lz = _normalize_quat_wxyz(lhs)
    rw, rx, ry, rz = _normalize_quat_wxyz(rhs)
    out = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )
    return _normalize_quat_wxyz(out)


def _depth_is_z_for_buffer(buffer_name: str) -> bool:
    name = buffer_name.lower()
    if "pointdepth" in name or "linedepth" in name:
        return False
    return True


def _prepare_depth_for_carving(depth: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
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


def _resolve_depth_scale(depth: np.ndarray, depth_max: float, buffer_name: str) -> float:
    name = str(buffer_name or "").strip().lower()
    if name:
        # For explicit render buffers such as DepthLinear / Position we treat values as already metric.
        return 1.0
    max_obs = float(np.nanmax(depth)) if np.isfinite(depth).any() else 0.0
    if depth_max > 1.0 and max_obs <= 1.0 + 1e-3:
        return float(depth_max)
    return 1.0


def _camera_pose(cam) -> Tuple[np.ndarray, np.ndarray] | None:
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


class ManiSkillActiveMappingEnv(ActiveMappingEnv):
    """Active mapping environment with depth rendered via ManiSkill/SAPIEN."""

    def __init__(self, cfg: ActiveMappingConfig, render_mode: Optional[str] = None):
        self._sapien = _import_sapien()

        camera_count = len(cfg.maniskill_camera_angles) if cfg.maniskill_camera_angles else 0
        if camera_count > 0 and cfg.max_depth_frames > camera_count:
            # Avoid padding empty depth frames when camera count is smaller than max_depth_frames.
            cfg.max_depth_frames = camera_count
            # pass

        self._scene = None
        self._scene_index_loaded: Optional[int] = None
        self._cameras: List[object] = []
        self._camera_angles = tuple(cfg.maniskill_camera_angles)
        self._camera_height = float(cfg.maniskill_camera_height)
        self._camera_axis_forward = cfg.maniskill_camera_axis_forward
        self._render_device = cfg.maniskill_render_device
        self._covered_target_voxels: Optional[np.ndarray] = None
        self._episode_return = 0.0
        self._episode_rewarded_steps = 0
        self._episode_visible_ratio_sum = 0.0
        self._episode_new_visible_ratio_sum = 0.0
        self._episode_overlap_ratio_sum = 0.0
        self._episode_overlap_in_band_steps = 0
        self._episode_zero_new_coverage_steps = 0
        self._episode_positive_new_coverage_steps = 0

        super().__init__(cfg, render_mode)

        if self._gleam_scene_index is None:
            self._gleam_scene_index = 0
        self._load_scene(self._gleam_scene_index)
        self._update_camera_poses()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        if self._use_glb_data:
            if self.cfg.glb_scene_index < 0 and self.cfg.gleam_resample_scene:
                self._set_glb_scene(self._select_initial_scene())
        elif self._gleam_dataset is not None and self.cfg.gleam_scene_index < 0 and self.cfg.gleam_resample_scene:
            self._set_gleam_scene(self._select_initial_scene())
        if self._gleam_scene_index is None:
            self._gleam_scene_index = 0
        if self._scene_index_loaded != self._gleam_scene_index:
            self._load_scene(self._gleam_scene_index)

        self._position = self._sample_init_pose()
        if getattr(self.cfg, "init_pose_resample", True) and not self._is_valid_position(self._position):
            candidate = self._sample_valid_position(int(getattr(self.cfg, "init_pose_max_tries", 200)))
            if candidate is not None:
                print("[reset] init pose invalid, resampled.")
                self._position = candidate
            else:
                print("[reset] init pose invalid; resample failed, using original pose.")
        self._init_position = self._position.copy()
        self._yaw = math.radians(float(self.cfg.maniskill_camera_yaw_init))
        # self._position = np.array([4.8, -4.8], dtype=np.float64)
        # print(self._position)
        # assert False, "debug"
        self._visited = np.zeros(self._grid_shape, dtype=bool)
        self._reward_visited = np.zeros(self._reward_grid_shape, dtype=bool)
        self._update_camera_poses()
        self._reset_target_coverage_state()
        self._reset_episode_reward_stats()
        obs = self._build_observation()
        accumulated_coverage_ratio = self._accumulated_target_coverage_ratio()
        info = {
            "coverage": accumulated_coverage_ratio,
            "accumulated_coverage_ratio": accumulated_coverage_ratio,
            "reward_total": 0.0,
            "reward_step_penalty": 0.0,
            "reward_crash": 0.0,
            "reward_collision": 0.0,
            "reward_coverage_new": 0.0,
            "reward_visibility": 0.0,
            "reward_overlap": 0.0,
            "visible_ratio": 0.0,
            "new_visible_ratio": 0.0,
            "overlap_ratio": 0.0,
            "crash_reason": "none",
            "collision": False,
            "out_of_bounds": False,
            "is_collision": False,
            "is_out_of_bounds": False,
            "episode_step_index": 0,
            "is_first_rewarded_step": False,
            "is_zero_reward_init_frame": bool(getattr(self.cfg, "reward_skip_first_frame", True)),
        }
        # self._maybe_debug_reset()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)
        try:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        except Exception:
            pass
        prev_position = self._position.copy()
        next_position = prev_position.copy()
        mode = getattr(self.cfg, "action_mode", None)
        if mode is None:
            mode = "xy" if self.cfg.xy_only else "xyz"
        if mode == "xy_cos_sin":
            if action.size < 4:
                raise ValueError(f"Expected 4D action for xy_cos_sin, got shape {action.shape}.")
            next_position[:2] = prev_position[:2] + action[:2] * self.cfg.action_scale
            self._yaw = float(math.atan2(action[3], action[2]))
        elif mode in ("xyz_quat", "xyz_delta_quat"):
            if action.size < 7:
                raise ValueError(f"Expected 7D action for {mode}, got shape {action.shape}.")
            next_position = prev_position + action[:3] * self.cfg.action_scale
            quat = _normalize_quat_wxyz(action[3:7])
            if mode == "xyz_delta_quat":
                current_quat = _normalize_quat_wxyz(
                    getattr(self, "_camera_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                )
                quat = _quat_mul_wxyz(current_quat, quat)
            if float(np.linalg.norm(quat)) > 1e-6:
                if bool(getattr(self.cfg, "lock_camera_pitch_roll", False)):
                    # Keep yaw only: force quaternion x/y to zero and renormalize w/z.
                    quat[1] = 0.0
                    quat[2] = 0.0
                    wz_norm = float(np.linalg.norm(quat[[0, 3]]))
                    if wz_norm > 1e-6:
                        quat[0] /= wz_norm
                        quat[3] /= wz_norm
                    else:
                        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                self._camera_quat_wxyz = _normalize_quat_wxyz(quat)
            elif not hasattr(self, "_camera_quat_wxyz"):
                self._camera_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif mode == "xy":
            if action.size < 2:
                raise ValueError(f"Expected 2D action for xy, got shape {action.shape}.")
            next_position[:2] = prev_position[:2] + action[:2] * self.cfg.action_scale
        else:
            if action.size < 3:
                raise ValueError(f"Expected 3D action for xyz, got shape {action.shape}.")
            next_position = prev_position + action[:3] * self.cfg.action_scale
        crash, crash_reason, resolved_position, patch_idx, path_patches = self._sweep_for_collision(
            prev_position, next_position
        )
        self._position = resolved_position
        self._update_camera_poses()
        terminated = crash
        coverage_complete = False
        # Keep path visitation for visualization/debugging, but reward now comes from
        # privileged-depth voxel coverage instead of grid visits.
        for path_idx in path_patches:
            if self._valid_mask is None or self._valid_mask[path_idx]:
                mark_patch_visited(self._visited, path_idx)

        collision = crash_reason == "collision"
        out_of_bounds = crash_reason == "out_of_bounds"
        reward_breakdown = self._compute_reward_breakdown(crash_reason=crash_reason if crash else None)
        reward = reward_breakdown["reward_total"]
        self._update_episode_reward_stats(reward_breakdown)
        accumulated_coverage_ratio = reward_breakdown["accumulated_coverage_ratio"]
        if accumulated_coverage_ratio >= 1.0 - max(float(getattr(self.cfg, "reward_eps", 1e-6)), 1e-12):
            terminated = True
            coverage_complete = True

        truncated = self._step_count >= self.cfg.max_steps
        obs = self._build_observation()
        episode_step_index = int(self._step_count)
        is_first_rewarded_step = episode_step_index == 1
        info = {
            "coverage": accumulated_coverage_ratio,
            "crash": crash,
            "out_of_bounds": out_of_bounds,
            "collision": collision,
            "is_out_of_bounds": out_of_bounds,
            "is_collision": collision,
            "crash_reason": crash_reason or "none",
            "coverage_complete": coverage_complete,
            "patch_index": None if crash else patch_idx,
            "episode_step_index": episode_step_index,
            "is_first_rewarded_step": is_first_rewarded_step,
            "is_zero_reward_init_frame": False,
            **reward_breakdown,
        }
        if terminated or truncated:
            info["episode"] = self._build_episode_summary(
                collision=collision,
                out_of_bounds=out_of_bounds,
                terminated=terminated,
                truncated=truncated,
                accumulated_coverage_ratio=accumulated_coverage_ratio,
            )
        self._maybe_debug_step(action, patch_idx, crash, terminated, truncated)
        # print("step reward:", reward)
        # print("current position:", self._position)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._cameras = []
        self._scene = None
        self._render_device = ""

    def _target_voxel_mask(self) -> np.ndarray | None:
        if self._obstacle_mask_3d is None:
            return None
        return np.asarray(self._obstacle_mask_3d, dtype=bool)

    def _prepare_target_coverage_mask(self, mask: np.ndarray | None, shape: Tuple[int, int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(shape, dtype=bool)
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != shape:
            return np.zeros(shape, dtype=bool)
        return arr

    def _current_visible_target_voxels(self) -> np.ndarray | None:
        target_mask = self._target_voxel_mask()
        if target_mask is None:
            return None
        current_visible = np.zeros_like(target_mask, dtype=bool)
        try:
            masks = self.get_current_view_voxel_masks(target_name=self.cfg.maniskill_depth_buffer or None)
        except Exception:
            masks = None
        if masks is not None:
            current_visible = np.asarray(masks.occupied, dtype=bool)
            if current_visible.shape != target_mask.shape:
                current_visible = np.zeros_like(target_mask, dtype=bool)
        return current_visible & target_mask

    def _reset_target_coverage_state(self) -> None:
        target_mask = self._target_voxel_mask()
        if target_mask is None:
            self._covered_target_voxels = None
            return
        if bool(getattr(self.cfg, "reward_skip_first_frame", True)):
            current_visible = self._current_visible_target_voxels()
            if current_visible is not None:
                self._covered_target_voxels = current_visible.copy()
                return
        self._covered_target_voxels = np.zeros_like(target_mask, dtype=bool)

    def _reset_episode_reward_stats(self) -> None:
        self._episode_return = 0.0
        self._episode_rewarded_steps = 0
        self._episode_visible_ratio_sum = 0.0
        self._episode_new_visible_ratio_sum = 0.0
        self._episode_overlap_ratio_sum = 0.0
        self._episode_overlap_in_band_steps = 0
        self._episode_zero_new_coverage_steps = 0
        self._episode_positive_new_coverage_steps = 0

    def _update_episode_reward_stats(self, reward_breakdown: dict[str, float]) -> None:
        self._episode_return += float(reward_breakdown.get("reward_total", 0.0))
        self._episode_rewarded_steps += 1
        self._episode_visible_ratio_sum += float(reward_breakdown.get("visible_ratio", 0.0))
        self._episode_new_visible_ratio_sum += float(reward_breakdown.get("new_visible_ratio", 0.0))
        self._episode_overlap_ratio_sum += float(reward_breakdown.get("overlap_ratio", 0.0))

        low = float(getattr(self.cfg, "reward_overlap_low", 0.30))
        high = float(getattr(self.cfg, "reward_overlap_high", 0.50))
        if low > high:
            low, high = high, low
        overlap_ratio = float(reward_breakdown.get("overlap_ratio", 0.0))
        if low <= overlap_ratio <= high:
            self._episode_overlap_in_band_steps += 1

        new_visible_ratio = float(reward_breakdown.get("new_visible_ratio", 0.0))
        eps = max(float(getattr(self.cfg, "reward_eps", 1e-6)), 1e-12)
        if new_visible_ratio <= eps:
            self._episode_zero_new_coverage_steps += 1
        if new_visible_ratio > eps:
            self._episode_positive_new_coverage_steps += 1

    def _build_episode_summary(
        self,
        *,
        collision: bool,
        out_of_bounds: bool,
        terminated: bool,
        truncated: bool,
        accumulated_coverage_ratio: float,
    ) -> dict[str, float]:
        step_count = max(0, int(self._episode_rewarded_steps))
        denom = float(step_count) if step_count > 0 else 1.0
        ended_by_max_step = bool(truncated and not terminated)
        # These episode summaries are logged through the existing rsl_rl episode-info path
        # and are intended for reward debugging / parameter tuning.
        return {
            "episode_return": float(self._episode_return),
            "episode_length": float(step_count),
            "final_accumulated_coverage_ratio": float(accumulated_coverage_ratio),
            "mean_visible_ratio": float(self._episode_visible_ratio_sum / denom) if step_count > 0 else 0.0,
            "mean_new_visible_ratio": float(self._episode_new_visible_ratio_sum / denom) if step_count > 0 else 0.0,
            "mean_overlap_ratio": float(self._episode_overlap_ratio_sum / denom) if step_count > 0 else 0.0,
            "fraction_steps_overlap_in_band": (
                float(self._episode_overlap_in_band_steps) / denom if step_count > 0 else 0.0
            ),
            "num_collision_terminations": float(1.0 if collision else 0.0),
            "num_out_of_bounds_terminations": float(1.0 if out_of_bounds else 0.0),
            "ended_by_max_step": float(1.0 if ended_by_max_step else 0.0),
            "num_zero_new_coverage_steps": float(self._episode_zero_new_coverage_steps),
            "num_positive_new_coverage_steps": float(self._episode_positive_new_coverage_steps),
        }

    def _accumulated_target_coverage_ratio(self) -> float:
        target_mask = self._target_voxel_mask()
        if target_mask is None:
            return 0.0
        total_target = int(target_mask.sum())
        if total_target <= 0:
            return 0.0
        covered = self._prepare_target_coverage_mask(self._covered_target_voxels, target_mask.shape) & target_mask
        eps = max(float(getattr(self.cfg, "reward_eps", 1e-6)), 1e-12)
        return float(covered.sum()) / float(max(float(total_target), eps))

    def _overlap_band_score(self, overlap_ratio: float) -> float:
        low = float(getattr(self.cfg, "reward_overlap_low", 0.30))
        high = float(getattr(self.cfg, "reward_overlap_high", 0.50))
        if low > high:
            low, high = high, low
        if overlap_ratio < low:
            return -float((low - overlap_ratio) ** 2)
        if overlap_ratio <= high:
            return 1.0
        return -float((overlap_ratio - high) ** 2)

    def _compute_reward_breakdown(self, *, crash_reason: str | None) -> dict[str, float]:
        step_penalty = float(self.cfg.step_reward)
        reward_crash = float(self.cfg.crash_penalty) if crash_reason in ("collision", "out_of_bounds") else 0.0
        coverage_reward = 0.0
        visibility_reward = 0.0
        overlap_reward = 0.0
        visible_ratio = 0.0
        new_visible_ratio = 0.0
        overlap_ratio = 0.0

        target_mask = self._target_voxel_mask()
        accumulated_coverage_ratio = 0.0
        if target_mask is not None:
            total_target = int(target_mask.sum())
            prev_union = self._prepare_target_coverage_mask(self._covered_target_voxels, target_mask.shape) & target_mask
            next_union = prev_union
            eps = max(float(getattr(self.cfg, "reward_eps", 1e-6)), 1e-12)
            if total_target > 0 and crash_reason is None:
                current_visible = self._current_visible_target_voxels()
                if current_visible is None:
                    current_visible = np.zeros_like(target_mask, dtype=bool)
                visible_count = int(current_visible.sum())
                new_visible = current_visible & ~prev_union
                overlap_count = int(np.logical_and(current_visible, prev_union).sum())

                visible_ratio = float(visible_count) / float(max(float(total_target), eps))
                new_visible_ratio = float(new_visible.sum()) / float(max(float(total_target), eps))
                if visible_count > 0:
                    overlap_ratio = float(overlap_count) / float(max(float(visible_count), eps))

                coverage_reward = float(getattr(self.cfg, "reward_w_new", 1.0)) * new_visible_ratio
                visibility_reward = float(getattr(self.cfg, "reward_w_visible", 0.1)) * visible_ratio
                overlap_reward = float(getattr(self.cfg, "reward_w_overlap", 0.2)) * self._overlap_band_score(
                    overlap_ratio
                )
                next_union = prev_union | current_visible
            elif total_target > 0:
                overlap_reward = float(getattr(self.cfg, "reward_w_overlap", 0.2)) * self._overlap_band_score(
                    overlap_ratio
                )

            self._covered_target_voxels = next_union.copy()
            accumulated_coverage_ratio = float(next_union.sum()) / float(max(float(total_target), eps))
        reward_total = step_penalty + reward_crash + coverage_reward + visibility_reward + overlap_reward
        return {
            "reward_total": float(reward_total),
            "reward_step_penalty": step_penalty,
            "reward_crash": reward_crash,
            "reward_collision": reward_crash,
            "reward_coverage_new": float(coverage_reward),
            "reward_visibility": float(visibility_reward),
            "reward_overlap": float(overlap_reward),
            "visible_ratio": float(visible_ratio),
            "new_visible_ratio": float(new_visible_ratio),
            "overlap_ratio": float(overlap_ratio),
            "accumulated_coverage_ratio": float(accumulated_coverage_ratio),
        }

    def _load_scene(self, scene_index: int) -> None:
        self._scene_index_loaded = int(scene_index)
        self._scene = self._create_scene()
        if hasattr(self._scene, "set_timestep"):
            self._scene.set_timestep(self.cfg.maniskill_time_step)
        if hasattr(self._scene, "set_ambient_light"):
            self._scene.set_ambient_light([0.5, 0.5, 0.5])
        if hasattr(self._scene, "add_ground"):
            self._scene.add_ground(0.0)

        if self._use_glb_data:
            glb_path = self._resolve_scene_glb(self._scene_index_loaded)
            builder = self._scene.create_actor_builder()
            pose = self._sapien.Pose()
            if self.cfg.glb_y_up:
                theta = math.radians(90.0)
                q = np.array([math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
                pose = self._sapien.Pose(q=q)
            scale = [float(self.cfg.glb_scale)] * 3
            builder.add_visual_from_file(str(glb_path), pose=pose, scale=scale)
            builder.build_static(name="glb_scene")
        else:
            urdf_path = self._resolve_scene_urdf(self._scene_index_loaded)
            loader = self._scene.create_urdf_loader()
            if hasattr(loader, "fix_root_link"):
                loader.fix_root_link = True
            if hasattr(loader, "load_multiple_collisions_from_file"):
                loader.load_multiple_collisions_from_file = True
            if hasattr(loader, "load_multiple_visuals_from_file"):
                loader.load_multiple_visuals_from_file = True
            if hasattr(loader, "load_multiple"):
                loader.load_multiple(str(urdf_path))
            else:
                try:
                    loader.load(str(urdf_path))
                except Exception as exc:  # pragma: no cover
                    if "multiple objects" in str(exc).lower():
                        loader.load_multiple(str(urdf_path))
                    else:
                        raise

        self._setup_cameras()

    def _create_scene(self):
        device_alias = self._render_device or "cuda:0"
        try:
            render_system = self._sapien.render.RenderSystem(device_alias)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to create render system for device '{device_alias}'. "
                "Set maniskill_render_device to a valid SAPIEN device alias "
                "(see sapien.render.get_device_summary())."
            ) from exc
        physx_system = self._sapien.physx.PhysxCpuSystem()
        return self._sapien.Scene([physx_system, render_system])

    def _resolve_scene_urdf(self, scene_index: int) -> Path:
        if self.cfg.maniskill_urdf_dir:
            urdf_dir = Path(self.cfg.maniskill_urdf_dir)
        elif self.cfg.gleam_data_dir:
            urdf_dir = Path(self.cfg.gleam_data_dir) / "urdf"
        else:
            raise ValueError("gleam_data_dir or maniskill_urdf_dir must be set for ManiSkill.")
        urdf_path = urdf_dir / f"scene_{scene_index}.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"Scene URDF not found: {urdf_path}")
        return urdf_path

    def _resolve_scene_glb(self, scene_index: int) -> Path:
        if not self._glb_scene_paths:
            raise ValueError("GLB scene list is empty; set glb_data_dir.")
        idx = int(np.clip(scene_index, 0, len(self._glb_scene_paths) - 1))
        path = self._glb_scene_paths[idx]
        if not path.exists():
            raise FileNotFoundError(f"Scene GLB not found: {path}")
        return path

    def _setup_cameras(self) -> None:
        self._cameras = []
        height, width = self.cfg.depth_image_shape
        fov = math.radians(self.cfg.maniskill_camera_fov)
        for idx, _ in enumerate(self._camera_angles):
            cam = self._scene.add_camera(
                f"depth_cam_{idx}",
                width,
                height,
                fov,
                self.cfg.maniskill_camera_near,
                self.cfg.maniskill_camera_far,
            )
            self._cameras.append(cam)

    def _update_camera_poses(self) -> None:
        if self._scene is None:
            return
        z_pos = (
            float(self._position[2])
            if self._position is not None and self._position.shape[0] >= 3
            else float(self._camera_height)
        )
        cam_pos = np.array([self._position[0], self._position[1], z_pos], dtype=np.float32)
        mode = getattr(self.cfg, "action_mode", "xyz")
        for angle, cam in zip(self._camera_angles, self._cameras):
            if mode == "xyz_quat" and hasattr(self, "_camera_quat_wxyz"):
                quat_wxyz = tuple(np.asarray(self._camera_quat_wxyz, dtype=np.float32).tolist())
            else:
                if mode == "xy_cos_sin" and hasattr(self, "_yaw"):
                    yaw_rad = float(self._yaw)
                else:
                    yaw_rad = math.radians(float(angle))
                quat_wxyz = _quat_from_yaw(yaw_rad)
            if angle == self._camera_angles[0]:
                self._camera_quat_wxyz = np.array(quat_wxyz, dtype=np.float32)

            quat = quat_wxyz
            pose = self._sapien.Pose(cam_pos, quat)

            if hasattr(cam, "set_pose"):
                cam.set_pose(pose)
            elif hasattr(cam, "set_local_pose"):
                cam.set_local_pose(pose)
            else:  # pragma: no cover
                raise RuntimeError("Camera pose setter not found for SAPIEN camera.")

    def _is_valid_position(self, position: np.ndarray) -> bool:
        if self._range_gt is None or self._voxel_size is None:
            return True
        crash, _, _, _, _ = self._sweep_for_collision(position, position)
        return not crash

    def _sample_valid_position(self, max_tries: int) -> Optional[np.ndarray]:
        if max_tries <= 0:
            return None
        if self._obstacle_mask_3d is not None and self._range_gt is not None and self._voxel_size is not None:
            grid = self._obstacle_mask_3d
            nx, ny, nz = grid.shape
            x_min = float(self._range_gt[1])
            y_min = float(self._range_gt[3])
            z_min = float(self._range_gt[5])
            vx, vy, vz = self._voxel_size[:3]
            for _ in range(max_tries):
                ix = int(self.rng.integers(0, nx))
                iy = int(self.rng.integers(0, ny))
                iz = int(self.rng.integers(0, nz))
                if grid[ix, iy, iz]:
                    continue
                pos = np.array(
                    [x_min + ix * vx, y_min + iy * vy, z_min + iz * vz],
                    dtype=np.float32,
                )
                if self._is_valid_position(pos):
                    return pos
        if self._bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
            for _ in range(max_tries):
                pos = np.array(
                    [
                        self.rng.uniform(x_min, x_max),
                        self.rng.uniform(y_min, y_max),
                        self.rng.uniform(z_min, z_max),
                    ],
                    dtype=np.float32,
                )
                if self._is_valid_position(pos):
                    return pos
        return None

    def _get_depth_images(self) -> List[np.ndarray]:
        depth_images = self._render_depth_images()
        if getattr(self.cfg, "single_camera_obs", False) and depth_images:
            depth_images = [depth_images[0]]
        if len(depth_images) > self.cfg.max_depth_frames:
            depth_images = depth_images[: self.cfg.max_depth_frames]
        if not depth_images:
            depth_images = [
                self.rng.random(self.cfg.depth_image_shape, dtype=np.float32) for _ in range(self.cfg.max_depth_frames)
            ]
        return depth_images

    def _coverage_cameras(self) -> List:
        cameras = list(self._cameras)
        if getattr(self.cfg, "single_camera_obs", False) and cameras:
            return [cameras[0]]
        return cameras

    def _camera_intrinsics_for_carving(self, cam, width: int, height: int) -> Tuple[float, float, float, float]:
        if hasattr(cam, "get_intrinsic_matrix"):
            try:
                K = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float32)
            except Exception:
                K = None
            if K is not None and K.shape[0] >= 3 and K.shape[1] >= 3:
                return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        fov_rad = math.radians(float(self.cfg.maniskill_camera_fov))
        return intrinsics_from_fov(width, height, fov_rad)

    def get_current_view_voxel_masks(
        self,
        *,
        target_name: str | None = None,
        ground_z: float | None = 0.0,
    ) -> VoxelMasks | None:
        if self._scene is None or not self._cameras:
            return None
        if self._range_gt is None or self._voxel_size is None:
            return None
        if hasattr(self._scene, "update_render"):
            self._scene.update_render()

        if self._grid_shape_3d is not None:
            grid_shape = tuple(int(v) for v in self._grid_shape_3d)
        else:
            grid_shape = (int(self._grid_shape[0]), int(self._grid_shape[1]), 1)
        occ = np.zeros(grid_shape, dtype=bool)
        free = np.zeros_like(occ)

        used_camera = False
        carve_buffer = target_name if target_name is not None else (self.cfg.maniskill_depth_buffer or "")
        depth_is_z = _depth_is_z_for_buffer(carve_buffer)
        depth_max = float(self.cfg.maniskill_camera_far or self.cfg.depth_data_max_value)
        for cam in self._coverage_cameras():
            if hasattr(cam, "take_picture"):
                cam.take_picture()
            cam_pose = _camera_pose(cam)
            if cam_pose is None:
                continue
            cam_pos, cam_quat = cam_pose
            depth_raw = self._read_camera_depth(cam, target_name=carve_buffer or None)
            depth = _prepare_depth_for_carving(depth_raw, self.cfg.depth_image_shape)
            h, w = depth.shape[:2]
            fx, fy, cx, cy = self._camera_intrinsics_for_carving(cam, w, h)
            depth_scale = _resolve_depth_scale(depth, depth_max, carve_buffer)
            masks = carve_depth_to_voxels(
                depth,
                cam_pos,
                cam_quat,
                self._range_gt,
                self._voxel_size,
                grid_shape,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                forward_axis=self.cfg.maniskill_camera_axis_forward,
                depth_is_z=depth_is_z,
                depth_scale=depth_scale,
                depth_max=depth_max,
                depth_max_epsilon=depth_max * 0.01,
                depth_max_is_no_hit=True,
                ground_z=ground_z,
            )
            used_camera = True
            occ |= masks.occupied
            free |= masks.free

        if not used_camera:
            return None
        free[occ] = False
        unknown = ~(occ | free)
        return VoxelMasks(occupied=occ, free=free, unknown=unknown)

    def get_current_view_pointcloud(
        self,
        *,
        target_name: str | None = None,
    ) -> np.ndarray:
        if self._scene is None or not self._cameras:
            return np.zeros((0, 3), dtype=np.float32)
        if hasattr(self._scene, "update_render"):
            self._scene.update_render()

        carve_buffer = target_name if target_name is not None else (self.cfg.maniskill_depth_buffer or "")
        depth_is_z = _depth_is_z_for_buffer(carve_buffer)
        depth_max = float(self.cfg.maniskill_camera_far or self.cfg.depth_data_max_value)
        points_list: List[np.ndarray] = []
        for cam in self._coverage_cameras():
            if hasattr(cam, "take_picture"):
                cam.take_picture()
            cam_pose = _camera_pose(cam)
            if cam_pose is None:
                continue
            cam_pos, cam_quat = cam_pose
            depth_raw = self._read_camera_depth(cam, target_name=carve_buffer or None)
            depth = _prepare_depth_for_carving(depth_raw, self.cfg.depth_image_shape)
            h, w = depth.shape[:2]
            fx, fy, cx, cy = self._camera_intrinsics_for_carving(cam, w, h)
            depth_scale = _resolve_depth_scale(depth, depth_max, carve_buffer)
            points_world, _ = depth_to_world_points(
                depth,
                cam_pos,
                cam_quat,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                forward_axis=self.cfg.maniskill_camera_axis_forward,
                depth_is_z=depth_is_z,
                depth_scale=depth_scale,
                depth_max=depth_max,
                depth_max_is_no_hit=True,
                depth_max_epsilon=depth_max * 0.01,
                pose_is_camera_to_world=True,
            )
            if points_world.size > 0:
                points_list.append(points_world)
        if not points_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(points_list, axis=0).astype(np.float32)

    def _sight_reward_mask(self) -> np.ndarray | None:
        masks = self.get_current_view_voxel_masks()
        if masks is None:
            return None
        seen_xy = np.any(masks.occupied | masks.free, axis=2)
        seen_rc = np.flip(seen_xy.T, 0)
        return seen_rc

    def _render_depth_images(self) -> List[np.ndarray]:
        if self._scene is None or not self._cameras:
            return []
        if hasattr(self._scene, "update_render"):
            self._scene.update_render()
        depth_images: List[np.ndarray] = []
        target_name = self.cfg.maniskill_depth_buffer or None
        for cam in self._cameras:
            if hasattr(cam, "take_picture"):
                cam.take_picture()
            depth = self._read_camera_depth(cam, target_name=target_name)
            depth_images.append(self._prepare_depth_frame(depth))
        return depth_images

    def _render_rgb_images(self) -> List[np.ndarray]:
        if self._scene is None or not self._cameras:
            return []
        if hasattr(self._scene, "update_render"):
            self._scene.update_render()
        rgb_images: List[np.ndarray] = []
        for cam in self._cameras:
            if hasattr(cam, "take_picture"):
                cam.take_picture()
            rgb = self._read_camera_color(cam)
            if rgb is None:
                continue
            rgb_images.append(rgb)
        return rgb_images

    def _get_rgb_images(self) -> List[np.ndarray]:
        rgb_images = self._render_rgb_images()
        if getattr(self.cfg, "single_camera_obs", False) and rgb_images:
            rgb_images = [rgb_images[0]]
        if len(rgb_images) > self.cfg.max_depth_frames:
            rgb_images = rgb_images[: self.cfg.max_depth_frames]
        return rgb_images

    def _read_camera_color(self, cam) -> np.ndarray | None:
        if hasattr(cam, "get_picture"):
            names = ["Color", "color", "RGB", "rgb", "Albedo", "albedo", "BaseColor", "basecolor"]
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

    def _read_camera_depth(self, cam, target_name=None) -> np.ndarray:
        buffer_name = target_name or self.cfg.maniskill_depth_buffer or None
        if hasattr(cam, "get_picture"):
            names = ["Depth", "depth", "Position", "position"]
            if hasattr(cam, "get_picture_names"):
                try:
                    candidate_names = cam.get_picture_names()
                except Exception:
                    candidate_names = None
                if candidate_names:
                    filtered = [
                        name for name in candidate_names if "depth" in name.lower() or "position" in name.lower()
                    ]
                    if filtered:
                        names = filtered
            if buffer_name:
                names = [buffer_name] + [name for name in names if name != buffer_name]

            for name in names:
                texture = self._get_camera_texture(cam, name)
                if texture is None:
                    continue
                self._last_depth_buffer_name = name
                return texture
        if hasattr(cam, "get_depth"):
            depth = cam.get_depth()
            self._last_depth_buffer_name = "get_depth"
            return np.asarray(depth)

        raise RuntimeError("Unsupported camera depth API for SAPIEN.")

    def _get_camera_texture(self, cam, name: str) -> np.ndarray | None:
        try:
            texture = cam.get_picture(name)
        except Exception:
            return None
        if texture is None:
            return None
        arr = np.asarray(texture)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if "position" in name.lower() and arr.ndim == 3 and arr.shape[-1] >= 3:
            return np.abs(arr[..., 2])
        if "depth" in name.lower() and arr.ndim == 3 and arr.shape[-1] > 1:
            flat = arr.reshape(-1, arr.shape[-1])
            mins = np.nanmin(flat, axis=0)
            maxs = np.nanmax(flat, axis=0)
            ranges = maxs - mins
            idx = int(np.nanargmax(ranges))
            return arr[..., idx]
        if arr.ndim == 3 and arr.shape[-1] > 1:
            return arr[..., 0]
        return arr

    def _prepare_depth_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        frame = np.nan_to_num(frame, nan=0.0, neginf=0.0, posinf=0.0)
        if frame.ndim == 3:
            frame = frame[..., 0]
        # OpenGL-style buffers (e.g. DepthLinear) may encode forward depth as negative values.
        if frame.size > 0 and np.nanmax(frame) <= 0.0 and np.nanmin(frame) < 0.0:
            frame = -frame
        frame = np.clip(frame, 0.0, self.cfg.depth_data_max_value)
        target_h, target_w = self.cfg.depth_image_shape
        if frame.shape != (target_h, target_w):
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Pillow is required to resize depth images. Install it via `pip install pillow`."
                ) from exc
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
            frame = np.array(pil_img, dtype=np.float32)
        return frame
