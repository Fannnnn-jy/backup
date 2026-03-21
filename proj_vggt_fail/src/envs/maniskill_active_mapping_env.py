from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

from .active_mapping_env import ActiveMappingConfig, ActiveMappingEnv
from .patchify import mark_patch_visited
from .voxel_carving import carve_depth_to_voxels, intrinsics_from_fov


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
        obs = self._build_observation()
        info = {"coverage": self._coverage_ratio()}
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
        terminated = crash
        reward = 0.0
        coverage_complete = False

        reward += self.cfg.step_reward
        if crash:
            reward += self.cfg.crash_penalty
        else:
            # Always update path visitation for visualization/debugging.
            for path_idx in path_patches:
                if self._valid_mask is None or self._valid_mask[path_idx]:
                    mark_patch_visited(self._visited, path_idx)

            reward_by = getattr(self.cfg, "reward_by", "location")
            if reward_by == "sight":
                sight_mask = self._sight_reward_mask()
                if sight_mask is not None:
                    reward_mask = (
                        sight_mask
                        if sight_mask.shape == self._reward_visited.shape
                        else self._downsample_mask(sight_mask, self._reward_visited.shape)
                    )
                    if self._reward_valid_mask is not None:
                        reward_mask = reward_mask & self._reward_valid_mask
                    newly = reward_mask & ~self._reward_visited
                    reward += float(newly.sum()) * self.cfg.grid_reward
                    self._reward_visited |= reward_mask
            else:
                iter_patches = path_patches if self.cfg.reward_traverse else [patch_idx]
                for path_idx in iter_patches:
                    if self._valid_mask is None or self._valid_mask[path_idx]:
                        reward_idx = self._patch_to_reward_idx(path_idx)
                        if self._reward_valid_mask is None or self._reward_valid_mask[reward_idx]:
                            if mark_patch_visited(self._reward_visited, reward_idx):
                                reward += self.cfg.grid_reward

            if self._reward_visited_3d is not None and self._reward_grid_shape_3d is not None:
                traverse_3d = bool(getattr(self.cfg, "reward_grid_3d_traverse", True))
                positions = (
                    self._collect_traversed_positions(prev_position, resolved_position)
                    if traverse_3d
                    else [resolved_position]
                )
                newly_3d = 0
                for pos in positions:
                    voxel_idx = self._position_to_voxel(pos)
                    reward_idx = self._voxel_to_reward3d_idx(voxel_idx)
                    if self._reward_valid_mask_3d is not None and not self._reward_valid_mask_3d[reward_idx]:
                        continue
                    if not self._reward_visited_3d[reward_idx]:
                        self._reward_visited_3d[reward_idx] = True
                        print("using 3D reward")
                        newly_3d += 1
                weight_3d = float(getattr(self.cfg, "reward_grid_3d_weight", 0.0))
                if weight_3d != 0.0 and newly_3d > 0:
                    reward += float(newly_3d) * weight_3d

            if self._reward_valid_mask is None:
                if self._reward_visited.all():
                    terminated = True
                    coverage_complete = True
            else:
                all_valid_visited = np.logical_or(self._reward_visited, ~self._reward_valid_mask).all()
                if all_valid_visited:
                    terminated = True
                    coverage_complete = True

            if self._reward_visited_3d is not None:
                if self._reward_valid_mask_3d is None:
                    all_valid_visited_3d = self._reward_visited_3d.all()
                else:
                    all_valid_visited_3d = np.logical_or(self._reward_visited_3d, ~self._reward_valid_mask_3d).all()
                if all_valid_visited_3d:
                    coverage_complete = True

        truncated = self._step_count >= self.cfg.max_steps
        self._update_camera_poses()
        obs = self._build_observation()
        out_of_bounds = crash_reason == "out_of_bounds"
        collision = crash_reason == "collision"
        info = {
            "coverage": self._coverage_ratio(),
            "crash": crash,
            "out_of_bounds": out_of_bounds,
            "collision": collision,
            "coverage_complete": coverage_complete,
            "patch_index": None if crash else patch_idx,
        }
        if self._reward_visited_3d is not None:
            info["coverage_3d"] = self._coverage_ratio_3d()
        self._maybe_debug_step(action, patch_idx, crash, terminated, truncated)
        # print("step reward:", reward)
        # print("current position:", self._position)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._cameras = []
        self._scene = None
        self._render_device = ""

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

    def _sight_reward_mask(self) -> np.ndarray | None:
        if self._scene is None or not self._cameras:
            return None
        if self._range_gt is None or self._voxel_size is None:
            return None
        grid_size = int(self._grid_shape_3d[0]) if self._grid_shape_3d is not None else int(self._grid_shape[0])
        occ = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        free = np.zeros_like(occ)
        carve_buffer = self.cfg.maniskill_depth_buffer or ""
        depth_is_z = _depth_is_z_for_buffer(carve_buffer)
        depth_max = float(self.cfg.maniskill_camera_far or self.cfg.depth_data_max_value)
        for cam in self._cameras:
            cam_pose = _camera_pose(cam)
            if cam_pose is None:
                continue
            cam_pos, cam_quat = cam_pose
            depth_raw = self._read_camera_depth(cam, target_name=carve_buffer or None)
            depth = _prepare_depth_for_carving(depth_raw, self.cfg.depth_image_shape)
            h, w = depth.shape[:2]
            if hasattr(cam, "get_intrinsic_matrix"):
                K = cam.get_intrinsic_matrix()
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            else:
                fov_rad = math.radians(float(self.cfg.maniskill_camera_fov))
                fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_rad)
            depth_scale = 1.0
            max_obs = float(np.nanmax(depth)) if np.isfinite(depth).any() else 0.0
            if depth_max > 1.0 and max_obs <= 1.0 + 1e-3:
                depth_scale = depth_max
            masks = carve_depth_to_voxels(
                depth,
                cam_pos,
                cam_quat,
                self._range_gt,
                self._voxel_size,
                grid_size,
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
                ground_z=0.0,
            )
            occ |= masks.occupied
            free |= masks.free
        seen_xy = np.any(occ | free, axis=2)
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

    def _get_camera_poses(self) -> List[np.ndarray]:
        poses: List[np.ndarray] = []
        for cam in self._cameras:
            cam_pose = _camera_pose(cam)
            if cam_pose is None:
                continue
            pos, quat = cam_pose
            quat = _normalize_quat_wxyz(quat)
            poses.append(np.concatenate([pos.astype(np.float32), quat], axis=0))
        if getattr(self.cfg, "single_camera_obs", False) and poses:
            poses = [poses[0]]
        if len(poses) > self.cfg.max_depth_frames:
            poses = poses[: self.cfg.max_depth_frames]
        return poses

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
