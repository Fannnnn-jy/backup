from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import cv2


import trimesh

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:  # pragma: no cover
    torch = None
    torch_f = None
from .gleam_dataset import GleamDataset
from .patchify import (
    coverage_ratio,
    position_to_patch_range,
)


@dataclass
class ActiveMappingConfig:
    """Environment configuration and constants."""

    grid_size: Tuple[int, int] = (1, 1)
    world_size: Tuple[float, float] = (10.0, 10.0)
    world_size_z: float = 3.0
    max_steps: int = 500
    # patch_reward: float = 1.0
    grid_reward: float = 1.0
    step_reward: float = -0.1
    crash_penalty: float = -10.0
    action_scale: float = 0.1
    action_bound: float = 1.0
    xy_only: bool = False
    reward_traverse: bool = True
    action_mode: str = "xyz"  # xyz | xy | xy_cos_sin | xyz_quat | xyz_delta_quat
    single_camera_obs: bool = False
    reward_by: str = "location"  # location | sight
    max_depth_frames: int = 4
    depth_image_shape: Tuple[int, int] = (64, 64)
    depth_data_max_value: float = 18.0
    seed: int = 1

    sim_backend: str = "maniskill"
    gleam_data_dir: str = ""
    gleam_dataset_name: str = ""
    gleam_grid_size: int = 128
    reward_grid_size: Optional[Tuple[int, int]] = (16, 16)
    reward_grid_3d_enabled: bool = False
    reward_grid_3d_size: Optional[int] = None
    reward_grid_3d_weight: float = 0.0
    reward_grid_3d_traverse: bool = True
    gleam_scene_index: int = -1
    gleam_resample_scene: bool = True  # 是否开启场景随机重采样
    gleam_use_init_pose: bool = True  # 是否开启场景初始化位置随机
    maniskill_time_step: float = 1.0 / 60.0
    maniskill_camera_height: float = 1.5
    maniskill_camera_fov: float = 90.0
    maniskill_camera_near: float = 0.1
    maniskill_camera_far: float = 18.0
    maniskill_depth_buffer: str = ""
    agent_init_height: float = 1.5
    maniskill_camera_angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    maniskill_camera_yaw_range: Tuple[float, float] = (0.0, 360.0)
    maniskill_camera_yaw_init: float = 0.0
    maniskill_camera_yaw_delta: float = 10.0
    maniskill_camera_axis_forward: Optional[Tuple[float, float, float]] = None
    lock_camera_pitch_roll: bool = False
    maniskill_urdf_dir: str = ""
    maniskill_render_device: str = ""
    maniskill_shader_pack: str = "minimal"
    debug_print_interval: int = 1
    init_pose_resample: bool = True
    init_pose_max_tries: int = 200
    collision_distance_threshold: float = 0.0  # depricated, use 0 to disable
    collision_dilation: int = 0  # 用 dilation 方式来扩张 obstacle mask，从而处理碰撞逻辑
    carve_min_z: float = 1.0
    glb_data_dir: str = ""
    glb_scene_index: int = -1
    glb_scene_glob: str = "*.glb"
    glb_y_up: bool = True  # 场景坐标旋转，适配 Y-up 的 GLB 模型
    glb_scale: float = 1.0
    glb_voxel_cache_suffix: str = ".voxels.h5"
    glb_voxel_pitch: float = 0.0  # 指定场景体素化体素尺寸，非正则由场景范围和  glb_voxel_max_dim 共同决定
    glb_voxel_max_dim: int = (
        128  # 上述体素尺寸为 0.0 时生效，指定场景体素化最大边长的体素数量，最终体素尺寸 max_length / glb_voxel_max_dim
    )
    init_position: Optional[Tuple[float, ...]] = None


class ActiveMappingEnv(gym.Env):
    """Minimal Active Mapping environment stub for rsl_rl integration.

    Replace the depth image sampling with ManiSkill sensors in production.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: ActiveMappingConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(cfg.seed)
        self._use_glb_data = bool(cfg.glb_data_dir)
        self._glb_scene_paths: List[Path] = []
        self._gleam_dataset: Optional[GleamDataset] = None
        self._gleam_scene_index: Optional[int] = None
        self._range_gt: Optional[np.ndarray] = None
        self._voxel_size: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None
        self._obstacle_mask: Optional[np.ndarray] = None
        self._obstacle_mask_3d: Optional[np.ndarray] = None
        self._obstacle_centers: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._state_center = np.zeros(3, dtype=np.float32)
        self._state_scale = np.ones(3, dtype=np.float32)
        self._grid_shape = cfg.grid_size
        self._grid_shape_3d: Optional[Tuple[int, int, int]] = None
        self._reward_grid_shape: Tuple[int, int] = cfg.grid_size
        self._reward_visited = np.zeros(self._reward_grid_shape, dtype=bool)
        self._reward_valid_mask: Optional[np.ndarray] = None
        self._reward_grid_shape_3d: Optional[Tuple[int, int, int]] = None
        self._reward_visited_3d: Optional[np.ndarray] = None
        self._reward_valid_mask_3d: Optional[np.ndarray] = None
        self._depth_paths: List[Path] = []
        self._depth_cursor = 0

        self._init_gleam_dataset()

        self._step_count = 0
        self._position = np.zeros(3, dtype=np.float32)
        self._init_position = None
        self._visited = np.zeros(self._grid_shape, dtype=bool)
        self._init_reward_grid()

        depth_shape = (cfg.max_depth_frames, *cfg.depth_image_shape)
        depth_high = float(cfg.depth_data_max_value) if cfg.depth_data_max_value > 0 else 1.0
        rgb_shape = (cfg.max_depth_frames, *cfg.depth_image_shape, 3)
        self.observation_space = gym.spaces.Dict({
            "obs_depth_tensor": gym.spaces.Box(
                low=0.0,
                high=depth_high,
                shape=depth_shape,
                dtype=np.float32,
            ),
            "obs_rgb_tensor": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=rgb_shape,
                dtype=np.float32,
            ),
            "obs_embedding": gym.spaces.Box(
                low=0.0,
                high=depth_high,
                shape=cfg.depth_image_shape,
                dtype=np.float32,
            ),
            "agent_state": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            ),
            "obs_pose": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
            "obs_cam_pose": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.max_depth_frames, 7),
                dtype=np.float32,
            ),
        })

        mode = getattr(self.cfg, "action_mode", None)
        if mode is None:
            mode = "xy" if self.cfg.xy_only else "xyz"
        if mode == "xy_cos_sin":
            bound_xy = float(self.cfg.action_bound)
            low = np.array([-bound_xy, -bound_xy, -1.0, -1.0], dtype=np.float32)
            high = np.array([bound_xy, bound_xy, 1.0, 1.0], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif mode in ("xyz_quat", "xyz_delta_quat"):
            bound_xyz = float(self.cfg.action_bound)
            low = np.array(
                [-bound_xyz, -bound_xyz, -bound_xyz, -1.0, -1.0, -1.0, -1.0],
                dtype=np.float32,
            )
            high = np.array(
                [bound_xyz, bound_xyz, bound_xyz, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            action_dim = 2 if mode == "xy" else 3
            bound = float(self.cfg.action_bound)
            self.action_space = gym.spaces.Box(low=-bound, high=bound, shape=(action_dim,), dtype=np.float32)

    def _init_gleam_dataset(self) -> None:
        if self.cfg.glb_data_dir:
            self._init_glb_dataset()
            return
        if not self.cfg.gleam_data_dir:
            return
        dataset_name = self.cfg.gleam_dataset_name or Path(self.cfg.gleam_data_dir).name
        self._gleam_dataset = GleamDataset(self.cfg.gleam_data_dir, dataset_name, self.cfg.gleam_grid_size)
        self._grid_shape = (self.cfg.gleam_grid_size, self.cfg.gleam_grid_size)
        self._grid_shape_3d = (
            self.cfg.gleam_grid_size,
            self.cfg.gleam_grid_size,
            self.cfg.gleam_grid_size,
        )
        self._set_gleam_scene(self._select_initial_scene())

    def _resolve_reward_grid_shape(self) -> Tuple[int, int]:
        target = self.cfg.reward_grid_size
        if target is None:
            return self._grid_shape
        if isinstance(target, int):
            return (int(target), int(target))
        return (int(target[0]), int(target[1]))

    def _resolve_reward_grid_shape_3d(self) -> Optional[Tuple[int, int, int]]:
        if not bool(getattr(self.cfg, "reward_grid_3d_enabled", False)):
            return None
        target = getattr(self.cfg, "reward_grid_3d_size", None)
        if target is None:
            return self._grid_shape_3d
        size = int(target)
        return (size, size, size)

    def _init_reward_grid(self) -> None:
        self._reward_grid_shape = self._resolve_reward_grid_shape()
        self._reward_visited = np.zeros(self._reward_grid_shape, dtype=bool)
        self._init_reward_grid_3d()
        self._update_reward_valid_mask()

    def _init_reward_grid_3d(self) -> None:
        self._reward_grid_shape_3d = self._resolve_reward_grid_shape_3d()
        if self._reward_grid_shape_3d is None:
            self._reward_visited_3d = None
            self._reward_valid_mask_3d = None
            return
        self._reward_visited_3d = np.zeros(self._reward_grid_shape_3d, dtype=bool)
        self._reward_valid_mask_3d = None

    def _update_reward_valid_mask(self) -> None:
        if self._valid_mask is None:
            self._reward_valid_mask = None
        else:
            if self._valid_mask.shape == self._reward_grid_shape:
                self._reward_valid_mask = self._valid_mask
            else:
                self._reward_valid_mask = self._downsample_mask(self._valid_mask, self._reward_grid_shape)
        if self._reward_grid_shape_3d is None:
            self._reward_valid_mask_3d = None
            return
        if self._obstacle_mask_3d is None:
            self._reward_valid_mask_3d = None
            return
        valid_3d = ~self._obstacle_mask_3d
        if valid_3d.shape == self._reward_grid_shape_3d:
            self._reward_valid_mask_3d = valid_3d
        else:
            self._reward_valid_mask_3d = self._downsample_mask_3d(valid_3d, self._reward_grid_shape_3d)

    @staticmethod
    def _downsample_mask_3d(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        if mask.shape == target_shape:
            return mask
        z_t, y_t, x_t = target_shape
        z_s, y_s, x_s = mask.shape
        out = np.zeros((z_t, y_t, x_t), dtype=bool)
        for iz in range(z_t):
            z0 = int(np.floor(iz * z_s / z_t))
            z1 = int(np.ceil((iz + 1) * z_s / z_t))
            for iy in range(y_t):
                y0 = int(np.floor(iy * y_s / y_t))
                y1 = int(np.ceil((iy + 1) * y_s / y_t))
                for ix in range(x_t):
                    x0 = int(np.floor(ix * x_s / x_t))
                    x1 = int(np.ceil((ix + 1) * x_s / x_t))
                    out[iz, iy, ix] = bool(np.any(mask[z0:z1, y0:y1, x0:x1]))
        return out

    def _voxel_to_reward3d_idx(self, voxel_idx: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if self._reward_grid_shape_3d is None:
            return voxel_idx
        if self._grid_shape_3d is None or self._reward_grid_shape_3d == self._grid_shape_3d:
            return voxel_idx
        vx, vy, vz = voxel_idx
        sx, sy, sz = self._grid_shape_3d
        tx, ty, tz = self._reward_grid_shape_3d
        rx = int(vx * tx / max(1, sx))
        ry = int(vy * ty / max(1, sy))
        rz = int(vz * tz / max(1, sz))
        rx = int(np.clip(rx, 0, tx - 1))
        ry = int(np.clip(ry, 0, ty - 1))
        rz = int(np.clip(rz, 0, tz - 1))
        return (rx, ry, rz)

    def _collect_traversed_positions(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        delta = end - start
        dist = float(np.linalg.norm(delta))
        step = self._collision_check_step()
        steps = max(1, int(math.ceil(dist / step)))
        return [start + delta * (idx / steps) for idx in range(1, steps + 1)]

    @staticmethod
    def _downsample_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask.shape == target_shape:
            return mask
        rows_f, cols_f = mask.shape
        rows_t, cols_t = target_shape
        out = np.zeros(target_shape, dtype=bool)
        for r in range(rows_t):
            r0 = int(math.floor(r * rows_f / rows_t))
            r1 = int(math.floor((r + 1) * rows_f / rows_t))
            if r1 <= r0:
                r1 = min(r0 + 1, rows_f)
            for c in range(cols_t):
                c0 = int(math.floor(c * cols_f / cols_t))
                c1 = int(math.floor((c + 1) * cols_f / cols_t))
                if c1 <= c0:
                    c1 = min(c0 + 1, cols_f)
                out[r, c] = bool(np.any(mask[r0:r1, c0:c1]))
        return out

    def _patch_to_reward_idx(self, patch_idx: Tuple[int, int]) -> Tuple[int, int]:
        if self._reward_grid_shape == self._grid_shape:
            return patch_idx
        row, col = patch_idx
        rows_f, cols_f = self._grid_shape
        rows_r, cols_r = self._reward_grid_shape
        row_r = int(row * rows_r / rows_f)
        col_r = int(col * cols_r / cols_f)
        row_r = int(np.clip(row_r, 0, rows_r - 1))
        col_r = int(np.clip(col_r, 0, cols_r - 1))
        return (row_r, col_r)

    def _select_initial_scene(self) -> int:
        if self.cfg.glb_data_dir:
            if self.cfg.glb_scene_index >= 0:
                return int(self.cfg.glb_scene_index)
            if self._glb_scene_paths:
                return int(self.rng.integers(0, len(self._glb_scene_paths)))
            return 0
        if self.cfg.gleam_scene_index >= 0:
            return int(self.cfg.gleam_scene_index)
        if self._gleam_dataset is None:
            return 0
        return int(self.rng.integers(0, self._gleam_dataset.num_scenes))

    def _set_gleam_scene(self, scene_index: int) -> None:
        if self._gleam_dataset is None:
            return
        scene = self._gleam_dataset.get_scene(scene_index)
        self._gleam_scene_index = int(scene_index)
        self._range_gt = scene.range_gt.astype(np.float32)
        self._voxel_size = scene.voxel_size.astype(np.float32)
        self._valid_mask = None
        self._obstacle_mask = None
        self._obstacle_mask_3d = None
        if scene.layout_mask is not None:
            obstacle_mask = scene.layout_mask.astype(np.uint8)
            obstacle_mask = cv2.transpose(obstacle_mask)
            obstacle_mask = cv2.flip(obstacle_mask, 0)

            # visualization
            # img = np.full((128, 128, 3), 200, dtype=np.uint8)
            # img[obstacle_mask] = (0, 0, 0)
            # cv2.imshow("mask", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            dilation_threshold = getattr(self.cfg, "collision_dilation", 0)
            if dilation_threshold > 0:
                kernel = np.ones((dilation_threshold, dilation_threshold), np.uint8)
                obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

            obstacle_mask = obstacle_mask.astype(bool)
            self._obstacle_mask = obstacle_mask

            self._valid_mask = ~obstacle_mask
        self._obstacle_mask_3d = scene.obstacle_mask_3d
        if self._obstacle_mask_3d is not None:
            dilation_threshold = getattr(self.cfg, "collision_dilation", 0)
            if dilation_threshold > 0:
                self._obstacle_mask_3d = self._dilate_mask_3d(self._obstacle_mask_3d, dilation_threshold)

        self._obstacle_centers = None

        self._finalize_scene()

    def _init_glb_dataset(self) -> None:
        glb_dir = Path(self.cfg.glb_data_dir)
        if not glb_dir.exists():
            raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
        pattern = self.cfg.glb_scene_glob or "*.glb"
        paths = sorted([p for p in glb_dir.glob(pattern) if p.suffix.lower() == ".glb"])
        if not paths:
            raise FileNotFoundError(f"No GLB files found in {glb_dir} with pattern '{pattern}'.")
        self._glb_scene_paths = paths
        self._grid_shape_3d = None
        self._set_glb_scene(self._select_initial_scene())

    @staticmethod
    def _compute_voxel_pitch(size: np.ndarray, pitch_arg: float, max_dim: int) -> float:
        if pitch_arg > 0:
            return float(pitch_arg)
        max_dim = max(8, int(max_dim))
        span = float(np.max(size)) if size.size else 1.0
        if span <= 0:
            span = 1.0
        return span / float(max_dim)

    def _load_glb_mesh_for_voxels(self, glb_path: Path) -> Optional["trimesh.Trimesh"]:
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover
            raise ImportError("trimesh is required to voxelize GLB scenes.") from exc

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

        if self.cfg.glb_y_up:
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
        if float(self.cfg.glb_scale) != 1.0:
            mesh = mesh.copy()
            mesh.apply_scale(float(self.cfg.glb_scale))
        return mesh

    def _save_glb_voxel_cache(
        self,
        cache_path: Path,
        glb_path: Path,
        grid: np.ndarray,
        origin: np.ndarray,
        pitch: float,
    ) -> None:
        try:
            import h5py
        except ImportError:
            print("[glb] h5py not available; voxel cache disabled.")
            return

        try:
            os.makedirs(cache_path.parent or Path("."), exist_ok=True)
            glb_mtime = os.path.getmtime(glb_path)
            glb_size = os.path.getsize(glb_path)
        except OSError:
            glb_mtime = None
            glb_size = None

        try:
            with h5py.File(str(cache_path), "w") as f:
                f.create_dataset("grid", data=grid.astype(np.uint8), compression="gzip", compression_opts=1)
                f.create_dataset("origin", data=np.asarray(origin, dtype=np.float32))
                f.create_dataset("pitch", data=np.asarray(pitch, dtype=np.float32))
                if glb_mtime is not None:
                    f.attrs["glb_mtime"] = float(glb_mtime)
                if glb_size is not None:
                    f.attrs["glb_size"] = int(glb_size)
                f.attrs["glb_path"] = str(glb_path)
                f.attrs["y_up"] = bool(self.cfg.glb_y_up)
                f.attrs["scale"] = float(self.cfg.glb_scale)
                f.attrs["voxel_pitch_arg"] = float(self.cfg.glb_voxel_pitch)
                f.attrs["voxel_max_dim"] = int(self.cfg.glb_voxel_max_dim)
            print(f"[glb] saved voxel cache: {cache_path}")
        except Exception as exc:
            print(f"[glb] failed to write voxel cache: {exc}")

    def _build_glb_voxel_cache(self, glb_path: Path, cache_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        mesh = self._load_glb_mesh_for_voxels(glb_path)
        if mesh is None:
            raise ValueError(f"Failed to load GLB mesh for voxelization: {glb_path}")

        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        pitch = self._compute_voxel_pitch(size, self.cfg.glb_voxel_pitch, self.cfg.glb_voxel_max_dim)
        pitch = max(1e-4, float(pitch))
        try:
            vox = mesh.voxelized(pitch)
        except Exception as exc:
            raise RuntimeError(f"Voxelization failed: {exc}") from exc

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
        pitch_arr = np.asarray(pitch_val, dtype=np.float32).reshape(-1) if pitch_val is not None else np.array([1.0])
        if pitch_arr.size >= 3 and not np.allclose(pitch_arr[:3], pitch_arr[0]):
            print(f"[glb] non-uniform pitch detected: {pitch_arr[:3]}")
        pitch = float(np.mean(pitch_arr[:3])) if pitch_arr.size >= 3 else float(pitch_arr[0])

        if grid.size == 0:
            raise ValueError(f"Empty voxel grid generated for {glb_path}")

        self._save_glb_voxel_cache(cache_path, glb_path, grid, origin, pitch)
        return grid, origin, pitch

    def _load_glb_voxel_cache(self, glb_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        suffix = self.cfg.glb_voxel_cache_suffix or ".voxels.h5"
        cache_path = Path(str(glb_path) + suffix)
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            print("[glb] h5py not available; will not cache voxelization.")
            h5py = None

        if cache_path.exists() and h5py is not None:
            try:
                glb_mtime = os.path.getmtime(glb_path)
                glb_size = os.path.getsize(glb_path)
            except OSError:
                glb_mtime = None
                glb_size = None

            try:
                with h5py.File(str(cache_path), "r") as f:
                    if "grid" not in f or "origin" not in f or "pitch" not in f:
                        raise ValueError("Missing datasets in voxel cache.")
                    attrs = f.attrs
                    if glb_mtime is not None and float(attrs.get("glb_mtime", -1.0)) != float(glb_mtime):
                        raise ValueError("glb mtime mismatch.")
                    if glb_size is not None and int(attrs.get("glb_size", -1)) != int(glb_size):
                        raise ValueError("glb size mismatch.")
                    y_up_attr = attrs.get("y_up", None)
                    if y_up_attr is not None and bool(y_up_attr) != bool(self.cfg.glb_y_up):
                        raise ValueError("y_up mismatch.")
                    scale_attr = attrs.get("scale", None)
                    if scale_attr is not None and float(scale_attr) != float(self.cfg.glb_scale):
                        raise ValueError("scale mismatch.")
                    grid = np.asarray(f["grid"][()], dtype=bool)
                    origin = np.asarray(f["origin"][()], dtype=np.float32)
                    pitch = float(np.asarray(f["pitch"][()]).reshape(()))
                if grid.size == 0:
                    raise ValueError("Empty voxel grid in cache.")
                print(f"[glb] loaded voxel cache: {cache_path}")
                return grid, origin, pitch
            except Exception as exc:
                print(f"[glb] cache invalid, rebuilding: {exc}")

        grid, origin, pitch = self._build_glb_voxel_cache(glb_path, cache_path)
        return grid, origin, pitch

    def _set_glb_scene(self, scene_index: int) -> None:
        if not self._glb_scene_paths:
            return
        idx = int(scene_index)
        idx = int(np.clip(idx, 0, len(self._glb_scene_paths) - 1))
        glb_path = self._glb_scene_paths[idx]
        self._gleam_scene_index = idx
        grid, origin, pitch = self._load_glb_voxel_cache(glb_path)

        nx, ny, nz = grid.shape
        voxel_size = np.array([pitch, pitch, pitch], dtype=np.float32)
        x_min = float(origin[0] + 0.5 * pitch)
        y_min = float(origin[1] + 0.5 * pitch)
        z_min = float(origin[2] + 0.5 * pitch)
        x_max = float(origin[0] + (nx - 0.5) * pitch)
        y_max = float(origin[1] + (ny - 0.5) * pitch)
        z_max = float(origin[2] + (nz - 0.5) * pitch)

        self._range_gt = np.array([x_max, x_min, y_max, y_min, z_max, z_min], dtype=np.float32)
        self._voxel_size = voxel_size
        self._valid_mask = None
        self._obstacle_mask = None
        self._obstacle_mask_3d = grid.astype(bool)
        self._grid_shape_3d = (nx, ny, nz)
        self._grid_shape = (ny, nx)
        self._finalize_scene()

    def _finalize_scene(self) -> None:
        if self._obstacle_mask_3d is not None:
            dilation_threshold = getattr(self.cfg, "collision_dilation", 0)
            if dilation_threshold > 0:
                self._obstacle_mask_3d = self._dilate_mask_3d(self._obstacle_mask_3d, dilation_threshold)
        self._obstacle_centers = None

        self._update_reward_valid_mask()

        x_max, x_min, y_max, y_min, z_max, z_min = self._range_gt
        self._bounds = (
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
            float(z_min),
            float(z_max),
        )
        center = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0],
            dtype=np.float32,
        )
        scale = np.array(
            [(x_max - x_min) / 2.0, (y_max - y_min) / 2.0, (z_max - z_min) / 2.0],
            dtype=np.float32,
        )
        scale = np.where(scale <= 0.0, 1.0, scale)
        self._state_center = center
        self._state_scale = scale

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """base reset function for active mapping envs."""
        raise NotImplementedError()

    def step(self, action: np.ndarray):
        """base step function for active mapping envs."""
        raise NotImplementedError()

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        if self._bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
            return bool(
                position[0] < x_min
                or position[0] > x_max
                or position[1] < y_min
                or position[1] > y_max
                or position[2] < z_min
                or position[2] > z_max
            )
        else:
            assert False, "没有提供世界边界！"

    def _collision_check_step(self) -> float:
        if self._voxel_size is not None:
            cell_x = float(self._voxel_size[0])
            cell_y = float(self._voxel_size[1])
            cell_z = float(self._voxel_size[2])
        else:
            cols, rows = self._grid_shape
            world_x, world_y = self.cfg.world_size
            cell_x = world_x / float(cols)
            cell_y = world_y / float(rows)
            cell_z = float(self.cfg.world_size_z) / float(max(rows, cols))
        step = 0.5 * min(cell_x, cell_y, cell_z)

        return max(step, 1e-4)

    def _sweep_for_collision(
        self, start_pos: np.ndarray, end_pos: np.ndarray
    ) -> Tuple[bool, str | None, np.ndarray, Tuple[int, int], List[Tuple[int, int]]]:
        start = np.asarray(start_pos, dtype=np.float32)
        end = np.asarray(end_pos, dtype=np.float32)
        delta = end - start
        dist = float(np.linalg.norm(delta))
        step = self._collision_check_step()
        steps = max(1, int(math.ceil(dist / step)))
        traversed: List[Tuple[int, int]] = []
        last_patch: Tuple[int, int] | None = None
        for idx in range(1, steps + 1):
            pos = start + delta * (idx / steps)
            patch_idx = self._position_to_patch(tuple(pos))
            if patch_idx != last_patch:
                traversed.append(patch_idx)
                last_patch = patch_idx
            if self._out_of_bounds(pos):
                return True, "out_of_bounds", pos, patch_idx, traversed
            if self._check_collision(pos, patch_idx):
                return True, "collision", pos, patch_idx, traversed
        patch_idx = self._position_to_patch(tuple(end))
        if patch_idx != last_patch:
            traversed.append(patch_idx)
        return False, None, end, patch_idx, traversed

    def _build_observation(self) -> Dict[str, np.ndarray]:
        depth_images = self._get_depth_images()
        obs_depth_tensor, obs_embedding = self._encode_depth_images(depth_images)
        rgb_images = self._get_rgb_images()
        obs_rgb_tensor = self._encode_rgb_images(rgb_images)
        cam_poses = self._get_camera_poses()
        obs_cam_pose = self._encode_camera_poses(cam_poses)
        agent_state = (self._position - self._state_center) / self._state_scale
        init_pos = self._init_position if self._init_position is not None else self._position
        rel_pos = (self._position - init_pos) / self._state_scale
        cam_quat = getattr(self, "_camera_quat_wxyz", None)
        if cam_quat is None:
            yaw = float(getattr(self, "_yaw", 0.0))
            cam_quat = np.array(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                dtype=np.float32,
            )
        obs_pose = np.concatenate([rel_pos.astype(np.float32), cam_quat.astype(np.float32)], axis=0)
        return {
            "obs_depth_tensor": obs_depth_tensor,
            "obs_rgb_tensor": obs_rgb_tensor,
            "obs_embedding": obs_embedding,
            "agent_state": agent_state.astype(np.float32),
            "obs_pose": obs_pose,
            "obs_cam_pose": obs_cam_pose,
        }

    def _get_depth_images(self) -> List[np.ndarray]:
        """Get depth images for the current position."""
        raise NotImplementedError()

    def _get_rgb_images(self) -> List[np.ndarray]:
        return []

    def _get_camera_poses(self) -> List[np.ndarray]:
        return []

    def _sample_init_pose(self) -> np.ndarray:
        init_pos = getattr(self.cfg, "init_position", None)
        if init_pos is not None:
            arr = np.asarray(init_pos, dtype=np.float32).reshape(-1)
            if arr.size == 2:
                default_z = self._default_init_z()
                return np.array([arr[0], arr[1], default_z], dtype=np.float32)
            if arr.size >= 3:
                return np.array([arr[0], arr[1], arr[2]], dtype=np.float32)
            raise ValueError("init_position must have 2 or 3 values.")
        default_z = self._default_init_z()
        if self._gleam_dataset is None or not self.cfg.gleam_use_init_pose:
            return np.array([0.0, 0.0, default_z], dtype=np.float32)
        if self._gleam_scene_index is None:
            return np.array([0.0, 0.0, default_z], dtype=np.float32)
        scene = self._gleam_dataset.get_scene(self._gleam_scene_index)
        if scene.init_positions is None or scene.init_positions.size == 0:
            return np.array([0.0, 0.0, default_z], dtype=np.float32)
        idx = int(self.rng.integers(0, scene.init_positions.shape[0]))
        xy = scene.init_positions[idx].astype(np.float32)
        return np.array([xy[0], xy[1], default_z], dtype=np.float32)

    def _position_to_patch(self, position: Tuple[float, float] | Tuple[float, float, float]) -> Tuple[int, int]:
        pos_xy = (float(position[0]), float(position[1]))
        assert self._range_gt is not None and self._voxel_size is not None, (
            "GLEAM range_gt/voxel_size required for position_to_patch."
        )
        return position_to_patch_range(pos_xy, tuple(self._range_gt), tuple(self._voxel_size), self._grid_shape)

    def _coverage_ratio(self) -> float:
        if self._reward_valid_mask is None:
            return coverage_ratio(self._reward_visited)
        total = int(self._reward_valid_mask.sum())
        if total == 0:
            return 0.0
        visited = np.logical_and(self._reward_visited, self._reward_valid_mask).sum()
        return float(visited) / float(total)

    def _coverage_ratio_3d(self) -> float:
        if self._reward_visited_3d is None:
            return 0.0
        if self._reward_valid_mask_3d is None:
            total = int(self._reward_visited_3d.size)
            visited = int(self._reward_visited_3d.sum())
        else:
            total = int(self._reward_valid_mask_3d.sum())
            visited = int(np.logical_and(self._reward_visited_3d, self._reward_valid_mask_3d).sum())
        if total == 0:
            return 0.0
        return float(visited) / float(total)

    def _compute_obstacle_centers(self, obstacle_mask: np.ndarray) -> np.ndarray | None:
        if obstacle_mask is None or obstacle_mask.size == 0:
            return None
        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return None
        if obstacle_mask.ndim == 3:
            xs_idx = indices[:, 0].astype(np.float32)
            ys_idx = indices[:, 1].astype(np.float32)
            zs_idx = indices[:, 2].astype(np.float32)
            if self._range_gt is not None and self._voxel_size is not None:
                x_min = float(self._range_gt[1])
                y_min = float(self._range_gt[3])
                z_min = float(self._range_gt[5])
                voxel_x = float(self._voxel_size[0])
                voxel_y = float(self._voxel_size[1])
                voxel_z = float(self._voxel_size[2])
                xs = x_min + xs_idx * voxel_x
                ys = y_min + ys_idx * voxel_y
                zs = z_min + zs_idx * voxel_z
            else:
                size = obstacle_mask.shape[0]
                world_x, world_y = self.cfg.world_size
                world_z = float(self.cfg.world_size_z)
                cell_x = world_x / float(size)
                cell_y = world_y / float(size)
                cell_z = world_z / float(size)
                x_min = -world_x / 2.0 + 0.5 * cell_x
                y_min = -world_y / 2.0 + 0.5 * cell_y
                z_min = -world_z / 2.0 + 0.5 * cell_z
                xs = x_min + xs_idx * cell_x
                ys = y_min + ys_idx * cell_y
                zs = z_min + zs_idx * cell_z
            return np.stack([xs, ys, zs], axis=1).astype(np.float32)
        rows = indices[:, 0].astype(np.float32)
        cols = indices[:, 1].astype(np.float32)
        if self._range_gt is not None and self._voxel_size is not None:
            x_min = float(self._range_gt[1])
            y_min = float(self._range_gt[3])
            voxel_x = float(self._voxel_size[0])
            voxel_y = float(self._voxel_size[1])
            xs = x_min + cols * voxel_x
            ys = y_min + rows * voxel_y
        else:
            rows_count, cols_count = obstacle_mask.shape
            world_x, world_y = self.cfg.world_size
            cell_x = world_x / float(cols_count)
            cell_y = world_y / float(rows_count)
            x_min = -world_x / 2.0 + 0.5 * cell_x
            y_min = -world_y / 2.0 + 0.5 * cell_y
            xs = x_min + cols * cell_x
            ys = y_min + rows * cell_y
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _min_obstacle_distance(self, position: np.ndarray) -> float | None:
        centers = self._obstacle_centers
        if centers is None or centers.size == 0:
            return None
        if centers.shape[1] == 3:
            delta = centers - position[:3]
        else:
            delta = centers - position[:2]
        dist_sq = np.sum(delta * delta, axis=1)
        return float(np.sqrt(np.min(dist_sq)))

    def _check_collision(self, position: np.ndarray, patch_idx: Tuple[int, int]) -> bool:
        if self._out_of_bounds(position):
            return True
        if self._obstacle_mask_3d is not None:
            voxel_idx = self._position_to_voxel(position)
            if self._obstacle_mask_3d[voxel_idx]:
                return True
        elif self._obstacle_mask is not None and self._obstacle_mask[patch_idx]:
            return True
        return False

    def _maybe_debug_reset(self) -> None:
        if self.cfg.debug_print_interval <= 0:
            return
        pos = np.asarray(self._position, dtype=np.float32)
        bounds = self._bounds
        print(
            f"[reset] scene={self._gleam_scene_index} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
            f"bounds={bounds} grid={self._grid_shape} world={self.cfg.world_size} "
            f"cam_height={self.cfg.maniskill_camera_height} cam_angles={self.cfg.maniskill_camera_angles}"
        )

    def _maybe_debug_step(
        self,
        action: np.ndarray,
        patch_idx: Tuple[int, int],
        crash: bool,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.cfg.debug_print_interval <= 0:
            return
        if self._step_count % self.cfg.debug_print_interval != 0:
            return
        pos = np.asarray(self._position, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        agent_state = (pos - self._state_center) / self._state_scale
        heading = getattr(self, "_heading", None)
        if heading is None and hasattr(self, "_yaw"):
            heading = float(getattr(self, "_yaw"))
        cam_pos = (float(pos[0]), float(pos[1]), float(self.cfg.maniskill_camera_height))
        obs_dist = self._min_obstacle_distance(pos)
        obs_dist_str = "None" if obs_dist is None else f"{obs_dist:.3f}"
        action_str = np.array2string(action, precision=3, floatmode="fixed")
        # print(
        #     f"[step {self._step_count}] action={action_str} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
        #     f"state=({agent_state[0]:.3f},{agent_state[1]:.3f},{agent_state[2]:.3f}) patch={patch_idx} "
        #     f"crash={crash} term={terminated} trunc={truncated} "
        #     f"coverage={self._coverage_ratio():.3f} heading={heading} "
        #     f"cam_pos=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f}) "
        #     f"obs_dist={obs_dist_str}"
        # )

    def _encode_depth_images(self, depth_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        frames = np.stack(depth_images, axis=0).astype(np.float32)
        obs_embedding = np.max(frames, axis=0)
        padded = np.zeros((self.cfg.max_depth_frames, *self.cfg.depth_image_shape), dtype=np.float32)
        padded[: frames.shape[0]] = frames
        return padded, obs_embedding

    def _encode_rgb_images(self, rgb_images: List[np.ndarray]) -> np.ndarray:
        target_h, target_w = self.cfg.depth_image_shape
        padded = np.zeros((self.cfg.max_depth_frames, target_h, target_w, 3), dtype=np.float32)
        if not rgb_images:
            return padded
        frames = []
        for img in rgb_images:
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[-1] < 3:
                continue
            arr = arr[..., :3]
            if arr.shape[:2] != (target_h, target_w):
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            max_val = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            if max_val > 1.0 + 1e-3:
                arr = arr / 255.0
            frames.append(arr)
        if not frames:
            return padded
        frames = np.stack(frames, axis=0)
        frames = frames[: self.cfg.max_depth_frames]
        padded[: frames.shape[0]] = frames
        return padded

    def _encode_camera_poses(self, camera_poses: List[np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.cfg.max_depth_frames, 7), dtype=np.float32)
        if not camera_poses:
            return padded
        frames = []
        for pose in camera_poses:
            arr = np.asarray(pose, dtype=np.float32).reshape(-1)
            if arr.size < 7:
                continue
            frames.append(arr[:7])
        if not frames:
            return padded
        frames = np.stack(frames, axis=0)
        frames = frames[: self.cfg.max_depth_frames]
        padded[: frames.shape[0]] = frames
        return padded

    def _default_init_z(self) -> float:
        z = float(self.cfg.agent_init_height)
        if self._range_gt is None:
            return z
        z_min = float(self._range_gt[5])
        z_max = float(self._range_gt[4])
        return float(np.clip(z, z_min, z_max))

    def _position_to_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        if self._range_gt is None or self._voxel_size is None:
            raise ValueError("GLEAM range_gt/voxel_size required for voxel collision checks.")
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        x_max, x_min, y_max, y_min, z_max, z_min = self._range_gt
        voxel_x, voxel_y, voxel_z = self._voxel_size[:3]
        x_min_voxel = x_min - 0.5 * voxel_x
        y_min_voxel = y_min - 0.5 * voxel_y
        z_min_voxel = z_min - 0.5 * voxel_z
        ix = int(np.floor((x - x_min_voxel) / voxel_x))
        iy = int(np.floor((y - y_min_voxel) / voxel_y))
        iz = int(np.floor((z - z_min_voxel) / voxel_z))
        if self._grid_shape_3d is not None:
            max_x, max_y, max_z = self._grid_shape_3d
        else:
            max_x = max_y = max_z = int(self._grid_shape[0])
        ix = int(np.clip(ix, 0, max_x - 1))
        iy = int(np.clip(iy, 0, max_y - 1))
        iz = int(np.clip(iz, 0, max_z - 1))
        return ix, iy, iz

    def _dilate_mask_3d(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return mask
        if torch is None or torch_f is None:  # pragma: no cover
            raise ImportError("torch is required for 3D obstacle dilation.")
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
        pad = int(kernel_size) // 2
        dilated = torch_f.max_pool3d(mask_t, kernel_size, stride=1, padding=pad)
        return dilated[0, 0].cpu().numpy().astype(bool)
