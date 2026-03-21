from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


ForwardAxis = Literal["x+", "x-", "y+", "y-", "z+", "z-"]


@dataclass
class VoxelMasks:
    occupied: np.ndarray
    free: np.ndarray
    unknown: np.ndarray


@dataclass
class VoxelCoverageUpdate:
    occupied: np.ndarray
    free: np.ndarray
    seen: np.ndarray
    new_seen: np.ndarray
    num_new_voxels: int
    num_total_seen_voxels: int


def intrinsics_from_fov(
    width: int, height: int, fov_rad: float, *, fov_axis: Literal["vertical", "horizontal"] = "vertical"
) -> Tuple[float, float, float, float]:
    if fov_axis == "horizontal":
        fx = 0.5 * width / float(np.tan(0.5 * fov_rad))
        fy = fx * (height / float(width))
    else:
        fy = 0.5 * height / float(np.tan(0.5 * fov_rad))
        fx = fy * (width / float(height))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def carve_depth_to_voxels(
    depth: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    range_gt: np.ndarray,
    voxel_size: np.ndarray,
    grid_size: int | Tuple[int, int, int],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_axis: ForwardAxis | None = None,
    depth_is_z: bool = True,
    depth_scale: float = 1.0,
    depth_max: float | None = None,
    depth_max_is_no_hit: bool = True,
    depth_max_epsilon: float = 1e-3,
    step_size: float | None = None,
    pose_is_camera_to_world: bool = True,
    ground_z: float | None = None,
    ground_epsilon: float = 1e-3,
) -> VoxelMasks:
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape[:2]
    depth = depth * float(depth_scale)
    grid_shape = _normalize_grid_shape(grid_size)

    max_val = float(depth_max) if depth_max is not None else None
    if max_val is not None:
        depth = np.minimum(depth, max_val)

    valid = np.isfinite(depth) & (depth > 0.0)
    if depth_max_is_no_hit and max_val is not None:
        hit_mask = valid & (depth < (max_val - float(depth_max_epsilon)))
    else:
        hit_mask = valid
    if depth_max_is_no_hit and max_val is not None:
        ray_mask = hit_mask
    else:
        ray_mask = valid
    if not np.any(valid):
        occ = np.zeros(grid_shape, dtype=bool)
        free = np.zeros_like(occ)
        unknown = np.ones_like(occ)
        return VoxelMasks(occupied=occ, free=free, unknown=unknown)

    if forward_axis is None:
        # Project convention used by active_mapping: identity quaternion means camera/world align,
        # image-right maps to -Y and image-down maps to -Z when looking along +X.
        right, down, forward = _camera_basis("x+")
    else:
        right, down, forward = _camera_basis(forward_axis)
    right = right.astype(np.float32)
    down = down.astype(np.float32)
    forward = forward.astype(np.float32)

    u = np.arange(width, dtype=np.float32) + 0.5
    v = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(u, v)
    x = (xx - float(cx)) / float(fx)
    y = (yy - float(cy)) / float(fy)

    dirs = x[..., None] * right + y[..., None] * down + forward
    if not depth_is_z:
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        norms = np.where(norms <= 1e-6, 1.0, norms)
        dirs = dirs / norms

    cam_pos = np.asarray(camera_position, dtype=np.float32).reshape(1, 1, 3)
    rot = _quat_wxyz_to_matrix(np.asarray(camera_quat_wxyz, dtype=np.float32))
    if not pose_is_camera_to_world:
        rot = rot.T

    depth_expand = depth[..., None]
    points_cam = dirs * depth_expand
    points_world = (points_cam.reshape(-1, 3) @ rot.T) + cam_pos.reshape(1, 3)
    points_world = points_world.reshape(height, width, 3)

    occ = np.zeros(grid_shape, dtype=bool)
    free = np.zeros_like(occ)

    occ_hit_mask = hit_mask
    if ground_z is not None:
        ground_mask = points_world[..., 2] <= (float(ground_z) + float(ground_epsilon))
        occ_hit_mask = hit_mask & ~ground_mask
    occ_indices = _world_to_voxel(points_world[occ_hit_mask], range_gt, voxel_size, grid_size)
    if occ_indices.size > 0:
        occ[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2]] = True

    step = float(step_size) if step_size is not None else float(np.min(voxel_size))
    step = max(step, 1e-4)
    depths_flat = depth[ray_mask]
    dirs_flat = dirs[ray_mask]
    if depth_is_z:
        ray_dirs = dirs_flat
    else:
        ray_dirs = dirs_flat
    steps_per_ray = np.floor((depths_flat - 1e-4) / step).astype(np.int32)
    max_steps = int(steps_per_ray.max(initial=0))
    if max_steps > 0:
        cam_pos_flat = cam_pos.reshape(1, 3)
        rot_t = rot.T
        for step_idx in range(1, max_steps + 1):
            active = steps_per_ray >= step_idx
            if not np.any(active):
                break
            t = float(step_idx) * step
            pts_cam = ray_dirs[active] * t
            pts_world = (pts_cam @ rot_t) + cam_pos_flat
            indices = _world_to_voxel(pts_world, range_gt, voxel_size, grid_size)
            free[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    free[occ] = False
    unknown = ~(occ | free)
    return VoxelMasks(occupied=occ, free=free, unknown=unknown)


def depth_to_world_points(
    depth: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_axis: ForwardAxis | None = None,
    depth_is_z: bool = True,
    depth_scale: float = 1.0,
    depth_max: float | None = None,
    depth_max_is_no_hit: bool = True,
    depth_max_epsilon: float = 1e-3,
    pose_is_camera_to_world: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    del depth_max, depth_max_is_no_hit, depth_max_epsilon

    depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    height, width = depth.shape[:2]
    if forward_axis is None:
        right, down, forward = _camera_basis("x+")
    else:
        right, down, forward = _camera_basis(forward_axis)
    right = right.astype(np.float32)
    down = down.astype(np.float32)
    forward = forward.astype(np.float32)

    u = np.arange(width, dtype=np.float32) + 0.5
    v = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(u, v)
    x = (xx - float(cx)) / float(fx)
    y = (yy - float(cy)) / float(fy)
    dirs = x[..., None] * right + y[..., None] * down + forward
    if not depth_is_z:
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        norms = np.where(norms <= 1e-6, 1.0, norms)
        dirs = dirs / norms

    rot = _quat_wxyz_to_matrix(np.asarray(camera_quat_wxyz, dtype=np.float32))
    if not pose_is_camera_to_world:
        rot = rot.T
    cam_pos = np.asarray(camera_position, dtype=np.float32).reshape(1, 3)
    points_cam = dirs[valid] * depth[valid, None]
    points_world = (points_cam @ rot.T) + cam_pos
    return points_world.astype(np.float32), valid


def accumulate_voxel_observation(
    previous_occupied: np.ndarray | None,
    previous_free: np.ndarray | None,
    current_masks: VoxelMasks,
    *,
    valid_mask: np.ndarray | None = None,
) -> VoxelCoverageUpdate:
    current_occ = np.asarray(current_masks.occupied, dtype=bool)
    current_free = np.asarray(current_masks.free, dtype=bool)
    if current_occ.shape != current_free.shape:
        raise ValueError("occupied/free voxel masks must share the same shape.")

    prev_occ = _prepare_accumulator_mask(previous_occupied, current_occ.shape)
    prev_free = _prepare_accumulator_mask(previous_free, current_occ.shape)

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != current_occ.shape:
            raise ValueError("valid_mask shape must match current voxel mask shape.")
        current_occ = current_occ & valid
        current_free = current_free & valid
        prev_occ = prev_occ & valid
        prev_free = prev_free & valid

    prev_seen = prev_occ | prev_free
    current_seen = current_occ | current_free
    new_seen = current_seen & ~prev_seen

    next_occ = prev_occ | current_occ
    next_free = prev_free | current_free
    next_free[next_occ] = False
    next_seen = next_occ | next_free
    return VoxelCoverageUpdate(
        occupied=next_occ,
        free=next_free,
        seen=next_seen,
        new_seen=new_seen,
        num_new_voxels=int(new_seen.sum()),
        num_total_seen_voxels=int(next_seen.sum()),
    )


def _camera_basis(forward_axis: ForwardAxis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if forward_axis == "z+":
        return np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if forward_axis == "z-":
        return np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    if forward_axis == "x+":
        return np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    if forward_axis == "x-":
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])
    if forward_axis == "y+":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    if forward_axis == "y-":
        return np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])
    raise ValueError(f"Unsupported forward axis: {forward_axis}")


def _prepare_accumulator_mask(mask: np.ndarray | None, shape: Tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"Accumulator mask shape mismatch: expected {shape}, got {arr.shape}.")
    return arr.copy()


def _normalize_grid_shape(grid_size: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(grid_size, tuple):
        if len(grid_size) != 3:
            raise ValueError(f"grid_size tuple must have 3 values, got {grid_size}.")
        return (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    size = int(grid_size)
    return (size, size, size)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _world_to_voxel(
    points: np.ndarray, range_gt: np.ndarray, voxel_size: np.ndarray, grid_size: int | Tuple[int, int, int]
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    grid_shape = _normalize_grid_shape(grid_size)
    x_min = float(range_gt[1])
    y_min = float(range_gt[3])
    z_min = float(range_gt[5])
    x_max = float(range_gt[0])
    y_max = float(range_gt[2])
    z_max = float(range_gt[4])
    in_bounds = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
        & (pts[:, 2] >= z_min)
        & (pts[:, 2] <= z_max)
    )
    pts = pts[in_bounds]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    vx, vy, vz = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    x_min_voxel = x_min - 0.5 * vx
    y_min_voxel = y_min - 0.5 * vy
    z_min_voxel = z_min - 0.5 * vz
    ix = np.floor((pts[:, 0] - x_min_voxel) / vx)
    iy = np.floor((pts[:, 1] - y_min_voxel) / vy)
    iz = np.floor((pts[:, 2] - z_min_voxel) / vz)
    # print(iz)
    # assert False
    ix = np.clip(ix, 0, grid_shape[0] - 1)
    iy = np.clip(iy, 0, grid_shape[1] - 1)
    iz = np.clip(iz, 0, grid_shape[2] - 1)
    return np.stack([ix, iy, iz], axis=1).astype(np.int32)
