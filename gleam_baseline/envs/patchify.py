from __future__ import annotations

from typing import Tuple

import numpy as np


def position_to_patch(
    position: Tuple[float, float],
    world_size: Tuple[float, float],
    grid_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map a continuous (x, y) position to a discrete (row, col) patch index."""
    raise NotImplementedError()
    x, y = position
    world_x, world_y = world_size
    rows, cols = grid_size

    u = (x + world_x / 2.0) / world_x
    v = (y + world_y / 2.0) / world_y

    col = int(np.clip(u * cols, 0, cols - 1))
    row = int(np.clip(v * rows, 0, rows - 1))
    return row, col

def position_to_patch_range(
    position: Tuple[float, float],
    range_gt: Tuple[float, float, float, float, float, float],
    voxel_size: Tuple[float, float, float],
    grid_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map a position to a patch index using GLEAM-style range and voxel size."""
    """
    Fan's note: position 用标准世界坐标 世界中心为原点，右侧为x正方向，上为y正方向，右手系 z朝外
    返回 row col 为 numpy 索引格式，openCV 与 numpy 一致，左上角为原点，右为col正方向，下为row正方向
    """
    x, y = position
    x_max, x_min, y_max, y_min, _, _ = range_gt
    voxel_x, voxel_y = voxel_size[:2]

    x_min_voxel = x_min - 0.5 * voxel_x
    y_max_voxel = y_max + 0.5 * voxel_y

    col = int(np.floor((x - x_min_voxel) / voxel_x))
    row = int(np.floor((y_max_voxel - y) / voxel_y))
    rows, cols = grid_size
    col = int(np.clip(col, 0, cols - 1))
    row = int(np.clip(row, 0, rows - 1))
    return row, col

def mark_patch_visited(visited: np.ndarray, patch_idx: Tuple[int, int]) -> bool:
    """Mark a patch as visited and return True if it was newly visited."""
    row, col = patch_idx
    if not visited[row, col]:
        visited[row, col] = True
        return True
    return False

def coverage_ratio(visited: np.ndarray) -> float:
    """Compute coverage ratio for the visited grid."""
    total = visited.size
    return float(visited.sum()) / float(total) if total else 0.0
