from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
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


def quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # Hamilton product in (w, x, y, z) format.
    lw, lx, ly, lz = normalize_quat_wxyz(lhs)
    rw, rx, ry, rz = normalize_quat_wxyz(rhs)
    out = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )
    return normalize_quat_wxyz(out)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
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


def project_quat_wxyz_to_yaw_only(quat: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    q[1] = 0.0
    q[2] = 0.0
    wz_norm = float(np.linalg.norm(q[[0, 3]]))
    if wz_norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q[0] /= wz_norm
    q[3] /= wz_norm
    return normalize_quat_wxyz(q)


def build_local_pose_update_debug_defaults() -> dict[str, float | str]:
    return {
        "action_translation_local_x": 0.0,
        "action_translation_local_y": 0.0,
        "action_translation_local_z": 0.0,
        "action_translation_world_x": 0.0,
        "action_translation_world_y": 0.0,
        "action_translation_world_z": 0.0,
        "action_quat_delta_w": 1.0,
        "action_quat_delta_x": 0.0,
        "action_quat_delta_y": 0.0,
        "action_quat_delta_z": 0.0,
        "pose_update_frame": "camera_local",
    }


def apply_local_camera_delta_pose(
    position_world: np.ndarray,
    quat_world_from_camera_wxyz: np.ndarray,
    translation_local: np.ndarray,
    delta_quat_local_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Apply a local camera-frame pose increment to a camera-to-world pose.

    The active-mapping env stores pose as `(p_w, q_wc)`, where `p_w` is camera
    position in world coordinates and `q_wc` / `R_wc` rotates camera-frame
    vectors into the world frame. Under `xyz_delta_quat`, `xyz` means local
    camera-frame translation and `delta_quat` means a local/body-frame rotation.
    """

    position_world = np.asarray(position_world, dtype=np.float32).reshape(-1)
    if position_world.size < 3:
        padded_position = np.zeros(3, dtype=np.float32)
        padded_position[: position_world.size] = position_world
        position_world = padded_position
    else:
        position_world = position_world[:3].astype(np.float32, copy=False)

    current_quat = normalize_quat_wxyz(quat_world_from_camera_wxyz)
    translation_local = np.asarray(translation_local, dtype=np.float32).reshape(-1)
    if translation_local.size < 3:
        padded_translation = np.zeros(3, dtype=np.float32)
        padded_translation[: translation_local.size] = translation_local
        translation_local = padded_translation
    else:
        translation_local = translation_local[:3].astype(np.float32, copy=False)
    delta_quat_local = normalize_quat_wxyz(delta_quat_local_wxyz)

    rot_world_from_camera = quat_wxyz_to_matrix(current_quat)
    # Translation action is defined in camera/local coordinates and rotated into
    # world coordinates before pose update.
    translation_world = rot_world_from_camera @ translation_local
    next_position_world = position_world + translation_world

    # Multiplication order matters: q_current ⊗ q_delta applies the increment in
    # the current camera/body frame. Left-multiplication would instead apply a
    # world-frame increment around global axes.
    next_quat_world_from_camera = quat_mul_wxyz(current_quat, delta_quat_local)

    debug_info = build_local_pose_update_debug_defaults()
    debug_info.update(
        {
            "action_translation_local_x": float(translation_local[0]),
            "action_translation_local_y": float(translation_local[1]),
            "action_translation_local_z": float(translation_local[2]),
            "action_translation_world_x": float(translation_world[0]),
            "action_translation_world_y": float(translation_world[1]),
            "action_translation_world_z": float(translation_world[2]),
            "action_quat_delta_w": float(delta_quat_local[0]),
            "action_quat_delta_x": float(delta_quat_local[1]),
            "action_quat_delta_y": float(delta_quat_local[2]),
            "action_quat_delta_z": float(delta_quat_local[3]),
        }
    )
    return next_position_world.astype(np.float32), next_quat_world_from_camera.astype(np.float32), debug_info
