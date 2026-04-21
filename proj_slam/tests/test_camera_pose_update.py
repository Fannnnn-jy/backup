from __future__ import annotations

import math

import numpy as np

from src.utils.camera_pose_update import (
    apply_local_camera_delta_pose,
    normalize_quat_wxyz,
    quat_mul_wxyz,
    quat_wxyz_to_matrix,
)


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    s = float(math.sin(half))
    return normalize_quat_wxyz(
        np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)
    )


def test_local_translation_rotates_into_world_frame() -> None:
    current_quat = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), math.pi / 2.0)
    next_position, next_quat, debug_info = apply_local_camera_delta_pose(
        np.zeros(3, dtype=np.float32),
        current_quat,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(next_position, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        [debug_info["action_translation_world_x"], debug_info["action_translation_world_y"], debug_info["action_translation_world_z"]],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(quat_wxyz_to_matrix(next_quat), quat_wxyz_to_matrix(current_quat), atol=1e-6)


def test_local_rotation_uses_body_frame_right_multiplication() -> None:
    current_quat = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), math.pi / 2.0)
    delta_local = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), math.pi / 3.0)

    _, next_quat, _ = apply_local_camera_delta_pose(
        np.zeros(3, dtype=np.float32),
        current_quat,
        np.zeros(3, dtype=np.float32),
        delta_local,
    )

    expected_local = quat_mul_wxyz(current_quat, delta_local)
    wrong_world = quat_mul_wxyz(delta_local, current_quat)

    np.testing.assert_allclose(quat_wxyz_to_matrix(next_quat), quat_wxyz_to_matrix(expected_local), atol=1e-6)
    assert not np.allclose(quat_wxyz_to_matrix(next_quat), quat_wxyz_to_matrix(wrong_world), atol=1e-6)


def test_identity_pose_keeps_local_and_world_updates_equal() -> None:
    local_translation = np.array([0.25, -0.1, 0.4], dtype=np.float32)
    delta_local = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float32), math.pi / 5.0)

    next_position, next_quat, _ = apply_local_camera_delta_pose(
        np.zeros(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        local_translation,
        delta_local,
    )

    np.testing.assert_allclose(next_position, local_translation, atol=1e-6)
    np.testing.assert_allclose(quat_wxyz_to_matrix(next_quat), quat_wxyz_to_matrix(delta_local), atol=1e-6)


def test_quaternion_output_stays_normalized() -> None:
    _, next_quat, _ = apply_local_camera_delta_pose(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )

    assert math.isclose(float(np.linalg.norm(next_quat)), 1.0, rel_tol=1e-6, abs_tol=1e-6)
