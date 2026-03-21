from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _pose_position(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    if hasattr(pose, "p"):
        try:
            return np.asarray(pose.p, dtype=np.float32)
        except Exception:
            return None
    return None


def _pose_quat(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    if hasattr(pose, "q"):
        try:
            return np.asarray(pose.q, dtype=np.float32)
        except Exception:
            return None
    return None


def _camera_pose(cam: Any) -> Any | None:
    for getter in ("get_pose", "get_local_pose"):
        if hasattr(cam, getter):
            try:
                return getattr(cam, getter)()
            except Exception:
                continue
    return None


def _camera_picture_names(cam: Any) -> Iterable[str] | None:
    if not hasattr(cam, "get_picture_names"):
        return None
    try:
        return list(cam.get_picture_names())
    except Exception:
        return None


def dump_camera_buffer_stats(cam: Any, names: Iterable[str] | None = None, *, note: str = "") -> None:
    if cam is None:
        return
    if names is None:
        names = _camera_picture_names(cam)
    if not names:
        return
    for name in names:
        if not name:
            continue
        try:
            texture = cam.get_picture(name)
        except Exception:
            continue
        if texture is None:
            continue
        arr = np.asarray(texture)
        if arr.size == 0:
            continue
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            means = flat.mean(axis=0)
            mins = flat.min(axis=0)
            maxs = flat.max(axis=0)
            stats = ",".join(f"{means[idx]:.4f}/{mins[idx]:.4f}/{maxs[idx]:.4f}" for idx in range(flat.shape[-1]))
            stat_str = f"mean/min/max={stats}"
        else:
            stat_str = f"mean={float(arr.mean()):.4f},min={float(arr.min()):.4f},max={float(arr.max()):.4f}"
        print(f"[cam-buffer {note}] {name} shape={arr.shape} {stat_str}")


def print_camera_debug(env: Any, step_idx: int, *, note: str = "") -> None:
    pos = getattr(env, "_position", None)
    pos_xy = None if pos is None else np.asarray(pos, dtype=np.float32)
    cam_pos = getattr(env, "_last_camera_pos", None)
    cam_quats = getattr(env, "_last_camera_quats", None)
    depth_buffer = getattr(env, "_last_depth_buffer_name", None)
    cameras = getattr(env, "_cameras", []) or []
    first_cam = cameras[0] if cameras else None
    pose = _camera_pose(first_cam) if first_cam is not None else None
    pose_p = _pose_position(pose)
    pose_q = _pose_quat(pose)
    names = _camera_picture_names(first_cam) if first_cam is not None else None

    pos_str = "None"
    if pos_xy is not None:
        if pos_xy.shape[0] >= 3:
            pos_str = f"({pos_xy[0]:.3f},{pos_xy[1]:.3f},{pos_xy[2]:.3f})"
        else:
            pos_str = f"({pos_xy[0]:.3f},{pos_xy[1]:.3f})"
    cam_pos_str = "None"
    if cam_pos is not None:
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        cam_pos_str = f"({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f})"
    pose_p_str = "None"
    if pose_p is not None:
        pose_p_str = f"({pose_p[0]:.3f},{pose_p[1]:.3f},{pose_p[2]:.3f})"
    pose_q_str = "None"
    if pose_q is not None:
        pose_q_str = f"({pose_q[0]:.3f},{pose_q[1]:.3f},{pose_q[2]:.3f},{pose_q[3]:.3f})"
    quat_str = "None"
    if cam_quats:
        quat_str = "[" + ", ".join(np.array2string(np.asarray(q), precision=3) for q in cam_quats) + "]"
    names_str = "None" if names is None else ",".join(names)
    depth_buffer_str = "None" if depth_buffer is None else str(depth_buffer)
    all_cam_pose_strs = []
    for idx, cam in enumerate(cameras):
        cam_pose = _camera_pose(cam)
        cp = _pose_position(cam_pose)
        cq = _pose_quat(cam_pose)
        if cp is None or cq is None:
            continue
        all_cam_pose_strs.append(
            f"cam{idx}:p=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}),q=({cq[0]:.3f},{cq[1]:.3f},{cq[2]:.3f},{cq[3]:.3f})"
        )
    all_cam_pose_str = " | ".join(all_cam_pose_strs) if all_cam_pose_strs else "None"

    print(
        f"[cam-debug step={step_idx} {note}] drone_pos={pos_str} "
        f"cam_pos={cam_pos_str} cam_pose_p={pose_p_str} cam_pose_q={pose_q_str} "
        f"cam_quats={quat_str} depth_buffer={depth_buffer_str} buffers={names_str}"
    )
    print(f"[cam-debug step={step_idx} {note}] camera_poses={all_cam_pose_str}")
