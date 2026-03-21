from __future__ import annotations

import argparse
import hashlib
import os
import math
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh


STANDARD_GLB_BOUNDS_Y_UP = {
    "x": (-1.0, 1.0),
    "y": (0.0, 1.0),
    "z": (-1.0, 1.0),
}
STANDARD_GLB_INTERNAL_BOUNDS_Z_UP = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 1.0),
}
STANDARD_GLB_VOXEL_GRID = (128, 128, 64)
STANDARD_GLB_VOXEL_PITCH = 2.0 / 128.0
STANDARD_GLB_AGENT_HEIGHT = 0.5
DEFAULT_DEPTH_BUFFER = "DepthLinear"


def _parse_float_list(text: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected comma-separated floats, e.g. 0,90,180")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Values must be floats") from exc


def _parse_vec3(text: str) -> Tuple[float, ...]:
    vals = _parse_float_list(text)
    if len(vals) not in (2, 3):
        raise argparse.ArgumentTypeError("Expected 2 or 3 floats: x,y[,z]")
    return tuple(float(v) for v in vals)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Active mapping viewer for normalized Y-up GLB scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("glb_path", help="Path to a GLB file (used to resolve scene index).")
    parser.add_argument("--save-path", default="", help="Optional path to save the initial rendered frame.")
    parser.add_argument("--y-up", action="store_true", default=True, help="Rotate scene from Y-up to Z-up.")
    parser.add_argument("--no-y-up", dest="y_up", action="store_false", help="Disable Y-up rotation.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale for the scene.")
    parser.add_argument("--width", type=int, default=256, help="Camera width.")
    parser.add_argument("--height", type=int, default=256, help="Camera height.")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2 wait time per frame.")
    parser.add_argument("--live", action="store_true", help="Render continuously in a window.")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Disable live rendering.")
    parser.add_argument("--move-step", type=float, default=0.1, help="Position step size per key press.")
    parser.add_argument("--rot-step-deg", type=float, default=5.0, help="Rotation step size in degrees.")
    parser.add_argument("--topdown", action="store_true", default=True, help="Render an additional top-down camera view.")
    parser.add_argument("--no-topdown", dest="topdown", action="store_false", help="Disable the top-down view.")
    parser.add_argument(
        "--init-pos",
        type=_parse_vec3,
        default=None,
        help="Initial position x,y[,z]. Leave unset to use the normalized-scene default height and env resampling.",
    )
    parser.add_argument("--glb-dir", default="", help="Directory containing GLB scenes.")
    parser.add_argument("--glb-glob", default="*.glb", help="GLB filename glob.")
    parser.add_argument("--glb-scene-index", type=int, default=-1, help="Scene index override.")
    parser.add_argument(
        "--am-action-mode",
        default="xyz_quat",
        help="Active mapping action mode.",
    )
    parser.add_argument(
        "--am-camera-angles",
        default="0",
        help="Comma-separated camera yaw angles in degrees (active_mapping mode).",
    )
    parser.add_argument(
        "--am-single-camera",
        action="store_true",
        default=True,
        help="Use first camera only (active_mapping mode).",
    )
    parser.add_argument(
        "--am-multi-camera",
        dest="am_single_camera",
        action="store_false",
        help="Use all configured cameras instead of only the first camera.",
    )
    parser.add_argument("--am-topdown-scale", type=int, default=4, help="Topdown scale (active_mapping mode).")
    parser.add_argument(
        "--am-depth-max", type=float, default=-1.0, help="Depth max for rendering (active_mapping mode)."
    )
    parser.add_argument(
        "--am-depth-buffer",
        default=DEFAULT_DEPTH_BUFFER,
        help="Depth buffer used for privileged depth observation / voxel carving.",
    )
    parser.add_argument(
        "--am-image-mode",
        choices=("rgb", "depth"),
        default="rgb",
        help="Render RGB or depth (active_mapping mode).",
    )
    parser.add_argument(
        "--am-camera-fov", type=float, default=-1.0, help="Camera FOV in degrees (active_mapping mode)."
    )
    parser.add_argument(
        "--am-camera-height", type=float, default=-1.0, help="Camera height in meters (active_mapping mode)."
    )
    parser.add_argument("--rerun", action="store_true", help="Log trajectory to Rerun (requires rerun-sdk).")
    parser.add_argument("--rerun-spawn", action="store_true", help="Spawn a local Rerun viewer.")
    parser.add_argument("--rerun-log-every", type=int, default=1, help="Log every N trajectory points to Rerun.")
    parser.add_argument("--rerun-mesh", action="store_true", help="Log the GLB mesh to Rerun.")
    parser.add_argument(
        "--rerun-mesh-mode",
        choices=("asset", "mesh", "both"),
        default="asset",
        help="Mesh logging mode for Rerun: asset=GLB with textures, mesh=Mesh3D, both=log both.",
    )
    parser.add_argument(
        "--rerun-asset-rotate",
        choices=("auto", "none", "x90", "x-90", "y_up"),
        default="auto",
        help=(
            "Rotation for textured asset: auto=use --y-up, none=identity, "
            "x90/+90 or x-90/-90 about X, y_up=declare asset as Y-up coords."
        ),
    )
    parser.add_argument(
        "--rerun-asset-bake",
        action="store_true",
        help="Bake transform into a temp GLB for textured assets (helps alignment).",
    )
    parser.add_argument(
        "--rerun-mesh-alpha",
        type=float,
        default=0.2,
        help="Alpha for the Rerun mesh (0..1).",
    )
    parser.add_argument(
        "--rerun-capture", action="store_true", help="Capture the Rerun viewer window into the right panel."
    )
    parser.add_argument("--rerun-window-title", default="Rerun", help="Window title to capture for Rerun viewer.")
    parser.add_argument("--rerun-capture-retry", type=int, default=30, help="Retry window lookup every N frames.")
    parser.set_defaults(live=True)
    return parser.parse_args()


def _build_standard_active_mapping_cfg(args: argparse.Namespace, glb_dir: Path, glb_glob: str, scene_index: int):
    from src.envs.active_mapping_env import ActiveMappingConfig

    cfg = ActiveMappingConfig()
    cfg.glb_data_dir = str(glb_dir)
    cfg.glb_scene_glob = glb_glob
    cfg.glb_scene_index = int(scene_index)
    cfg.glb_y_up = bool(args.y_up)
    cfg.glb_scale = float(args.scale)

    # Normalized GLB preset:
    # input coords are assumed Y-up with x/z in [-1, 1], y in [0, 1].
    # After the default Y-up -> Z-up rotation used by the env, the voxel grid becomes
    # x/y in [-1, 1], z in [0, 1], which matches a 128 x 128 x 64 cubic grid.
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


def _print_standard_glb_assumptions() -> None:
    print(
        "[active_mapping] normalized GLB preset: "
        f"input(Y-up) x={STANDARD_GLB_BOUNDS_Y_UP['x']} "
        f"y={STANDARD_GLB_BOUNDS_Y_UP['y']} "
        f"z={STANDARD_GLB_BOUNDS_Y_UP['z']} "
        f"-> internal(Z-up) x={STANDARD_GLB_INTERNAL_BOUNDS_Z_UP['x']} "
        f"y={STANDARD_GLB_INTERNAL_BOUNDS_Z_UP['y']} "
        f"z={STANDARD_GLB_INTERNAL_BOUNDS_Z_UP['z']} "
        f"voxel_grid={STANDARD_GLB_VOXEL_GRID} "
        f"voxel_pitch={STANDARD_GLB_VOXEL_PITCH:.6f}"
    )


def _write_xyz_ply(points: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")


def _prepare_voxel_mask(mask: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        return np.zeros(shape, dtype=bool)
    return arr


def _summarize_carve_stats(
    env,
    occupied: np.ndarray,
    previous_occupied: Optional[np.ndarray],
    previous_free: Optional[np.ndarray],
) -> Tuple[int, int, float, float]:
    current_occ = np.asarray(occupied, dtype=bool)
    current_occ_count = int(current_occ.sum())

    obstacle_mask_3d = getattr(env, "_obstacle_mask_3d", None)
    if obstacle_mask_3d is None:
        map_voxel_total = int(current_occ.size)
    else:
        map_voxel_total = int(np.asarray(obstacle_mask_3d, dtype=bool).sum())

    frame_ratio = float(current_occ_count) / float(map_voxel_total) if map_voxel_total > 0 else 0.0

    prev_occ = _prepare_voxel_mask(previous_occupied, current_occ.shape)
    prev_free = _prepare_voxel_mask(previous_free, current_occ.shape)
    prev_seen = prev_occ | prev_free
    overlap_count = int(np.logical_and(current_occ, prev_seen).sum())
    overlap_rate = float(overlap_count) / float(current_occ_count) if current_occ_count > 0 else 0.0
    return current_occ_count, map_voxel_total, frame_ratio, overlap_rate


def _load_scene_mesh(glb_path: str, y_up: bool, scale: float) -> Optional["trimesh.Trimesh"]:
    scene = trimesh.load(glb_path, force="scene", process=False)
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

    if y_up:
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
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(scale))
    return mesh


def _compute_scene_bounds(glb_path: str, y_up: bool, scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mesh = _load_scene_mesh(glb_path, y_up, scale)
    if mesh is None:
        return None
    verts = mesh.vertices
    if verts.size == 0:
        return None
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    return vmin, vmax


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


def _render_trajectory_3d(
    positions: List[np.ndarray],
    width: int,
    height: int,
    azim_deg: float = 45.0,
    elev_deg: float = 30.0,
) -> Optional[np.ndarray]:
    if width <= 0 or height <= 0:
        return None
    if not positions:
        return np.zeros((height, width, 3), dtype=np.uint8)
    import cv2

    pts = np.asarray(positions, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((height, width, 3), dtype=np.uint8)
    pts = pts[:, :3]
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return np.zeros((height, width, 3), dtype=np.uint8)
    pts = pts[finite]

    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    pts_centered = pts - center

    az = math.radians(float(azim_deg))
    el = math.radians(float(elev_deg))
    Rz = np.array(
        [
            [math.cos(az), -math.sin(az), 0.0],
            [math.sin(az), math.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(el), -math.sin(el)],
            [0.0, math.sin(el), math.cos(el)],
        ],
        dtype=np.float32,
    )
    rot = Rx @ Rz
    proj = (rot @ pts_centered.T).T
    xy = proj[:, :2]

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1e-6)
    scale = 0.8 * min(float(width), float(height)) / span
    xy_scaled = (xy - 0.5 * (min_xy + max_xy)) * scale

    xs = (xy_scaled[:, 0] + width * 0.5).astype(np.int32)
    ys = (height * 0.5 - xy_scaled[:, 1]).astype(np.int32)
    pts2d = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw axes.
    axis_len = 0.4 * span
    axes = np.array(
        [
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float32,
    )
    axes_proj = (rot @ axes.T).T[:, :2]
    axes_scaled = axes_proj * scale
    origin = np.array([width * 0.5, height * 0.5], dtype=np.float32)
    axes_pts = axes_scaled + origin
    axes_pts = axes_pts.astype(np.int32)
    cv2.line(canvas, tuple(origin.astype(np.int32)), tuple(axes_pts[0]), (0, 0, 255), 2)
    cv2.line(canvas, tuple(origin.astype(np.int32)), tuple(axes_pts[1]), (0, 255, 0), 2)
    cv2.line(canvas, tuple(origin.astype(np.int32)), tuple(axes_pts[2]), (255, 0, 0), 2)

    if pts2d.shape[0] >= 2:
        cv2.polylines(canvas, [pts2d], isClosed=False, color=(0, 255, 255), thickness=2)
    else:
        cv2.circle(canvas, tuple(pts2d[0, 0]), 2, (0, 255, 255), -1)
    cv2.circle(canvas, tuple(pts2d[-1, 0]), 3, (0, 0, 255), -1)
    return canvas


def _stack_bgr_row(frames: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    import cv2

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


def _parse_xwininfo_geometry(text: str) -> Optional[Tuple[int, int, int, int]]:
    x = y = w = h = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Absolute upper-left X:"):
            try:
                x = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Absolute upper-left Y:"):
            try:
                y = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Width:"):
            try:
                w = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Height:"):
            try:
                h = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _find_window_geometry(title: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        out = subprocess.check_output(["xwininfo", "-name", title], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return _parse_xwininfo_geometry(out)


def _capture_window_bgr(geom: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    try:
        from PIL import ImageGrab
    except Exception:
        return None
    x, y, w, h = geom
    try:
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    except Exception:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    return arr[..., :3][..., ::-1].copy()


def _init_rerun(enabled: bool, spawn: bool):
    if not enabled:
        return None
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        print(f"[rerun] not available: {exc}")
        return None
    if not hasattr(rr, "log") or not hasattr(rr, "LineStrips3D"):
        print("[rerun] rerun-sdk not available. Try: pip install rerun-sdk (and uninstall the 'rerun' package).")
        return None
    rr.init("glb_active_trajectory", spawn=spawn)
    return rr


def _log_glb_mesh_rerun(
    rr,
    glb_path: str,
    y_up: bool,
    scale: float,
    alpha: float,
    mode: str,
    root: str,
    asset_rotate: str,
    asset_bake: bool,
) -> None:
    if rr is None:
        return
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if mode in ("asset", "both"):
        try:
            color = np.array([200, 200, 200, int(round(alpha * 255.0))], dtype=np.uint8)
            asset_path = f"{root}/mesh" if mode == "asset" else f"{root}/mesh_textured"

            rot = np.eye(3, dtype=np.float32)
            if asset_rotate == "x90":
                theta = math.radians(90.0)
                rot = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, math.cos(theta), -math.sin(theta)],
                        [0.0, math.sin(theta), math.cos(theta)],
                    ],
                    dtype=np.float32,
                )
            elif asset_rotate == "x-90":
                theta = math.radians(-90.0)
                rot = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, math.cos(theta), -math.sin(theta)],
                        [0.0, math.sin(theta), math.cos(theta)],
                    ],
                    dtype=np.float32,
                )
            elif asset_rotate == "auto":
                if y_up:
                    theta = math.radians(90.0)
                    rot = np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, math.cos(theta), -math.sin(theta)],
                            [0.0, math.sin(theta), math.cos(theta)],
                        ],
                        dtype=np.float32,
                    )
            sc = float(scale)
            if abs(sc - 1.0) > 1e-6:
                rot = rot * sc

            asset_glb_path = glb_path
            if asset_bake and (not np.allclose(rot, np.eye(3, dtype=np.float32)) or asset_rotate == "y_up"):
                try:
                    digest = hashlib.md5(
                        f"{glb_path}|{os.path.getmtime(glb_path)}|{rot.tobytes()}".encode()
                    ).hexdigest()
                    bake_path = os.path.join("/tmp", f"rerun_asset_{digest}.glb")
                    if not os.path.exists(bake_path):
                        scene = trimesh.load(glb_path, force="scene", process=False)
                        if asset_rotate == "y_up":
                            theta = math.radians(90.0)
                            rot_yup = np.array(
                                [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, math.cos(theta), -math.sin(theta), 0.0],
                                    [0.0, math.sin(theta), math.cos(theta), 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            )
                            scene.apply_transform(rot_yup)
                        if not np.allclose(rot, np.eye(3, dtype=np.float32)):
                            rot4 = np.eye(4, dtype=np.float32)
                            rot4[:3, :3] = rot
                            scene.apply_transform(rot4)
                        scene.export(bake_path)
                    asset_glb_path = bake_path
                except Exception as exc:
                    print(f"[rerun] asset bake failed: {exc}")

            rr.log(asset_path, rr.Asset3D(path=asset_glb_path, albedo_factor=color), static=True)
            if asset_rotate == "y_up" and not asset_bake:
                rr.log(asset_path, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            if not asset_bake and not np.allclose(rot, np.eye(3, dtype=np.float32)):
                rr.log(asset_path, rr.Transform3D(mat3x3=rot), static=True)
        except Exception as exc:
            print(f"[rerun] asset logging failed: {exc}")

    if mode not in ("mesh", "both"):
        return

    mesh = _load_scene_mesh(glb_path, y_up, scale)
    if mesh is None:
        print("[rerun] mesh unavailable; skip mesh logging.")
        return
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = getattr(mesh, "faces", None)
    if verts.size == 0:
        print("[rerun] empty mesh; skip mesh logging.")
        return
    try:
        idx = np.asarray(faces, dtype=np.int32) if faces is not None else None
        color = np.array([200, 200, 200, int(round(alpha * 255.0))], dtype=np.uint8)
        colors = np.tile(color[None, :], (verts.shape[0], 1))
        mesh_path = f"{root}/mesh" if mode == "mesh" else f"{root}/mesh_alpha"
        if idx is not None and idx.size > 0:
            rr.log(
                mesh_path,
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=idx,
                    vertex_colors=colors,
                ),
            )
        else:
            rr.log(mesh_path, rr.Mesh3D(vertex_positions=verts, vertex_colors=colors))
    except Exception as exc:
        print(f"[rerun] mesh logging failed: {exc}")


def _pose_to_se3(pp, position: np.ndarray, quat_wxyz: np.ndarray):
    import torch

    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    vec = torch.tensor([*position.tolist(), *quat_xyzw.tolist()], dtype=torch.float32)
    return pp.SE3(vec)


def _log_trajectory_rerun(rr, positions: np.ndarray, log_every: int, root: str) -> None:
    if rr is None:
        return
    if log_every <= 0:
        log_every = 1
    if positions.shape[0] % log_every != 0:
        return
    rr.log(f"{root}/trajectory", rr.LineStrips3D([positions]))
    rr.log(f"{root}/trajectory/current", rr.Points3D(positions[-1:]))


def _print_scene_stats(glb_path: str, y_up: bool, scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    bounds = _compute_scene_bounds(glb_path, y_up, scale)
    if bounds is None:
        print("[scene] Unable to compute bounds (trimesh not available or empty scene).")
        return None
    vmin, vmax = bounds
    size = vmax - vmin
    center = 0.5 * (vmin + vmax)
    print("[scene] bounds min:", np.round(vmin, 4))
    print("[scene] bounds max:", np.round(vmax, 4))
    print("[scene] size (xyz):", np.round(size, 4))
    print("[scene] center:", np.round(center, 4))
    return bounds


def _find_glb_scene_index(glb_dir: Path, glb_glob: str, glb_path: Path) -> Optional[int]:
    if not glb_dir.exists():
        return None
    pattern = glb_glob or "*.glb"
    paths = sorted([p for p in glb_dir.glob(pattern) if p.suffix.lower() == ".glb"])
    if not paths:
        return None
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
    pattern = glb_glob or "*.glb"
    paths = sorted([p for p in glb_dir.glob(pattern) if p.suffix.lower() == ".glb"])
    if 0 <= scene_index < len(paths):
        return paths[scene_index]
    return fallback


def _run_active_mapping(args: argparse.Namespace) -> None:
    from src.envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv
    from src.envs.voxel_carving import accumulate_voxel_observation
    from src.utils.rendering import init_render_state, render_depth_and_topdown, reset_render_state

    glb_path = Path(args.glb_path)
    glb_dir = Path(args.glb_dir) if args.glb_dir else glb_path.parent
    glb_glob = args.glb_glob or "*.glb"
    scene_index = int(args.glb_scene_index)
    if scene_index < 0:
        idx = _find_glb_scene_index(glb_dir, glb_glob, glb_path)
        if idx is None:
            print("[active_mapping] glb path not found in dir; defaulting to index 0.")
            scene_index = 0
        else:
            scene_index = idx
    active_glb_path = _resolve_glb_scene_path(glb_dir, glb_glob, scene_index, glb_path)
    _print_standard_glb_assumptions()
    print("[active_mapping] glb path:", active_glb_path)
    _print_scene_stats(str(active_glb_path), args.y_up, args.scale)

    cfg = _build_standard_active_mapping_cfg(args, glb_dir, glb_glob, scene_index)
    env = ManiSkillActiveMappingEnv(cfg)

    rr = _init_rerun(args.rerun, args.rerun_spawn)
    rerun_root = "scene"
    if rr is not None:
        rr.log(rerun_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    pp = None
    if args.rerun and rr is not None:
        try:
            import pypose as pp  # type: ignore
        except Exception as exc:
            print(f"[rerun] pypose not available, falling back to raw positions: {exc}")
    if args.rerun_mesh:
        _log_glb_mesh_rerun(
            rr,
            str(active_glb_path),
            args.y_up,
            args.scale,
            args.rerun_mesh_alpha,
            args.rerun_mesh_mode,
            rerun_root,
            args.rerun_asset_rotate,
            args.rerun_asset_bake,
        )

    obs, _ = env.reset()
    render_state = init_render_state(cfg)
    accumulated_pointcloud = np.zeros((0, 3), dtype=np.float32)
    accumulated_ply_path = active_glb_path.parent / f"{active_glb_path.stem}.accumulated.ply"

    traj_positions: List[np.ndarray] = []
    traj_se3: List = []

    def _get_active_pose() -> Tuple[np.ndarray, np.ndarray]:
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

    def _record_pose(position: np.ndarray, quat: np.ndarray) -> None:
        print(f"[active_mapping] position: {np.round(position, 4)}")
        traj_positions.append(position.copy())
        if rr is None:
            return
        if pp is not None:
            traj_se3.append(_pose_to_se3(pp, position, quat))
            import torch

            data = torch.stack([s.tensor() for s in traj_se3], dim=0)
            positions = pp.SE3(data).translation().detach().cpu().numpy()
        else:
            positions = np.stack(traj_positions, axis=0)
        _log_trajectory_rerun(rr, positions, args.rerun_log_every, rerun_root)

    pos, quat = _get_active_pose()
    _record_pose(pos, quat)
    print(f"[active_mapping] initial position: {np.round(pos, 4)}")
    print(f"[active_mapping] depth buffer: {getattr(env, '_last_depth_buffer_name', cfg.maniskill_depth_buffer or 'auto')}")

    depth_max = float(args.am_depth_max) if args.am_depth_max > 0 else float(cfg.depth_data_max_value)
    topdown_scale = max(1, int(args.am_topdown_scale))
    show_topdown = bool(args.topdown)

    image_bgr, topdown_bgr, _ = render_depth_and_topdown(
        obs,
        env,
        state=render_state,
        depth_max=depth_max,
        topdown_scale=topdown_scale,
        view="both" if show_topdown else "depth",
        image_mode=args.am_image_mode,
        first_camera_only=args.am_single_camera,
    )
    traj3d_bgr = None
    if topdown_bgr is not None:
        traj3d_bgr = _render_trajectory_3d(traj_positions, topdown_bgr.shape[1], topdown_bgr.shape[0])
    elif image_bgr is not None:
        traj3d_bgr = _render_trajectory_3d(traj_positions, image_bgr.shape[1], image_bgr.shape[0])
    rerun_geom = None
    if args.rerun_capture:
        rerun_geom = _find_window_geometry(args.rerun_window_title)
        if rerun_geom is None:
            print(f"[rerun-capture] window '{args.rerun_window_title}' not found; falling back to local 3D.")
    rerun_bgr = _capture_window_bgr(rerun_geom) if rerun_geom is not None else None
    right_panel = rerun_bgr if rerun_bgr is not None else traj3d_bgr
    frame = _stack_bgr_row([image_bgr, topdown_bgr, right_panel])
    if args.save_path and frame is not None:
        import matplotlib.pyplot as plt

        plt.imsave(args.save_path, frame[..., ::-1])
        print(f"Saved RGB to: {args.save_path}")

    if not args.live:
        env.close()
        return

    try:
        import cv2
    except ImportError:
        print("cv2 not available; disable --live or install opencv-python.")
        env.close()
        return

    mode = getattr(cfg, "action_mode", "xyz")
    move_step = float(args.move_step)
    rot_step = math.radians(float(args.rot_step_deg))
    yaw = float(getattr(env, "_yaw", 0.0))
    action_scale = float(cfg.action_scale) if float(cfg.action_scale) != 0.0 else 1.0
    quat_wxyz = np.asarray(getattr(env, "_camera_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float32)

    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float32,
        )

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

    print("Controls: WASD = move (world X/Y), K/L = +Z/-Z (xyz only), 1/2 = yaw (xy_cos_sin), C = carve view, Q/Esc = quit")
    if mode == "xyz_quat":
        print("Controls (xyz_quat): 1/2 = yaw, 3/4 = pitch, 5/6 = roll")

    window_name = "active_mapping"
    frame_idx = 0
    while True:
        image_bgr, topdown_bgr, _ = render_depth_and_topdown(
            obs,
            env,
            state=render_state,
            depth_max=depth_max,
            topdown_scale=topdown_scale,
            view="both" if show_topdown else "depth",
            image_mode=args.am_image_mode,
            first_camera_only=args.am_single_camera,
        )
        traj3d_bgr = None
        if topdown_bgr is not None:
            traj3d_bgr = _render_trajectory_3d(traj_positions, topdown_bgr.shape[1], topdown_bgr.shape[0])
        elif image_bgr is not None:
            traj3d_bgr = _render_trajectory_3d(traj_positions, image_bgr.shape[1], image_bgr.shape[0])
        if args.rerun_capture:
            retry = max(1, int(args.rerun_capture_retry))
            if rerun_geom is None and frame_idx % retry == 0:
                rerun_geom = _find_window_geometry(args.rerun_window_title)
                if rerun_geom is not None:
                    print(f"[rerun-capture] found window '{args.rerun_window_title}' at {rerun_geom}.")
        rerun_bgr = _capture_window_bgr(rerun_geom) if rerun_geom is not None else None
        right_panel = rerun_bgr if rerun_bgr is not None else traj3d_bgr
        frame = _stack_bgr_row([image_bgr, topdown_bgr, right_panel])
        if frame is None:
            print("[active_mapping] no frame to render; exiting.")
            break
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(args.wait_ms) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("c"), ord("C")):
            try:
                masks = env.get_current_view_voxel_masks(target_name=cfg.maniskill_depth_buffer or None)
                points_world = env.get_current_view_pointcloud(target_name=cfg.maniskill_depth_buffer or None)
            except Exception as exc:
                print(f"[active_mapping] voxel carve failed: {exc}")
                continue
            if masks is None:
                print("[active_mapping] voxel carve skipped: no valid camera/depth data.")
                continue
            frame_occ_count, map_voxel_total, frame_ratio, overlap_rate = _summarize_carve_stats(
                env,
                masks.occupied,
                render_state.carved_occupied,
                render_state.carved_free,
            )
            coverage = accumulate_voxel_observation(
                render_state.carved_occupied,
                render_state.carved_free,
                masks,
            )
            render_state.carved_occupied = coverage.occupied
            render_state.carved_free = coverage.free
            render_state.last_new_voxel_count = coverage.num_new_voxels
            render_state.last_seen_voxel_count = coverage.num_total_seen_voxels
            show_topdown = True
            print(
                "[active_mapping] carve "
                f"buffer={getattr(env, '_last_depth_buffer_name', cfg.maniskill_depth_buffer or 'auto')} "
                f"frame_occ_voxels={frame_occ_count} "
                f"map_occ_voxels={map_voxel_total} "
                f"frame_occ_ratio={frame_ratio:.6f} "
                f"overlap_prev_seen={overlap_rate:.6f} "
                f"new_voxels={coverage.num_new_voxels} "
                f"total_seen={coverage.num_total_seen_voxels}"
            )
            if points_world.size == 0:
                print("[active_mapping] pointcloud export skipped: no valid depth hit points.")
            else:
                try:
                    if accumulated_pointcloud.size == 0:
                        accumulated_pointcloud = points_world.astype(np.float32, copy=True)
                    else:
                        accumulated_pointcloud = np.concatenate(
                            [accumulated_pointcloud, points_world.astype(np.float32, copy=False)],
                            axis=0,
                        )
                    _write_xyz_ply(accumulated_pointcloud, accumulated_ply_path)
                    print(
                        "[active_mapping] saved accumulated pointcloud "
                        f"current_points={int(points_world.shape[0])} "
                        f"total_points={int(accumulated_pointcloud.shape[0])} "
                        f"path={accumulated_ply_path}"
                    )
                except Exception as exc:
                    print(f"[active_mapping] pointcloud export failed: {exc}")
            continue

        delta = np.zeros(3, dtype=np.float32)
        moved = False
        quat_updated = False
        if key in (ord("w"), ord("W")):
            delta[1] += move_step
            moved = True
        elif key in (ord("s"), ord("S")):
            delta[1] -= move_step
            moved = True
        elif key in (ord("a"), ord("A")):
            delta[0] -= move_step
            moved = True
        elif key in (ord("d"), ord("D")):
            delta[0] += move_step
            moved = True
        elif key in (ord("k"), ord("K"), ord("l"), ord("L")) and mode != "xyz" and mode != "xyz_quat":
            """
            原则上只允许水平运动，但在非xyz模式下，K/L键被用来调整相机高度（Z轴位置），以便更好地观察环境和轨迹。
            """

            dz = move_step if key in (ord("k"), ord("K")) else -move_step
            prev_pos = np.asarray(getattr(env, "_position", np.zeros(3)), dtype=np.float32)
            new_pos = prev_pos.copy()
            new_pos[2] += dz
            bounds = getattr(env, "_bounds", None)
            if bounds is not None and len(bounds) == 6:
                z_min = float(bounds[4])
                z_max = float(bounds[5])
                new_pos[2] = float(np.clip(new_pos[2], z_min, z_max))
            try:
                env._position = new_pos
                if hasattr(env, "_update_camera_poses"):
                    env._update_camera_poses()
                if hasattr(env, "_build_observation"):
                    obs = env._build_observation()
                render_state.prev_position = prev_pos
                pos, quat = _get_active_pose()
                _record_pose(pos, quat)
            except Exception as exc:
                print(f"[active_mapping] manual Z update failed: {exc}")
            continue
        elif key in (ord("k"), ord("K")) and (mode == "xyz" or mode == "xyz_quat"):
            delta[2] += move_step
            moved = True
        elif key in (ord("l"), ord("L")) and (mode == "xyz" or mode == "xyz_quat"):
            delta[2] -= move_step
            moved = True
        elif key == ord("1"):
            if mode == "xyz_quat":
                quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 0.0, 1.0], rot_step), quat_wxyz)
                quat_updated = True
            else:
                yaw += rot_step
                moved = True
        elif key == ord("2"):
            if mode == "xyz_quat":
                quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 0.0, 1.0], -rot_step), quat_wxyz)
                quat_updated = True
            else:
                yaw -= rot_step
                moved = True
        elif key == ord("3") and mode == "xyz_quat":
            quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 1.0, 0.0], rot_step), quat_wxyz)
            quat_updated = True
        elif key == ord("4") and mode == "xyz_quat":
            quat_wxyz = _quat_mul(_quat_from_axis_angle([0.0, 1.0, 0.0], -rot_step), quat_wxyz)
            quat_updated = True
        elif key == ord("5") and mode == "xyz_quat":
            quat_wxyz = _quat_mul(_quat_from_axis_angle([1.0, 0.0, 0.0], rot_step), quat_wxyz)
            quat_updated = True
        elif key == ord("6") and mode == "xyz_quat":
            quat_wxyz = _quat_mul(_quat_from_axis_angle([1.0, 0.0, 0.0], -rot_step), quat_wxyz)
            quat_updated = True

        if not moved and not quat_updated:
            continue
        if quat_updated:
            quat_wxyz = _quat_normalize(quat_wxyz)

        if mode == "xy_cos_sin":
            action = np.array(
                [
                    delta[0] / action_scale,
                    delta[1] / action_scale,
                    math.cos(yaw),
                    math.sin(yaw),
                ],
                dtype=np.float32,
            )
        elif mode == "xyz_quat":
            action = np.array(
                [
                    delta[0] / action_scale,
                    delta[1] / action_scale,
                    delta[2] / action_scale,
                    quat_wxyz[0],
                    quat_wxyz[1],
                    quat_wxyz[2],
                    quat_wxyz[3],
                ],
                dtype=np.float32,
            )
        elif mode == "xy":
            action = np.array([delta[0] / action_scale, delta[1] / action_scale], dtype=np.float32)
        else:
            action = delta / action_scale

        prev_pos = np.asarray(getattr(env, "_position", np.zeros(3)), dtype=np.float32)
        try:
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as exc:
            print(f"[active_mapping] step failed: {exc}")
            continue
        render_state.prev_position = prev_pos

        pos, quat = _get_active_pose()
        _record_pose(pos, quat)
        if info.get("crash", False):
            print("[active_mapping] collision/out_of_bounds detected. Clearing trajectory.")
            traj_positions.clear()
            traj_se3.clear()

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"[active_mapping] episode {reason}; resetting.")
            obs, _ = env.reset()
            reset_render_state(render_state)
            traj_positions.clear()
            traj_se3.clear()
            yaw = float(getattr(env, "_yaw", 0.0))
            quat_wxyz = np.asarray(
                getattr(env, "_camera_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])),
                dtype=np.float32,
            )
            pos, quat = _get_active_pose()
            _record_pose(pos, quat)

        frame_idx += 1
        time.sleep(0.001)

    env.close()
    cv2.destroyAllWindows()


def main() -> None:
    args = _parse_args()
    _run_active_mapping(args)


if __name__ == "__main__":
    main()
