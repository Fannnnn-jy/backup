#!/usr/bin/env python3
"""
Convert planner-world XYZ and view angles into an ActiveGS init_pose matrix.

Convention:
- Planner/world frame is Z-up.
- init_pose is an OpenCV camera-to-world matrix.
- Columns of the 3x3 rotation are [camera_right, camera_down, camera_forward].
- yaw_deg is the horizontal heading in the XY plane measured from +X toward +Y.
- pitch_deg is elevation from the XY plane toward +Z.

Examples:
  python pose_from_xyz_angles.py --x 0.0 --y 0.5 --z 0.0 --yaw-deg 0
  python pose_from_xyz_angles.py --x 0.0 --y 0.5 --z 0.0 --yaw-deg 90
  python pose_from_xyz_angles.py --x 1.0 --y 2.0 --z 0.6 --yaw-deg 30 --pitch-deg -10
"""

import argparse
import json
import math
import numpy as np


def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("zero-length vector cannot be normalized")
    return v / norm


def pose_from_xyz_yaw_pitch(x, y, z, yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # OpenCV camera forward axis in planner/world Z-up coordinates.
    forward = np.array(
        [
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float32,
    )
    forward = _normalize(forward)

    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    right = np.cross(world_down, forward)
    if np.linalg.norm(right) < 1e-8:
        # Degenerate case: looking straight up/down. Pick a stable right axis.
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = _normalize(right)
    down = _normalize(np.cross(forward, right))

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = np.array([x, z, -y], dtype=np.float32)
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="also print the pose as JSON for downstream scripting",
    )
    args = parser.parse_args()

    pose = pose_from_xyz_yaw_pitch(
        x=args.x,
        y=args.y,
        z=args.z,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
    )
    pose_list = pose.tolist()

    print("planner/world convention: Z-up, OpenCV c2w")
    print(
        f"position=({args.x:.6f}, {args.y:.6f}, {args.z:.6f}) "
        f"yaw_deg={args.yaw_deg:.6f} pitch_deg={args.pitch_deg:.6f}"
    )
    print("init_pose:")
    print(pose_list)
    print("hydra override:")
    print(f"'planner.init_pose={json.dumps(pose_list, separators=(',', ':'))}'")

    if args.pretty_json:
        print("json:")
        print(json.dumps({"init_pose": pose_list}, indent=2))


if __name__ == "__main__":
    main()
