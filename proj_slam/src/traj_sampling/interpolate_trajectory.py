"""Interpolate a sampled trajectory to produce a denser pose sequence.

Reads a raw_poses JSON (as produced by ``sample_trajectory.py``), interpolates
between consecutive keyframes using **linear interpolation** for position and
**spherical linear interpolation (SLERP)** for quaternion orientation, then
writes the result as a new JSON in the same schema.

Usage::

    python -m src.traj_sampling.interpolate_trajectory \
        src/traj_sampling/my_trajectories/raw_poses/replicacad__apt_0_run_000.json \
        --num-interp 5 \
        --output-dir src/traj_sampling/my_trajectories/interpolated_poses
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion SLERP
# ---------------------------------------------------------------------------

def _quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.dot(q1, q2))


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (wxyz)."""
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))

    dot = _quat_dot(q0, q1)

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        # Very close – fall back to linear interpolation
        result = q0 + t * (q1 - q0)
        return _quat_normalize(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return _quat_normalize(s0 * q0 + s1 * q1)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_views(views: List[dict], num_interp: int) -> List[dict]:
    """Insert *num_interp* evenly-spaced frames between each pair of keyframes.

    The original keyframes are preserved (at their new indices).
    """
    if len(views) < 2 or num_interp < 1:
        return views

    result: List[dict] = []
    for i in range(len(views) - 1):
        p0 = np.array(views[i]["position"], dtype=np.float64)
        p1 = np.array(views[i + 1]["position"], dtype=np.float64)
        q0 = np.array(views[i]["quaternion_wxyz"], dtype=np.float64)
        q1 = np.array(views[i + 1]["quaternion_wxyz"], dtype=np.float64)

        total_segments = num_interp + 1
        for j in range(total_segments):
            t = j / total_segments
            pos = ((1.0 - t) * p0 + t * p1).tolist()
            quat = _slerp(q0, q1, t).tolist()
            result.append({
                "view_idx": len(result),
                "position": [float(v) for v in pos],
                "quaternion_wxyz": [float(v) for v in quat],
            })

    # Append the last keyframe
    last = views[-1]
    result.append({
        "view_idx": len(result),
        "position": [float(v) for v in last["position"]],
        "quaternion_wxyz": [float(v) for v in last["quaternion_wxyz"]],
    })
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate a sampled trajectory to a denser pose sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_json", help="Path to raw_poses JSON file.")
    parser.add_argument(
        "--num-interp", type=int, default=5,
        help="Number of interpolated frames to insert between each pair of keyframes.",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Output directory. Defaults to the same directory as input.",
    )
    parser.add_argument(
        "--output-name", default="",
        help="Output filename. Defaults to <input_stem>_interp<N>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_json)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    views_key = "views" if "views" in data else "frames"
    views = data[views_key]
    print(f"[interpolate] loaded {len(views)} keyframes from {input_path}")

    interpolated = interpolate_views(views, args.num_interp)
    print(f"[interpolate] produced {len(interpolated)} frames (num_interp={args.num_interp})")

    # Build output JSON – preserve all metadata, replace views
    out_data = {k: v for k, v in data.items() if k not in ("views", "frames")}
    out_data["max_views"] = len(interpolated)
    out_data["views"] = interpolated

    # Determine output path
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        out_name = args.output_name
    else:
        out_name = f"{input_path.stem}_interp{args.num_interp}.json"
    out_path = out_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"[interpolate] saved to {out_path}")


if __name__ == "__main__":
    main()
