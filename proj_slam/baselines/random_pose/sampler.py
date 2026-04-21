"""Random camera pose sampler for normalised GLB scenes.

Scenes are assumed to be normalised so that:
  X, Y ∈ [-1, 1]  (horizontal floor-plan extent)
  Z   ∈ [0,  1]   (vertical; Z=0 is the floor)

Occupancy ground-truth comes from the per-scene `.voxels.h5` sidecar.
A candidate position is *valid* iff:
  - it lies within the scene bounding box (with a small wall-safety margin), and
  - no occupied voxel exists within `collision_radius` voxels in XY and
    `collision_radius_z` voxels in Z around the candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Voxel grid loader
# ---------------------------------------------------------------------------

@dataclass
class VoxelGrid:
    grid:   np.ndarray   # (X, Y, Z) bool – True = occupied
    origin: np.ndarray   # (3,) float32 – world coord of voxel [0,0,0]
    pitch:  float        # voxel side length (m)

    @classmethod
    def from_h5(cls, h5_path: Path) -> "VoxelGrid":
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            grid   = f["grid"][:].astype(bool)
            origin = f["origin"][:].astype(np.float32)
            pitch  = float(f["pitch"][()])
        return cls(grid=grid, origin=origin, pitch=pitch)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.grid.shape   # (Nx, Ny, Nz)

    def world_to_idx(self, pos: np.ndarray) -> np.ndarray:
        return np.floor((pos - self.origin) / self.pitch).astype(np.int64)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) world-coord bounding box."""
        lo = self.origin
        hi = self.origin + np.array(self.shape, dtype=np.float32) * self.pitch
        return lo, hi


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def _is_valid(
    pos: np.ndarray,
    vg: VoxelGrid,
    xy_radius: int,
    z_radius:  int,
    border_voxels: int,
) -> bool:
    """Return True iff `pos` is free of occupied voxels in a neighbourhood."""
    idx = vg.world_to_idx(pos)
    Nx, Ny, Nz = vg.shape

    # Must be strictly inside the grid (keeping `border_voxels` clearance)
    bv = border_voxels
    if not (bv <= idx[0] < Nx - bv and
            bv <= idx[1] < Ny - bv and
            bv <= idx[2] < Nz - bv):
        return False

    # Check neighbourhood for occupied voxels
    ix0, iy0, iz0 = int(idx[0]), int(idx[1]), int(idx[2])
    for dx in range(-xy_radius, xy_radius + 1):
        for dy in range(-xy_radius, xy_radius + 1):
            for dz in range(-z_radius, z_radius + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                iz = iz0 + dz
                if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
                    if vg.grid[ix, iy, iz]:
                        return False
    return True


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _yaw_quat(yaw: float) -> np.ndarray:
    """Rotation around world +Z by `yaw` radians → wxyz quaternion."""
    return np.array(
        [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32
    )


def _pitch_quat(pitch: float) -> np.ndarray:
    """Rotation around world +Y by `pitch` radians → wxyz quaternion.

    Positive pitch tilts the camera nose down (looks slightly downward).
    """
    return np.array(
        [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0], dtype=np.float32
    )


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2  (both wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _random_orientation(
    rng: np.random.Generator,
    pitch_range: Tuple[float, float],
) -> np.ndarray:
    """Random yaw (uniform 0–2π) with random pitch in `pitch_range` (radians)."""
    yaw   = rng.uniform(0.0, 2.0 * np.pi)
    pitch = rng.uniform(*pitch_range)
    # Apply yaw first (world Z), then pitch (world Y after yaw)
    q = _quat_mul(_yaw_quat(yaw), _pitch_quat(pitch))
    q /= np.linalg.norm(q) + 1e-9
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------

@dataclass
class RandomSamplerConfig:
    # Camera height range above floor (Z in scene units)
    z_min: float = 0.1
    z_max: float = 0.9
    # Collision neighbourhood (voxels)
    xy_radius:     int = 8    # ~12 cm at pitch=0.015625 m
    z_radius:      int = 8    # ~12 cm
    border_voxels: int = 4    # keep this many voxels clearance from grid edge
    # Random pitch range (radians). 0 = horizontal; positive = nose-down.
    pitch_min: float = -1.0
    pitch_max: float =  1.0
    # Max resampling attempts per view before giving up
    max_attempts: int = 2000


def sample_random_poses(
    vg: VoxelGrid,
    n_views: int,
    cfg: RandomSamplerConfig,
    rng: np.random.Generator,
) -> List[dict]:
    """Sample `n_views` valid (position, orientation) pairs.

    Invalid candidates (inside walls, outside bounds) are silently skipped;
    a warning is printed if a view cannot be filled within `cfg.max_attempts`.
    """
    lo, hi = vg.bounds()

    # Constrain Z to camera-height band
    z_lo = lo[2] + cfg.z_min
    z_hi = lo[2] + cfg.z_max
    z_hi = min(z_hi, hi[2] - vg.pitch)   # never exceed grid top

    views: List[dict] = []
    for view_idx in range(n_views):
        placed = False
        for _ in range(cfg.max_attempts):
            pos = np.array([
                rng.uniform(lo[0], hi[0]),
                rng.uniform(lo[1], hi[1]),
                rng.uniform(z_lo, z_hi),
            ], dtype=np.float32)

            if _is_valid(pos, vg, cfg.xy_radius, cfg.z_radius, cfg.border_voxels):
                quat = _random_orientation(rng, (cfg.pitch_min, cfg.pitch_max))
                views.append({
                    "view_idx":        view_idx,
                    "position":        pos.tolist(),
                    "quaternion_wxyz": quat.tolist(),
                })
                placed = True
                break

        if not placed:
            print(f"  [warn] view {view_idx}: no valid position found after "
                  f"{cfg.max_attempts} attempts — skipped.")

    return views
