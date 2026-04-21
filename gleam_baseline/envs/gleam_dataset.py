from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


@dataclass
class GleamScene:
    range_gt: np.ndarray
    voxel_size: np.ndarray
    layout_mask: Optional[np.ndarray]
    obstacle_mask_3d: Optional[np.ndarray]
    init_positions: Optional[np.ndarray]


class GleamDataset:
    def __init__(
        self,
        data_dir: Path | str,
        dataset_name: str | None,
        grid_size: int,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required to load GLEAM dataset files.")

        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name or self.data_dir.name
        self.grid_size = int(grid_size)

        self._range_gt: np.ndarray
        self._voxel_size: np.ndarray
        self._layout_mask: Optional[np.ndarray]
        self._init_positions: Optional[List[np.ndarray]]
        self._grid_dict_path: Optional[Path] = None
        self._obstacle_mask_3d_cache: dict[int, np.ndarray] = {}
        self._load()

    @property
    def num_scenes(self) -> int:
        return int(self._range_gt.shape[0])

    def get_scene(self, scene_index: int) -> GleamScene:
        idx = int(scene_index)
        return GleamScene(
            range_gt=self._range_gt[idx],
            voxel_size=self._voxel_size[idx],
            layout_mask=None if self._layout_mask is None else self._layout_mask[idx],
            obstacle_mask_3d=self._load_obstacle_mask_3d(idx),
            init_positions=None if self._init_positions is None else self._init_positions[idx],
        )

    def _load(self) -> None:
        gt_dir = self.data_dir / "gt"
        dataset = self.dataset_name

        range_gt = torch.load(
            gt_dir / f"{dataset}_{self.grid_size}_range_gt.pt", map_location="cpu"
        )
        voxel_size = torch.load(
            gt_dir / f"{dataset}_{self.grid_size}_voxel_size_gt.pt", map_location="cpu"
        )

        layout_path = gt_dir / f"{dataset}_{self.grid_size}_occ_map_height_1d5_gt.pt"
        if layout_path.exists():
            layout_maps = torch.load(layout_path, map_location="cpu").to(torch.float32)
            layout_maps = (layout_maps / 255.0).numpy()
            layout_mask = layout_maps > 0.0
        else:
            layout_mask = None

        grid_dict_path = gt_dir / f"{dataset}_{self.grid_size}_grid_dict_gt.h5"
        if grid_dict_path.exists():
            if h5py is None:  # pragma: no cover
                raise ImportError("h5py is required to load GLEAM grid_dict_gt.h5 files.")
            self._grid_dict_path = grid_dict_path

        init_positions = self._load_init_positions(gt_dir, range_gt)

        self._range_gt = range_gt.numpy()
        self._voxel_size = voxel_size.numpy()
        self._layout_mask = layout_mask
        self._init_positions = init_positions

    def _load_obstacle_mask_3d(self, scene_index: int) -> Optional[np.ndarray]:
        if self._grid_dict_path is None:
            return None
        if scene_index in self._obstacle_mask_3d_cache:
            return self._obstacle_mask_3d_cache[scene_index]
        if h5py is None:  # pragma: no cover
            return None
        with h5py.File(self._grid_dict_path, "r") as handle:
            key = f"scene_{scene_index}"
            if key not in handle:
                return None
            idx_coord = handle[key][:]
        if idx_coord.size == 0:
            mask = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
            self._obstacle_mask_3d_cache[scene_index] = mask
            return mask
        indices = idx_coord[:, :3].astype(np.int64)
        mask = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        self._obstacle_mask_3d_cache[scene_index] = mask
        return mask

    def _load_init_positions(
        self, gt_dir: Path, range_gt: "torch.Tensor"
    ) -> Optional[List[np.ndarray]]:
        dataset = self.dataset_name
        init_10_path = gt_dir / f"{dataset}_{self.grid_size}_init_10.pt"
        if init_10_path.exists():
            init_list = torch.load(init_10_path, map_location="cpu")
            return [item.numpy().astype(np.float32) for item in init_list]

        init_map_path = gt_dir / f"{dataset}_{self.grid_size}_init_map_1d5.pt"
        if not init_map_path.exists():
            return None

        init_map = torch.load(init_map_path, map_location="cpu").to(torch.float32) / 255.0
        positions: List[np.ndarray] = []
        for idx in range(init_map.shape[0]):
            coords = torch.nonzero(init_map[idx])
            if coords.numel() == 0:
                positions.append(np.zeros((0, 2), dtype=np.float32))
                continue
            norm = coords.float() / float(self.grid_size - 1) * 2.0 - 1.0
            ranges = range_gt[idx, :4:2]
            positions.append((norm * ranges).numpy().astype(np.float32))
        return positions
