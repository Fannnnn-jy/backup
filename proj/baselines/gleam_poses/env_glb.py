"""Single-scene GLB environment for GLEAM pose generation.

Inherits from Env_GLEAM_Stage1 and overrides:
  - num_scene = 1
  - motion_height (configurable, default 0.5 for Z∈[0,1] scenes)
  - _additional_create: loads URDF from a custom directory
  - _init_load_all:     loads GT tensors from a custom directory
"""

import os
import torch
from isaacgym import gymapi

# GLEAM root must be on sys.path before importing
from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1
from gleam.env.env_gleam_base   import Env_GLEAM_Base


class Env_GLEAM_SingleGLB(Env_GLEAM_Stage1):
    """GLEAM environment configured for a single pre-processed GLB scene."""

    def __init__(self, *args,
                 glb_data_dir: str,
                 dataset_name: str,
                 drone_height: float = 0.5,
                 **kwargs):
        """
        Args:
            glb_data_dir:  directory that contains the preprocessed data
                           (sub-dirs: urdf/, gt/).
            dataset_name:  name used in the preprocessed file names.
            drone_height:  Z-coordinate at which the drone flies (default 0.5).
        """
        self._glb_data_dir   = glb_data_dir
        self._dataset_name   = dataset_name
        self._drone_height   = drone_height

        self.num_scene = 1
        self.visualize_flag = False

        # Skip Env_GLEAM_Base.__init__ (num_scene override must happen first)
        super(Env_GLEAM_Base, self).__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _additional_create(self, env_handle, env_index):
        """Load the single pre-processed scene URDF."""
        urdf_path = os.path.join(self._glb_data_dir, "urdf", self._dataset_name)
        urdf_name = "scene_0.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link  = True
        asset_options.disable_gravity = True

        asset = self.gym.load_asset(self.sim, urdf_path, urdf_name, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(
            self.env_origins[env_index][0],
            self.env_origins[env_index][1],
            self.env_origins[env_index][2],
        )
        pose.r = gymapi.Quat(0, 0, 0, 1)   # mesh already Z-up from preprocessing
        self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)

        self.additional_actors[env_index] = [1]

    # ------------------------------------------------------------------
    # GT data loading
    # ------------------------------------------------------------------

    def _init_load_all(self):
        self.grid_size    = 128
        self.motion_height = self._drone_height

        gt_path = os.path.join(self._glb_data_dir, "gt", f"gt_{self._dataset_name}", "")
        name    = self._dataset_name
        g       = self.grid_size
        dev     = self.device

        # [1, 3]
        self.voxel_size_gt = torch.load(
            gt_path + f"{name}_{g}_voxel_size_gt.pt", map_location=dev)[:1]

        # [1, 6]  (x_max, x_min, y_max, y_min, z_max, z_min)
        self.range_gt = torch.load(
            gt_path + f"{name}_{g}_range_gt.pt", map_location=dev)[:1]

        # occupancy map filename uses the drone height (e.g. 0d5)
        h_str = (f"{int(self._drone_height)}"
                 if self._drone_height == int(self._drone_height)
                 else f"{self._drone_height:.1f}".replace(".", "d"))
        occ_fname = f"{name}_{g}_occ_map_height_{h_str}_gt.pt"
        self.layout_maps_height = torch.load(
            gt_path + occ_fname, map_location=dev)[:1].to(torch.float16)
        self.layout_maps_height /= 255.

        # valid pixel count  [1]
        self.num_valid_pixel_gt = self.layout_maps_height.sum(dim=(1, 2))

        # initial positions from init_map
        init_maps = torch.load(
            gt_path + f"{name}_{g}_init_map_{h_str}.pt", map_location=dev)[:1]
        init_maps /= 255.
        self.init_maps_list = [
            (torch.nonzero(init_maps[idx]) / (self.grid_size - 1) * 2 - 1)
            * self.range_gt[idx, :4:2]
            for idx in range(1)
        ]

        print(f"[GLB env] Loaded GT data from {gt_path}")
        print(f"[GLB env] range_gt = {self.range_gt}")
        print(f"[GLB env] motion_height = {self.motion_height}")
