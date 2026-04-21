"""Single-scene GLB environment for GLEAM pose generation.

Inherits from Env_GLEAM_Stage1 and overrides:
  - num_scene = 1
  - motion_height (configurable, default 0.5 for Z∈[0,1] scenes)
  - _additional_create: loads URDF from a custom directory
  - _init_load_all:     loads GT tensors from a custom directory
"""

import os
import random
from typing import Optional, Tuple
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
                 forced_init_xy: Optional[Tuple[float, float]] = None,
                 forced_init_yaw: Optional[float] = None,
                 **kwargs):
        """
        Args:
            glb_data_dir:    directory that contains the preprocessed data
                             (sub-dirs: urdf/, gt/).
            dataset_name:    name used in the preprocessed file names.
            drone_height:    Z-coordinate at which the drone flies (default 0.5).
            forced_init_xy:  if given, always start at this (x, y) in Z-up world
                             coordinates instead of sampling from the init map.
            forced_init_yaw: if given, always start facing this yaw (radians,
                             Z-up, CCW from +X) instead of the reset default.
        """
        self._glb_data_dir    = glb_data_dir
        self._dataset_name    = dataset_name
        self._drone_height    = drone_height
        self._forced_init_xy  = forced_init_xy   # (x, y) in Z-up world coords, or None
        self._forced_init_yaw = forced_init_yaw  # float (radians) or None

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
        layout_maps_height_cpu = torch.load(
            gt_path + occ_fname, map_location="cpu")[:1].to(torch.float32)
        self.layout_maps_height = (layout_maps_height_cpu / 255.).to(dev).to(torch.float16)

        # valid pixel count  [1]
        self.num_valid_pixel_gt = self.layout_maps_height.sum(dim=(1, 2))

        # initial positions from init_map
        init_maps_cpu = torch.load(
            gt_path + f"{name}_{g}_init_map_{h_str}.pt", map_location="cpu")[:1].to(torch.float32)
        init_maps_cpu /= 255.
        self.init_maps_list = []
        for idx in range(1):
            range_xy_cpu = self.range_gt[idx, :4:2].detach().cpu()
            init_positions = (
                (torch.nonzero(init_maps_cpu[idx] > 0) / (self.grid_size - 1) * 2 - 1)
                * range_xy_cpu
            ).to(dev)
            free_positions = torch.nonzero(layout_maps_height_cpu[idx] < 127.5)
            print(
                f"[GLB env] scene={idx} init_candidates={len(init_positions)} "
                f"free_cells={len(free_positions)} "
                f"init_sum={float(init_maps_cpu[idx].sum())} "
                f"occ_sum={float(layout_maps_height_cpu[idx].sum())}"
            )
            if init_positions.numel() == 0:
                if free_positions.numel() > 0:
                    init_positions = (
                        (free_positions / (self.grid_size - 1) * 2 - 1)
                        * range_xy_cpu
                    ).to(dev)
                    print(
                        f"[GLB env] WARNING: init_map is empty; "
                        f"falling back to {len(free_positions)} free occupancy cells"
                    )
                else:
                    init_positions = torch.zeros((1, 2), dtype=torch.float32, device=dev)
                    print(
                        "[GLB env] WARNING: init_map and occupancy map are both empty; "
                        "falling back to origin"
                    )
            self.init_maps_list.append(init_positions)

        print(f"[GLB env] Loaded GT data from {gt_path}")
        print(f"[GLB env] range_gt = {self.range_gt}")
        print(f"[GLB env] motion_height = {self.motion_height}")
        print("[GLB env] _init_load_all: complete, returning to _init_buffers")

    def reset_idx(self, env_ids):
        """Single-scene reset that bypasses Stage1 multi-scene swapping logic."""
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        if env_ids.numel() == 0:
            return

        device = self.device
        env_ids_list = [int(env_idx) for env_idx in env_ids.detach().cpu().tolist()]
        num_env_ids = len(env_ids_list)

        # Keep scene bookkeeping fixed for the single-scene case.
        self.scene_per_env = 1
        self.active_scene_ids = [0 for _ in range(self.num_envs)]
        self.inactive_scene_ids = []
        self.env_to_scene = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.range_gt_scenes = self.range_gt[self.env_to_scene]
        self.voxel_size_gt_scenes = self.voxel_size_gt[self.env_to_scene]
        self.num_valid_pixel_gt_scenes = self.num_valid_pixel_gt[self.env_to_scene]
        self.layout_maps_height_scenes = self.layout_maps_height[self.env_to_scene]

        self._reset_root_states(env_ids)

        self.ego_pose_buf[env_ids] = torch.zeros(
            num_env_ids, self.buffer_size, self.pose_size, dtype=torch.float32, device=device
        )
        self.world_pose_buf[env_ids] = torch.zeros(
            num_env_ids, self.buffer_size, self.pose_size, dtype=torch.float32, device=device
        )

        for buf_idx in range(self.buffer_size):
            self.reward_layout_ratio_buf[buf_idx][env_ids] = torch.zeros(
                num_env_ids, dtype=torch.float32, device=device
            )

        self.actions[env_ids] = self._init_action_tensor(num_env_ids, device=device)

        init_positions = self.init_maps_list[0]
        if self._forced_init_xy is not None:
            # Use the externally specified starting position.
            xy = torch.tensor(
                [self._forced_init_xy[0], self._forced_init_xy[1]],
                dtype=torch.float32, device=self.device,
            )
            self.poses[env_ids, :2] = xy.unsqueeze(0).expand(num_env_ids, -1)
        else:
            if init_positions.ndim != 2 or init_positions.shape[0] == 0:
                raise ValueError(f"[GLB env] invalid init_positions shape: {tuple(init_positions.shape)}")
            choice_idx = torch.randint(
                low=0,
                high=init_positions.shape[0],
                size=(num_env_ids,),
                device=init_positions.device,
            )
            self.poses[env_ids, :2] = init_positions[choice_idx]
        self.poses[env_ids, 2] = self.motion_height

        if self._forced_init_yaw is not None:
            self.poses[env_ids, 5] = self._forced_init_yaw

        if self.visualize_flag:
            for env_idx in env_ids_list:
                self.local_paths[env_idx] = []

        self.prob_map[env_ids] = torch.zeros(
            num_env_ids, self.grid_size, self.grid_size, dtype=torch.float32, device=device
        )
        self.scanned_gt_map[env_ids] = torch.zeros(
            num_env_ids, self.grid_size, self.grid_size, dtype=torch.float32, device=device
        )

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ------------------------------------------------------------------
    # Diagnostic override: mirror parent _init_buffers_visual exactly,
    # adding per-call prints to locate any hang.
    # ------------------------------------------------------------------

    def _init_buffers_visual(self):
        from isaacgym import gymtorch
        print("[GLB env] _init_buffers_visual: start")
        print(f"[GLB env]   cfg.return_visual_observation={self.cfg.return_visual_observation}")
        if not self.cfg.return_visual_observation:
            print("[GLB env] _init_buffers_visual: skipped (return_visual_observation=False)")
            return
        print(f"[GLB env]   num_envs={self.num_envs}  num_cam={self.num_cam}  visualize_flag={self.visualize_flag}")

        # RGB tensors (only for visualization)
        if self.visualize_flag:
            self.rgb_cam_tensors = []
            for i in range(self.num_envs):
                for cam_idx in range(self.num_cam):
                    print(f"[GLB env]   env={i} cam={cam_idx} → get_camera_image_gpu_tensor (COLOR) ...")
                    im_rgb = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i],
                        self.camera_handles[i * self.num_cam + cam_idx],
                        gymapi.IMAGE_COLOR)
                    print(f"[GLB env]   wrap_tensor COLOR ...")
                    torch_cam_tensor_rgb = gymtorch.wrap_tensor(im_rgb)
                    self.rgb_cam_tensors.append(torch_cam_tensor_rgb)
                    print(f"[GLB env]   done rgb cam {cam_idx}")

            self.rgb_cam_col_tensors = []
            for i in range(self.num_envs):
                print(f"[GLB env]   env={i} → get_camera_image_gpu_tensor (COLOR col) ...")
                im_rgb_col = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i],
                    self.camera_col_handles[i],
                    gymapi.IMAGE_COLOR)
                print(f"[GLB env]   wrap_tensor COLOR_COL ...")
                torch_cam_tensor_rgb_col = gymtorch.wrap_tensor(im_rgb_col)
                self.rgb_cam_col_tensors.append(torch_cam_tensor_rgb_col)
                print(f"[GLB env]   done rgb_col env {i}")

        # Depth tensors (always)
        self.depth_cam_tensors = []
        for i in range(self.num_envs):
            for cam_idx in range(self.num_cam):
                print(f"[GLB env]   env={i} cam={cam_idx} → get_camera_image_gpu_tensor (DEPTH) ...")
                im_depth = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i],
                    self.camera_handles[i * self.num_cam + cam_idx],
                    gymapi.IMAGE_DEPTH)
                print(f"[GLB env]   wrap_tensor DEPTH ...")
                torch_cam_tensor_depth = gymtorch.wrap_tensor(im_depth)
                self.depth_cam_tensors.append(torch_cam_tensor_depth)
                print(f"[GLB env]   done depth cam {cam_idx}")

        self.depth_cam_col_tensors = []
        for i in range(self.num_envs):
            print(f"[GLB env]   env={i} → get_camera_image_gpu_tensor (DEPTH col) ...")
            im_depth_col = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i],
                self.camera_col_handles[i],
                gymapi.IMAGE_DEPTH)
            print(f"[GLB env]   wrap_tensor DEPTH_COL ...")
            torch_cam_tensor_depth_col = gymtorch.wrap_tensor(im_depth_col)
            self.depth_cam_col_tensors.append(torch_cam_tensor_depth_col)
            print(f"[GLB env]   done depth_col env {i}")

        print("[GLB env] _init_buffers_visual: done")
