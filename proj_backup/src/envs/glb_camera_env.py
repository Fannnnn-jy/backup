from __future__ import annotations

import copy
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import sapien
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from src.utils.math import matrix_from_euler, quat_from_matrix
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig


def _as_tensor(value: torch.Tensor | Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _euler_to_quat(euler_xyz: torch.Tensor) -> torch.Tensor:
    rot = matrix_from_euler(euler_xyz, "XYZ")
    return quat_from_matrix(rot)


def _normalize_quat(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return quat / norm


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    # Both quaternions are (w, x, y, z)
    w1, x1, y1, z1 = torch.unbind(lhs, dim=-1)
    w2, x2, y2, z2 = torch.unbind(rhs, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def _scale_quat_rotation(quat: torch.Tensor, scale: float, eps: float = 1e-8) -> torch.Tensor:
    if scale == 1.0:
        return quat
    quat = _normalize_quat(quat, eps=eps)
    w = torch.clamp(quat[..., 0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    axis = quat[..., 1:]
    axis = torch.where(
        sin_half.unsqueeze(-1) > eps,
        axis / sin_half.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype),
    )
    new_angle = angle * scale
    new_w = torch.cos(new_angle / 2.0)
    new_xyz = axis * torch.sin(new_angle / 2.0).unsqueeze(-1)
    return _normalize_quat(torch.cat([new_w.unsqueeze(-1), new_xyz], dim=-1), eps=eps)


@register_env("GLBCameraAgent-v1", max_episode_steps=200)
class GLBCameraAgentEnv(BaseEnv):
    """A minimal ManiSkill environment with a movable camera agent in a GLB scene."""

    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        glb_path: str,
        y_up: bool = True,
        add_collision: bool = False,
        scale: float = 1.0,
        camera_uid: str = "agent_camera",
        camera_width: int = 256,
        camera_height: int = 256,
        camera_fov: float = 1.0,
        camera_near: float = 0.01,
        camera_far: float = 100.0,
        init_position: Sequence[float] = (2.0, 2.0, 1.8),
        init_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        init_euler: Optional[Sequence[float]] = None,
        action_mode: str = "delta_quat",
        action_scale: float = 0.1,
        action_scale_pos: Optional[float] = None,
        action_scale_rot: Optional[float] = None,
        position_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        **kwargs,
    ):
        if "new_sensor_configs" not in kwargs or kwargs["new_sensor_configs"] is None:
            kwargs["new_sensor_configs"] = {}
        self.glb_path = glb_path
        self.y_up = y_up
        self.add_collision = add_collision
        self.scale = float(scale)

        self.camera_uid = camera_uid
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov = float(camera_fov)
        self.camera_near = float(camera_near)
        self.camera_far = float(camera_far)

        self._init_position_np = np.asarray(init_position, dtype=np.float32)
        self._init_quat_np = np.asarray(init_quat, dtype=np.float32)
        if init_euler is not None:
            init_euler_np = np.asarray(init_euler, dtype=np.float32)
            euler = torch.tensor(init_euler_np, dtype=torch.float32)
            self._init_quat_np = _euler_to_quat(euler.unsqueeze(0))[0].cpu().numpy().astype(np.float32)

        self.action_mode = action_mode
        self.action_scale_pos = float(action_scale_pos) if action_scale_pos is not None else float(action_scale)
        self.action_scale_rot = float(action_scale_rot) if action_scale_rot is not None else float(action_scale)

        self._position_bounds_np = None
        if position_bounds is not None:
            low, high = position_bounds
            self._position_bounds_np = (
                np.asarray(low, dtype=np.float32),
                np.asarray(high, dtype=np.float32),
            )

        self._agent_pos: Optional[torch.Tensor] = None
        self._agent_quat: Optional[torch.Tensor] = None
        self._agent_camera = None
        self._pos_low: Optional[torch.Tensor] = None
        self._pos_high: Optional[torch.Tensor] = None

        super().__init__(*args, robot_uids="none", **kwargs)
        self._setup_action_space()

    @property
    def _default_sim_config(self):
        return SimConfig()

    def _initial_camera_pose(self) -> sapien.Pose:
        quat = _normalize_quat(torch.tensor(self._init_quat_np, dtype=torch.float32)).cpu().numpy()
        return sapien.Pose(p=self._init_position_np, q=quat)

    @property
    def _default_sensor_configs(self):
        pose = self._initial_camera_pose()
        return [
            CameraConfig(
                self.camera_uid,
                pose=pose,
                width=self.camera_width,
                height=self.camera_height,
                fov=self.camera_fov,
                near=self.camera_near,
                far=self.camera_far,
            )
        ]

    def _setup_sensors(self, options: dict):
        super()._setup_sensors(options)
        if self.camera_uid in self._sensors:
            self._agent_camera = self._sensors[self.camera_uid].camera
        else:
            self._agent_camera = None

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()

        pose = sapien.Pose()
        if self.y_up:
            theta = np.deg2rad(90.0)
            q = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
            pose = sapien.Pose(q=q)

        scale = [self.scale] * 3
        builder.add_visual_from_file(self.glb_path, pose=pose, scale=scale)
        if self.add_collision:
            builder.add_nonconvex_collision_from_file(self.glb_path, pose=pose, scale=scale)
        builder.initial_pose = sapien.Pose()
        builder.build_static(name="glb_scene")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self._agent_pos is None or self._agent_pos.shape[0] != self.num_envs:
            self._agent_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._agent_quat = torch.zeros((self.num_envs, 4), device=self.device)

        init_pos = _as_tensor(self._init_position_np, device=self.device, dtype=torch.float32)
        init_quat = _as_tensor(self._init_quat_np, device=self.device, dtype=torch.float32)
        init_quat = _normalize_quat(init_quat)
        self._agent_pos[env_idx] = init_pos
        self._agent_quat[env_idx] = init_quat

        self._ensure_bounds_tensors()
        self._clamp_pose()
        self._update_camera_pose()

    def _ensure_bounds_tensors(self) -> None:
        if self._position_bounds_np is not None and self._pos_low is None:
            low, high = self._position_bounds_np
            self._pos_low = _as_tensor(low, device=self.device, dtype=torch.float32)
            self._pos_high = _as_tensor(high, device=self.device, dtype=torch.float32)

    def _clamp_pose(self) -> None:
        if self._agent_pos is None or self._agent_quat is None:
            return
        if self._pos_low is not None and self._pos_high is not None:
            self._agent_pos = torch.clamp(self._agent_pos, min=self._pos_low, max=self._pos_high)
        self._agent_quat = _normalize_quat(self._agent_quat)

    def _update_camera_pose(self) -> None:
        if self._agent_camera is None or self._agent_pos is None or self._agent_quat is None:
            return
        quat = _normalize_quat(self._agent_quat)
        pose = sapien_utils.Pose.create_from_pq(p=self._agent_pos, q=quat)
        cam = self._agent_camera
        if hasattr(cam, "set_pose"):
            cam.set_pose(pose.sp)
        elif hasattr(cam, "set_local_pose"):
            cam.set_local_pose(pose.sp)
        elif hasattr(cam, "mount") and cam.mount is not None:
            cam.mount.set_pose(pose.sp)
        else:  # pragma: no cover
            raise RuntimeError("Camera pose setter not found for SAPIEN camera.")

    def _setup_action_space(self) -> None:
        action_dim = self._action_dim()
        if self.action_mode.startswith("delta"):
            low = -np.ones(action_dim, dtype=np.float32)
            high = np.ones(action_dim, dtype=np.float32)
        else:
            low = np.full(action_dim, -np.inf, dtype=np.float32)
            high = np.full(action_dim, np.inf, dtype=np.float32)
            if self._position_bounds_np is not None:
                pos_low, pos_high = self._position_bounds_np
                low[:3] = pos_low
                high[:3] = pos_high
            # keep quaternion entries bounded to a sensible range before normalization
            low[3:7] = -1.0
            high[3:7] = 1.0
        self.single_action_space = spaces.Box(low=low, high=high, shape=(action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)
        self._orig_single_action_space = copy.deepcopy(self.single_action_space)

    def _action_dim(self) -> int:
        if self.action_mode in {"delta_quat", "absolute_quat"}:
            return 7
        raise ValueError(f"Unsupported action_mode: {self.action_mode}")

    def _process_action(self, action):
        if action is None:
            return None
        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            else:
                raise TypeError("Dictionary actions are not supported for GLBCameraAgentEnv.")
        action = common.to_tensor(action, device=self.device).to(dtype=torch.float32)
        if action.shape == self._orig_single_action_space.shape and self.num_envs == 1:
            action = common.batch(action)
        low = torch.as_tensor(self.single_action_space.low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(self.single_action_space.high, device=action.device, dtype=action.dtype)
        action = torch.clamp(action, min=low, max=high)
        return action

    def _apply_action(self, action: torch.Tensor) -> None:
        if self._agent_pos is None or self._agent_quat is None:
            return
        if self.action_mode == "delta_quat":
            self._agent_pos = self._agent_pos + action[:, :3] * self.action_scale_pos
            delta_quat = _normalize_quat(action[:, 3:])
            if self.action_scale_rot != 1.0:
                delta_quat = _scale_quat_rotation(delta_quat, self.action_scale_rot)
            self._agent_quat = _quat_multiply(delta_quat, self._agent_quat)
        elif self.action_mode == "absolute_quat":
            self._agent_pos = action[:, :3]
            self._agent_quat = _normalize_quat(action[:, 3:])
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self._clamp_pose()
        self._update_camera_pose()

    def _step_action(self, action):
        action = self._process_action(action)
        if action is not None:
            self._apply_action(action)
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return action

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        if self._agent_pos is None or self._agent_quat is None:
            return dict()
        return {
            "agent_pos": self._agent_pos.clone(),
            "agent_quat": self._agent_quat.clone(),
        }

    def _get_obs_agent(self):
        # No robot agent in this environment.
        return dict()

    def compute_dense_reward(self, obs, action, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def get_state_dict(self):
        state = self.scene.get_sim_state()
        if self._agent_pos is not None and self._agent_quat is not None:
            state["camera_pose"] = torch.cat([self._agent_pos, self._agent_quat], dim=-1)
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        camera_pose = state.pop("camera_pose", None)
        self.scene.set_sim_state(state, env_idx)
        if camera_pose is not None:
            self._set_pose_from_state(camera_pose, env_idx)
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    def _set_pose_from_state(self, camera_pose: torch.Tensor, env_idx: Optional[torch.Tensor]) -> None:
        if self._agent_pos is None or self._agent_quat is None:
            self._agent_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._agent_quat = torch.zeros((self.num_envs, 4), device=self.device)
        camera_pose = common.to_tensor(camera_pose, device=self.device).to(dtype=torch.float32)
        if camera_pose.shape[-1] != 7:
            raise ValueError(f"Expected camera_pose with 7 dims, got shape {camera_pose.shape}.")
        if camera_pose.ndim == 1:
            camera_pose = camera_pose.unsqueeze(0)
        if env_idx is None:
            self._agent_pos = camera_pose[:, :3]
            self._agent_quat = camera_pose[:, 3:]
        else:
            self._agent_pos[env_idx] = camera_pose[:, :3]
            self._agent_quat[env_idx] = camera_pose[:, 3:]
        self._clamp_pose()
        self._update_camera_pose()

    def set_agent_pose(
        self,
        position: Optional[Sequence[float] | torch.Tensor] = None,
        quat: Optional[Sequence[float] | torch.Tensor] = None,
        env_idx: Optional[Iterable[int] | torch.Tensor] = None,
    ) -> None:
        if self._agent_pos is None or self._agent_quat is None:
            self._agent_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._agent_quat = torch.zeros((self.num_envs, 4), device=self.device)

        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        else:
            env_idx = common.to_tensor(env_idx, device=self.device, dtype=torch.int64)

        if position is not None:
            position = _as_tensor(position, device=self.device, dtype=torch.float32)
            if position.ndim == 1:
                position = position.unsqueeze(0)
            self._agent_pos[env_idx] = position
        if quat is not None:
            quat = _as_tensor(quat, device=self.device, dtype=torch.float32)
            if quat.ndim == 1:
                quat = quat.unsqueeze(0)
            self._agent_quat[env_idx] = quat
        self._clamp_pose()
        self._update_camera_pose()

    def get_agent_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._agent_pos is None or self._agent_quat is None:
            raise RuntimeError("Agent pose is not initialized yet.")
        return self._agent_pos.clone(), self._agent_quat.clone()

    def render_agent_rgb(
        self,
        position: Optional[Sequence[float] | torch.Tensor] = None,
        quat: Optional[Sequence[float] | torch.Tensor] = None,
        env_idx: Optional[Iterable[int] | torch.Tensor] = None,
    ) -> torch.Tensor:
        if position is not None or quat is not None:
            self.set_agent_pose(position=position, quat=quat, env_idx=env_idx)
        if self.camera_uid not in self._sensors:
            raise RuntimeError(f"Camera uid '{self.camera_uid}' is not available.")
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self._sensors[self.camera_uid].capture()
        obs = self._sensors[self.camera_uid].get_obs(
            rgb=True,
            depth=False,
            position=False,
            segmentation=False,
            normal=False,
            albedo=False,
        )
        return obs["rgb"]
