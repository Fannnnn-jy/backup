from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from envs.active_mapping_env import ActiveMappingConfig, ActiveMappingEnv


class GymnasiumVecEnv:
    """Minimal VecEnv adapter for rsl_rl using gymnasium.vector.SyncVectorEnv."""

    def __init__(self, cfg: ActiveMappingConfig, num_envs: int, seed: int, device: torch.device):
        env_cls = self._resolve_env_class(cfg.sim_backend)

        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _make() -> gym.Env:
                env_cfg = replace(cfg, seed=seed + rank)
                return env_cls(env_cfg)

            return _make

        self.env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)], copy=False)
        self.device = device
        self.num_envs = num_envs
        self._seeds = [seed + i for i in range(num_envs)]
        # Keep the full env config for loggers (e.g., wandb expects a dataclass/object).
        self.cfg = cfg

        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        if not isinstance(obs_space, gym.spaces.Dict):
            raise ValueError("ActiveMappingEnv must return a dict observation.")

        self.obs_keys = list(obs_space.spaces.keys())
        image_key = "obs_rgb_tensor" if "obs_rgb_tensor" in obs_space.spaces else "obs_depth_tensor"
        self.obs_image_key = image_key
        self.num_obs = int(np.prod(obs_space.spaces[image_key].shape))
        self.num_privileged_obs = 0
        self.num_actions = act_space.shape[0]
        self.max_episode_length = self.env.envs[0].cfg.max_steps
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._obs = None
        self._obs_td = None
        self.reset()

    @staticmethod
    def _resolve_env_class(sim_backend: str):
        backend = (sim_backend or "stub").lower()
        if backend in ("maniskill", "maniskill3"):
            from envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv

            return ManiSkillActiveMappingEnv
        if backend not in ("stub", "gym"):
            raise ValueError(f"Unsupported sim_backend: {sim_backend}")
        return ActiveMappingEnv

    def _process_obs(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        processed = {}
        for key, value in obs_dict.items():
            processed[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
        return processed

    def _as_tensordict(self, obs_dict: Dict[str, torch.Tensor]) -> TensorDict:
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        obs, _ = self.env.reset(seed=self._seeds)
        obs_dict = self._process_obs(obs)
        self._obs = obs_dict
        self._obs_td = self._as_tensordict(obs_dict)
        self.episode_length_buf.zero_()
        return self._obs_td

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
        dones = np.logical_or(terminated, truncated)
        obs_dict = self._process_obs(obs)
        self._obs = obs_dict
        self._obs_td = self._as_tensordict(obs_dict)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.bool)
        infos_out = dict(infos) if isinstance(infos, dict) else {}
        infos_out["time_outs"] = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        self.episode_length_buf += 1
        self.episode_length_buf[dones_t] = 0
        return self._obs_td, rewards_t, dones_t, infos_out

    def get_observations(self):
        return self._obs_td

    def get_privileged_observations(self):
        return None
