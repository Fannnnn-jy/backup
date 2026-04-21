from __future__ import annotations

import os 
# os.environ['MUJOCO_GL'] = 'egl'
 
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional
 
import gymnasium as gym
import numpy as np
import torch
import tyro
from rsl_rl.modules import ActorCritic, ActorCriticCNN
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from tensordict import TensorDict
from torch import nn
 
# from gymnasium.wrappers import *
from gymnasium.wrappers import AddRenderObservation, FrameStackObservation, ResizeObservation
import torchvision.transforms.functional as TF
from torch.distributions import Normal
from typing import Any

class GymnasiumVecEnv:
    """Minimal VecEnv adapter for rsl_rl using gymnasium.vector.SyncVectorEnv."""
 
    def __init__(self, env_id: str, num_envs: int, seed: int, device: torch.device):
        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _make() -> gym.Env:
                env = gym.make(env_id, render_mode="rgb_array") ### render_mode 还有一种是 rgb_array_list，
                
                env = gym.wrappers.AddRenderObservation(env, render_only=True)
                env = gym.wrappers.ResizeObservation(env, (64, 64))
                # env = gym.wrappers.FrameStackObservation(env, stack_size=4) # 不许使用 FrameStack
                env = gym.wrappers.RecordEpisodeStatistics(env) # 统计 wrapper，基于这个的位置有一些思考
                
                return env
 
            return _make
 
        self.env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)], copy=False)
        self.device = device
        self.num_envs = num_envs
        self._seeds = [seed + i for i in range(num_envs)]
        self.cfg = {"env_id": env_id, "num_envs": num_envs, "seed": seed}
 
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        """
        直接 print 看一下结构：
        原来的 param-based 的 Hopper-v4 环境下：
        # Observation space: Box(-inf, inf, (11,), float64)
        # Action space: Box(-1.0, 1.0, (3,), float32)

        rgb_based 的 Hopper-v4 环境下：
        # Observation space: Box(0, 255, (64, 64, 3), uint8),
        # Action space: Box(-1.0, 1.0, (3,), float32)
        """

        # if not isinstance(obs_space, gym.spaces.Box) or not isinstance(act_space, gym.spaces.Box):
        #     raise ValueError("This example only supports continuous Box spaces.")
        # if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
        #     raise ValueError("This example only supports 1D observation/action spaces.")

        if len(obs_space.shape) == 3:  # (H, W, C) 单帧图像
            self.obs_shape = obs_space.shape
            self.is_image = True
        elif len(obs_space.shape) == 4: # (Stack, H, W, C) 堆叠图像
            assert False, "不许用framepack！"
            self.obs_shape = obs_space.shape
            self.is_image = True
        else: # (11,) 原始向量
            assert False, "不许用除了rgb以外任何信息！"
            self.obs_shape = obs_space.shape
            self.is_image = False

 
        # self.num_obs = obs_space.shape[0]
        self.num_obs = np.prod(self.obs_shape)
        self.num_privileged_obs = 0
        self.num_actions = act_space.shape[0]
        self.max_episode_length = self.env.envs[0].spec.max_episode_steps or 1000
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._obs = None
        self._obs_td = None
        self.reset()
    
    def _process_obs(self, obs):
        obs_tensor = obs.to(self.device).float() / 255.0
        if self.is_image:
            # (8, 64, 64, 3) -> (8, 3, 64, 64)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2) 
            
        return obs_tensor
    
    def _as_tensordict(self, obs: torch.Tensor) -> TensorDict: # rsl_rl 库所要求的观测格式 TensorDict
        """
        直接 print 看一下结构，八个环境并行产生如下的观测：
        TensorDict(
            fields={
                obs: Tensor(shape=torch.Size([8, 11]), device=cuda:0, dtype=torch.float32, is_shared=True)},
            batch_size=torch.Size([8]),
            device=cuda:0,
            is_shared=True)
        """
        obs = self._process_obs(obs)
        # print(f"Processed obs shape: {obs.shape}") # (N, 12, 64, 64)
        return TensorDict({"obs": obs}, batch_size=[self.num_envs], device=obs.device)
 
    def reset(self):
        obs, _ = self.env.reset(seed=self._seeds) # 迭代的重置每个子环境，注意到最核心的代码没有 return，需要在子类（特定环境类）定义的时候用 __super__ 
        self._obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self._obs_td = self._as_tensordict(self._obs)
        self.episode_length_buf.zero_()
        return self._obs_td
 
    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
        dones = np.logical_or(terminated, truncated)
        self._obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self._obs_td = self._as_tensordict(self._obs)
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
 
 
def _activation_from_name(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Unsupported activation: {name}")
 
 
def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], activation: str) -> nn.Sequential:
    act = _activation_from_name(activation)
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class CustomActorCritic(ActorCritic):
    """Example Actor-Critic with easy-to-edit actor/critic definitions."""
 
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, List[str]],
        num_actions: int,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        activation: str,
        init_noise_std: float,
        state_dependent_std: bool = False,
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )
        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
        # Replace these two lines with your own nn.Module definitions as needed.
        self.actor = _build_mlp(num_actor_obs, actor_out_dim, actor_hidden_dims, activation)
        self.critic = _build_mlp(num_critic_obs, 1, critic_hidden_dims, activation)
 
class CustomActorCritic_CNN(ActorCriticCNN):
    """
    Custom Actor-Critic for CNNs.
    It lets the base class handle the complex CNN creation (from config),
    but allows you to customize the MLP heads using your own _build_mlp function.
    """
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, List[str]],
        num_actions: int,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        activation: str,
        init_noise_std: float,
        state_dependent_std: bool = False,
        **kwargs,
    ):
        # 1. 调用父类初始化
        # 父类会解析 actor_cnn_cfg/critic_cnn_cfg 并创建 self.actor_cnns 和 self.critic_cnns
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

        # 2. 重新计算 Actor MLP 的输入维度
        # Part A: 1D 向量输入的维度 (来自 "policy" 组)
        num_actor_obs_1d = sum(obs[group].shape[-1] for group in obs_groups.get("policy", []))
        
        # Part B: 2D 图像经过 CNN 后的特征维度 (来自 self.actor_cnns)
        # 父类已经把 CNN 建好了，我们可以直接问它输出是多少
        num_actor_obs_2d_encoded = 0
        if self.actor_cnns is not None:
            for cnn in self.actor_cnns.values():
                num_actor_obs_2d_encoded += cnn.output_dim
        
        # total_actor_input_dim = num_actor_obs_1d + num_actor_obs_2d_encoded
        total_actor_input_dim = num_actor_obs_2d_encoded
        

        # 3. 重新计算 Critic MLP 的输入维度 (逻辑同上)
        num_critic_obs_1d = sum(obs[group].shape[-1] for group in obs_groups.get("critic", []))
        
        num_critic_obs_2d_encoded = 0
        if self.critic_cnns is not None:
            for cnn in self.critic_cnns.values():
                num_critic_obs_2d_encoded += cnn.output_dim
        
        # total_critic_input_dim = num_critic_obs_1d + num_critic_obs_2d_encoded
        total_critic_input_dim = num_critic_obs_2d_encoded

        # 4. 自定义 MLP 头部 (Overwrite)
        # 这里使用你自己的 _build_mlp 函数，替换掉父类默认创建的 self.actor 和 self.critic
        actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
        
        self.actor = _build_mlp(total_actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
        self.critic = _build_mlp(total_critic_input_dim, 1, critic_hidden_dims, activation)
        
        print(f"CustomActorCriticCNN Initialized.")
        print(f"  Actor Input: {num_actor_obs_1d} (Vec) + {num_actor_obs_2d_encoded} (Img) = {total_actor_input_dim}")
        print(f"  Critic Input: {num_critic_obs_1d} (Vec) + {num_critic_obs_2d_encoded} (Img) = {total_critic_input_dim}")


    
    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        assert self.actor_cnns is not None
        # Encode the 2D actor observations
        cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
        cnn_enc = torch.cat(cnn_enc_list, dim=-1)
        # Concatenate to the MLP observations
        obs = torch.cat([cnn_enc], dim=-1)

        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            # print(mean_and_std)
            # assert False, "1"
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            # print(obs.shape) # (8, 1024)
            mean = self.actor(obs)

            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)
   
    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = None
        obs_dict_2d = {}
        for obs_group in self.actor_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return None, obs_dict_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = None
        obs_dict_2d = {}
        for obs_group in self.critic_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return None, obs_dict_2d
    
    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        # mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # Encode the 2D critic observations
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([cnn_enc], dim=-1)

        return self.critic(mlp_obs)
    

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)


        if self.actor_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([cnn_enc], dim=-1)

        if self.state_dependent_std:
            return self.actor(mlp_obs)[..., 0, :]
        else:
            return self.actor(mlp_obs)




@dataclass
class EnvConfig:
    env_id: str = "Hopper-v4"
    num_envs: int = 8
    seed: int = 1
 
 
@dataclass
class PolicyConfig:
    class_name: str = "CustomActorCritic" # 这里加 CCN 会报错，估计 Custom 本身就是单独的基类
    # init_noise_std: float = 1.0
    init_noise_std: float = 0.2
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
 
 
@dataclass
class AlgorithmConfig:
    class_name: str = "PPO"
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    num_learning_epochs: int = 10
    num_mini_batches: int = 32
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 0.5
 
 
@dataclass
class RunnerConfig:
    num_steps_per_env: int = 256
    max_iterations: int = 1000
    save_interval: int = 50
 
 
@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    log_dir: str = "runs/rsl_rl_hopper"
    device: str = "auto"
 
 
def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
 

def _build_train_cfg(cfg: TrainConfig) -> dict:
    return {
        "seed": cfg.env.seed,
        "obs_groups": {
            "policy": ["obs"],
            "critic": ["obs"],
        },
        "policy": {
            "class_name": cfg.policy.class_name,
            "init_noise_std": cfg.policy.init_noise_std,
            "actor_hidden_dims": cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": cfg.policy.critic_hidden_dims,
            "activation": cfg.policy.activation,

            "actor_cnn_cfg": {
                "obs": {
                    'kernel_size': [8, 4, 3],
                    'output_channels': [32, 64, 64],
                    'stride': [4, 2, 1],
                }
            },
            "critic_cnn_cfg": {
                "obs": {
                    'kernel_size': [8, 4, 3],
                    'output_channels': [32, 64, 64],
                    'stride': [4, 2, 1],
                }
            }
        },
        "algorithm": {
            "class_name": cfg.algorithm.class_name,
            "value_loss_coef": cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": cfg.algorithm.use_clipped_value_loss,
            "clip_param": cfg.algorithm.clip_param,
            "entropy_coef": cfg.algorithm.entropy_coef,
            "num_learning_epochs": cfg.algorithm.num_learning_epochs,
            "num_mini_batches": cfg.algorithm.num_mini_batches,
            "learning_rate": cfg.algorithm.learning_rate,
            "schedule": cfg.algorithm.schedule,
            "gamma": cfg.algorithm.gamma,
            "lam": cfg.algorithm.lam,
            "desired_kl": cfg.algorithm.desired_kl,
            "max_grad_norm": cfg.algorithm.max_grad_norm,
            "rnd_cfg": None,
        },
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "max_iterations": cfg.runner.max_iterations,
    }
 
 
def main(cfg: TrainConfig):
    device = _resolve_device(cfg.device)
    random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
 
    env = GymnasiumVecEnv(cfg.env.env_id, cfg.env.num_envs, cfg.env.seed, device)
    train_cfg = _build_train_cfg(cfg)
 
    # Make CustomActorCritic discoverable by OnPolicyRunner.
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner
 
    rsl_modules.CustomActorCritic = CustomActorCritic_CNN
    rsl_runner.CustomActorCritic = CustomActorCritic_CNN
 
    runner = OnPolicyRunner(
        env, train_cfg, log_dir=f"{cfg.log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}", device=device
    )
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)
 
 
if __name__ == "__main__":
    main(tyro.cli(TrainConfig))