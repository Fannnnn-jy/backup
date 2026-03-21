from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EnvConfig:
    task: str = "active_mapping"
    num_envs: int = 8
    seed: int = 1
    grid_size: Tuple[int, int] = (32, 32)
    world_size: Tuple[float, float] = (10.0, 10.0)
    world_size_z: float = 3.0
    max_steps: int = 1000
    grid_reward: float = 1.0
    reward_grid_size: Tuple[int, int] = (16, 16)
    step_reward: float = -0.05
    crash_penalty: float = -5.0
    action_scale: float = 0.1
    action_bound: float = 10.0
    xy_only: bool = True
    reward_traverse: bool = True
    action_mode: str = "xy_cos_sin"
    single_camera_obs: bool = True
    maniskill_camera_angles: Tuple[float, ...] = (0.0,)
    reward_by: str = "location"
    max_depth_frames: int = 4
    depth_image_shape: Tuple[int, int] = (64, 64)
    depth_data_dir: str = ""
    depth_data_pattern: str = "*.tiff"
    depth_data_max_value: float = 18.0
    depth_data_shuffle: bool = True
    sim_backend: str = "maniskill"
    gleam_data_dir: str = "data/eval_128"
    gleam_dataset_name: str = ""
    gleam_grid_size: int = 128
    gleam_scene_index: int = -1
    gleam_resample_scene: bool = True
    gleam_use_init_pose: bool = True
    maniskill_time_step: float = 1.0 / 60.0
    maniskill_camera_height: float = 1.5
    maniskill_camera_fov: float = 90.0
    maniskill_camera_near: float = 0.1
    maniskill_camera_far: float = 18.0
    maniskill_depth_buffer: str = ""
    agent_init_height: float = 1.5
    maniskill_camera_angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    maniskill_camera_yaw_range: Tuple[float, float] = (0.0, 360.0)
    maniskill_camera_yaw_init: float = 0.0
    maniskill_camera_yaw_delta: float = 30.0
    maniskill_urdf_dir: str = ""
    maniskill_render_device: str = ""


@dataclass
class PolicyConfig:
    class_name: str = "ActorCriticCNNMaxpooling"
    init_noise_std: float = 0.2
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False


@dataclass
class AlgorithmConfig:
    class_name: str = "PPO"
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.1
    num_learning_epochs: int = 10
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class RunnerConfig:
    num_steps_per_env: int = 128
    max_iterations: int = 20000
    save_interval: int = 100


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    log_dir: str = "runs/active_mapping"
    logger: str = "tensorboard"
    wandb_project: str = ""
    wandb_log_images: bool = True
    wandb_log_images_every: int = 200
    device: str = "auto"


def build_train_cfg(cfg: TrainConfig) -> dict:
    return {
        "seed": cfg.env.seed,
        "logger": cfg.logger,
        "wandb_project": cfg.wandb_project,
        "obs_groups": {
            "policy": ["obs_depth_tensor", "obs_pose"],
            "critic": ["obs_depth_tensor", "obs_pose"],
        },
        "policy": {
            "class_name": cfg.policy.class_name,
            "init_noise_std": cfg.policy.init_noise_std,
            "actor_hidden_dims": cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": cfg.policy.critic_hidden_dims,
            "activation": cfg.policy.activation,
            "actor_obs_normalization": cfg.policy.actor_obs_normalization,
            "critic_obs_normalization": cfg.policy.critic_obs_normalization,
            "actor_cnn_cfg": {
                "obs_depth_tensor": {
                    "kernel_size": [8, 4, 3],
                    "output_channels": [32, 64, 64],
                    "stride": [4, 2, 1],
                }
            },
            "critic_cnn_cfg": {
                "obs_depth_tensor": {
                    "kernel_size": [8, 4, 3],
                    "output_channels": [32, 64, 64],
                    "stride": [4, 2, 1],
                }
            },
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
