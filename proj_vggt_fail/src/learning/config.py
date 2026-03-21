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
    grid_reward: float = 0.0  # 二维占用奖励权重
    reward_grid_size: Tuple[int, int] = (16, 16)
    reward_grid_3d_enabled: bool = True
    reward_grid_3d_size: int = 16
    reward_grid_3d_weight: float = 1.0  # 三维占用奖励权重
    reward_grid_3d_traverse: bool = True
    step_reward: float = -0.05
    crash_penalty: float = -5.0
    action_scale: float = 0.1
    action_bound: float = 3.0
    xy_only: bool = False
    reward_traverse: bool = True
    action_mode: str = "xyz_delta_quat"  # "xy_cos_sin", "xyz_quat", "xyz_delta_quat"
    single_camera_obs: bool = True
    maniskill_camera_angles: Tuple[float, ...] = (0.0,)
    reward_by: str = "location"
    max_depth_frames: int = 4
    depth_image_shape: Tuple[int, int] = (64, 64)
    depth_data_dir: str = ""
    depth_data_pattern: str = "*.tiff"
    depth_data_max_value: float = 5.0
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
    maniskill_camera_far: float = 5.0
    maniskill_depth_buffer: str = ""
    agent_init_height: float = 0.5
    maniskill_camera_yaw_range: Tuple[float, float] = (0.0, 360.0)
    maniskill_camera_yaw_init: float = 0.0
    maniskill_camera_yaw_delta: float = 30.0
    lock_camera_pitch_roll: bool = False
    maniskill_urdf_dir: str = ""
    maniskill_render_device: str = ""
    glb_data_dir: str = ""
    glb_scene_index: int = -1
    glb_scene_glob: str = "*.glb"
    glb_y_up: bool = True
    glb_scale: float = 1.0
    glb_voxel_cache_suffix: str = ".voxels.h5"


@dataclass
class PolicyConfig:
    class_name: str = "ActorCriticRGBCNNPoseDelta"
    init_noise_std: float = 0.2
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    image_obs_key: str = "obs_rgb_tensor"
    frame_pose_hidden_dim: int = 64
    frame_joint_hidden_dim: int = 128
    robot_pose_hidden_dim: int = 128
    vggt_model_id: str = "facebook/VGGT-1B"
    vggt_model_cache_dir: str = ""
    vggt_model_local_files_only: bool = True
    vggt_input_size: int = 518
    vggt_embed_dim: int = 1024
    vggt_feature_layers: Tuple[int, ...] = (4, 11, 17, 23)
    vggt_frame_hidden_dim: int = 256
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


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
class PointCloudEvalConfig:
    enabled: bool = False
    model_id: str = "facebook/VGGT-1B"
    model_cache_dir: str = ""
    model_local_files_only: bool = True
    device: str = "auto"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    vggt_input_size: int = 518
    frame_stride: int = 1
    max_frames: int = 0
    min_frames: int = 3
    max_points: int = 20000
    max_buffer_frames: int = 0
    align_mode: str = "similarity"
    distance_mode: str = "chamfer"
    forward_axis: str = "x+"
    depth_is_z: bool = True
    depth_scale: float = 1.0
    depth_max: float = 0.0
    reward_enabled: bool = False
    reward_weight: float = 0.0
    reward_clip: float = 0.0
    reward_nan_fallback: float = 0.0
    debug_dump_first_episode: bool = False
    debug_dump_dir: str = "debug/pointcloud_eval"
    debug_print_buffers_once: bool = False


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    pointcloud_eval: PointCloudEvalConfig = field(default_factory=PointCloudEvalConfig)
    log_dir: str = "runs/active_mapping"
    logger: str = "wandb"
    wandb_project: str = "gleam_baseline"
    wandb_log_images: bool = True
    wandb_log_images_every: int = 5
    device: str = "auto"


def build_train_cfg(cfg: TrainConfig) -> dict:
    image_key = cfg.policy.image_obs_key
    obs_keys = [image_key, "agent_state", "obs_pose"]
    if cfg.policy.class_name == "ActorCriticRGBCNNPoseDelta":
        obs_keys = [image_key, "obs_cam_pose", "agent_state", "obs_pose"]
    if cfg.policy.class_name == "ActorCriticRGBVGGTFrozenPoseDelta":
        obs_keys = [image_key, "obs_cam_pose", "agent_state", "obs_pose"]
    policy_cfg = {
        "class_name": cfg.policy.class_name,
        "init_noise_std": cfg.policy.init_noise_std,
        "actor_hidden_dims": cfg.policy.actor_hidden_dims,
        "critic_hidden_dims": cfg.policy.critic_hidden_dims,
        "activation": cfg.policy.activation,
        "actor_obs_normalization": cfg.policy.actor_obs_normalization,
        "critic_obs_normalization": cfg.policy.critic_obs_normalization,
        "actor_cnn_cfg": {
            image_key: {
                "kernel_size": [8, 4, 3],
                "output_channels": [32, 64, 64],
                "stride": [4, 2, 1],
            }
        },
        "critic_cnn_cfg": {
            image_key: {
                "kernel_size": [8, 4, 3],
                "output_channels": [32, 64, 64],
                "stride": [4, 2, 1],
            }
        },
    }
    if cfg.policy.class_name == "ActorCriticRGBCNNRecurrent":
        policy_cfg.update(
            {
                "rnn_type": cfg.policy.rnn_type,
                "rnn_hidden_dim": cfg.policy.rnn_hidden_dim,
                "rnn_num_layers": cfg.policy.rnn_num_layers,
            }
        )
    if cfg.policy.class_name == "ActorCriticRGBCNNPoseDelta":
        policy_cfg.update(
            {
                "frame_pose_hidden_dim": cfg.policy.frame_pose_hidden_dim,
                "frame_joint_hidden_dim": cfg.policy.frame_joint_hidden_dim,
                "robot_pose_hidden_dim": cfg.policy.robot_pose_hidden_dim,
            }
        )
    if cfg.policy.class_name == "ActorCriticRGBVGGTFrozenPoseDelta":
        policy_cfg.update(
            {
                "frame_pose_hidden_dim": cfg.policy.frame_pose_hidden_dim,
                "frame_joint_hidden_dim": cfg.policy.frame_joint_hidden_dim,
                "robot_pose_hidden_dim": cfg.policy.robot_pose_hidden_dim,
                "vggt_model_id": cfg.policy.vggt_model_id,
                "vggt_model_cache_dir": cfg.policy.vggt_model_cache_dir,
                "vggt_model_local_files_only": cfg.policy.vggt_model_local_files_only,
                "vggt_input_size": cfg.policy.vggt_input_size,
                "vggt_embed_dim": cfg.policy.vggt_embed_dim,
                "vggt_feature_layers": list(cfg.policy.vggt_feature_layers),
                "vggt_frame_hidden_dim": cfg.policy.vggt_frame_hidden_dim,
            }
        )
    return {
        "seed": cfg.env.seed,
        "logger": cfg.logger,
        "wandb_project": cfg.wandb_project,
        "obs_groups": {
            "policy": obs_keys,
            "critic": obs_keys,
        },
        "policy": policy_cfg,
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
