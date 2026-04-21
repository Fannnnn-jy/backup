import numpy as np
from envs.active_mapping_env import ActiveMappingConfig
from envs.maniskill_active_mapping_env import ManiSkillActiveMappingEnv

cfg = ActiveMappingConfig(
    sim_backend="maniskill",
    gleam_data_dir="data/eval_128",
    gleam_dataset_name="eval_128",
    maniskill_render_device="cuda:0",  # 用你实际设备别名替换
    max_depth_frames=4,
    depth_image_shape=(64, 64),
)
env = ManiSkillActiveMappingEnv(cfg)
obs, info = env.reset(seed=0)
print("obs_depth_tensor", obs["obs_depth_tensor"].shape, "coverage", info["coverage"])
obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
print("reward", reward, "terminated", terminated, "coverage", info["coverage"])