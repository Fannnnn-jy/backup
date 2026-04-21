import numpy as np

from envs.active_mapping_env import ActiveMappingConfig, ActiveMappingEnv


def test_patch_reward_once():
    cfg = ActiveMappingConfig(grid_size=(2, 2), world_size=(2.0, 2.0), max_steps=10)
    env = ActiveMappingEnv(cfg)
    env.reset(seed=0)

    _, reward, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
    assert reward == cfg.grid_reward
    assert terminated is False
    assert truncated is False

    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert reward == 0.0


def test_crash_penalty():
    cfg = ActiveMappingConfig(grid_size=(2, 2), world_size=(1.0, 1.0), max_steps=10)
    env = ActiveMappingEnv(cfg)
    env.reset(seed=0)

    _, reward, terminated, _, info = env.step(np.array([10.0, 0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert info["crash"] is True
    assert reward == cfg.crash_penalty
