from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.active_mapping_env import ActiveMappingConfig
from learning.config import EnvConfig, TrainConfig, build_train_cfg
from learning.policy import register_custom_models
from learning.utils import GymnasiumVecEnv

try:
    import tyro
except ImportError:  # pragma: no cover
    tyro = None


@dataclass
class PlayConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    load_model: str = ""
    num_steps: int = 1000
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_env_cfg(cfg: PlayConfig) -> ActiveMappingConfig:
    env_fields = asdict(cfg.env)
    allowed = ActiveMappingConfig.__dataclass_fields__.keys()
    env_fields = {key: value for key, value in env_fields.items() if key in allowed}
    return ActiveMappingConfig(**env_fields)


def main(cfg: PlayConfig) -> None:
    device = _resolve_device(cfg.device)
    env_cfg = _build_env_cfg(cfg)

    env = GymnasiumVecEnv(env_cfg, cfg.env.num_envs, cfg.env.seed, device)
    train_cfg = build_train_cfg(TrainConfig(env=cfg.env, device=cfg.device))

    register_custom_models()

    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    runner = OnPolicyRunner(env, train_cfg, log_dir="", device=device)
    if cfg.load_model:
        runner.load(cfg.load_model)

    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    for _ in range(cfg.num_steps):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        if runner.alg.policy.is_recurrent:
            runner.alg.policy.reset(dones)
        if dones.any():
            obs = env.reset()


if __name__ == "__main__":
    if tyro is None:
        raise SystemExit("tyro is required for CLI parsing. Install it via `pip install tyro`.")
    main(tyro.cli(PlayConfig))
