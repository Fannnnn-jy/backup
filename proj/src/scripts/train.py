from __future__ import annotations

import json
import os

os.environ.setdefault(
    "WANDB_API_KEY",
    "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",
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time

import torch
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.active_mapping_env import ActiveMappingConfig, ActiveMappingDiagnosticsConfig
from learning.callbacks import build_callbacks
from learning.config import PointCloudEvalConfig, TrainConfig, build_train_cfg
from learning.policy import register_custom_models
from learning.utils import GymnasiumVecEnv
from utils.rendering import init_render_state, render_depth_and_topdown

try:
    import tyro
except ImportError:  # pragma: no cover
    tyro = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_env_cfg(cfg: TrainConfig) -> ActiveMappingConfig:
    policy_name = cfg.policy.class_name
    policies_need_xyz_quat = {"ActorCriticRGBCNNMaxpooling_3D"}
    policies_need_xyz_delta_quat = {"ActorCriticRGBCNNRecurrent", "VGGTRelativeActorCritic"}
    action_mode = cfg.env.action_mode
    if policy_name in policies_need_xyz_quat and action_mode != "xyz_quat":
        raise ValueError(f"{policy_name} requires action_mode=xyz_quat.")
    if policy_name in policies_need_xyz_delta_quat and action_mode != "xyz_delta_quat":
        raise ValueError(f"{policy_name} requires action_mode=xyz_delta_quat.")
    if action_mode == "xyz_quat" and policy_name not in policies_need_xyz_quat:
        raise ValueError(f"action_mode=xyz_quat is only supported by {sorted(policies_need_xyz_quat)}.")
    if action_mode == "xyz_delta_quat" and policy_name not in policies_need_xyz_delta_quat:
        raise ValueError(
            f"action_mode=xyz_delta_quat is only supported by {sorted(policies_need_xyz_delta_quat)}."
        )
    env_fields = asdict(cfg.env)
    # if is_policy_3d and is_action_3d:
    #     env_fields["reward_grid_3d_enabled"] = True
    #     env_fields["reward_grid_3d_weight"] = 1.0
    diagnostics_cfg = env_fields.get("diagnostics")
    if isinstance(diagnostics_cfg, dict):
        env_fields["diagnostics"] = ActiveMappingDiagnosticsConfig(**diagnostics_cfg)
    allowed = ActiveMappingConfig.__dataclass_fields__.keys()
    env_fields = {key: value for key, value in env_fields.items() if key in allowed}
    return ActiveMappingConfig(**env_fields)


class InstrumentedOnPolicyRunner:
    def __init__(
        self,
        runner,
        *,
        log_images: bool,
        log_every: int,
        pointcloud_eval_cfg: PointCloudEvalConfig | None = None,
    ):
        self._runner = runner
        self._log_images = log_images and log_every > 0
        self._log_every = max(1, int(log_every))
        self._render_state = None
        self._wandb = None
        self._pending_episode_info = None
        self._trace_dump_dir = None
        self._dumped_trace_ids: set[tuple[int, int]] = set()
        if pointcloud_eval_cfg is None:
            pointcloud_eval_cfg = PointCloudEvalConfig()
        self._callbacks = build_callbacks(
            pointcloud_eval_cfg,
            num_envs=self._runner.env.num_envs,
            get_base_env=self._get_base_env,
        )
        if self._log_images:
            try:
                import wandb
            except ModuleNotFoundError:
                print("wandb is not installed; skipping image logging.")
            else:
                self._wandb = wandb

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def _select_env_obs(self, obs_td, env_index: int) -> dict:
        obs = {}
        for key, value in obs_td.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)
            if value.ndim > 0 and value.shape[0] == self._runner.env.num_envs:
                obs[key] = value[env_index]
            else:
                obs[key] = value
        return obs

    def _get_base_env(self, env_index: int):
        try:
            return self._runner.env.env.envs[env_index]
        except Exception:
            return None

    @staticmethod
    def _extract_env_value(extras, key: str, env_index: int):
        if not extras or key not in extras:
            return None
        value = extras[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape == ():
            return value.item()
        if value.shape[0] <= env_index:
            return None
        return value[env_index]

    def _resolve_end_reason(self, extras, env_index: int) -> str:
        if bool(self._extract_env_value(extras, "out_of_bounds", env_index) or False):
            return "out_of_bounds"
        if bool(self._extract_env_value(extras, "collision", env_index) or False):
            return "collision"
        if bool(self._extract_env_value(extras, "coverage_complete", env_index) or False):
            return "coverage_complete"
        if bool(self._extract_env_value(extras, "time_outs", env_index) or False):
            return "max_steps"
        return "terminated"

    def _update_episode_end_info(
        self, actions: torch.Tensor, dones: torch.Tensor, extras: dict, it: int, env_index: int = 0
    ) -> None:
        if not self._wandb:
            return
        if dones is None:
            return
        done_val = dones[env_index]
        if hasattr(done_val, "item"):
            done_val = bool(done_val.item())
        if not done_val:
            return
        reason = self._resolve_end_reason(extras, env_index)
        action_val = actions[env_index].detach().cpu().numpy()
        self._pending_episode_info = {
            "reason": reason,
            "action": action_val,
            "it": int(it),
        }

    def _log_render_images(self, it: int) -> None:
        if not self._wandb or not self._log_images:
            return
        if it % self._log_every != 0:
            return
        env_index = 0
        base_env = self._get_base_env(env_index)
        if base_env is None:
            return
        if self._render_state is None:
            self._render_state = init_render_state(base_env.cfg)
        obs_td = self._runner.env.get_observations()
        obs = self._select_env_obs(obs_td, env_index)
        depth_img, topdown_img, current_pos = render_depth_and_topdown(
            obs,
            base_env,
            state=self._render_state,
            depth_max=float(base_env.cfg.depth_data_max_value),
            topdown_scale=4,
            view="both",
            image_mode="rgb",
            first_camera_only=False,
            stats_interval=-1,
            step_idx=it,
        )
        if current_pos is not None:
            self._render_state.prev_position = current_pos
        payload = {}
        caption = None
        pending = self._pending_episode_info
        if pending is not None:
            reason = pending.get("reason", "terminated")
            action_val = pending.get("action")
            action_str = None
            if action_val is not None:
                action_str = np.array2string(np.asarray(action_val), precision=3, floatmode="fixed")
            if reason in ("collision", "out_of_bounds"):
                caption = f"end_reason={reason}, end_action={action_str}"
                payload["render/episode_end_action"] = action_val
            else:
                caption = f"end_reason={reason}"
            payload["render/episode_end_reason"] = reason
            payload["render/episode_end_it"] = pending.get("it", int(it))
            self._pending_episode_info = None
        if depth_img is not None:
            payload["render/rgb"] = self._wandb.Image(depth_img[..., ::-1])
        if topdown_img is not None:
            payload["render/topdown"] = self._wandb.Image(topdown_img[..., ::-1], caption=caption)
        if payload:
            self._wandb.log(payload, step=it)

    def _log_mean_reward(self, it: int, mean_reward: float) -> None:
        if not self._wandb:
            return
        self._wandb.log({"Train/mean_reward_rollout": mean_reward}, step=it)

    def _log_scalar(self, name: str, value: float, it: int) -> None:
        logger = getattr(self._runner, "logger", None)
        writer = getattr(logger, "writer", None) if logger is not None else None
        if writer is not None and not bool(getattr(logger, "disable_logs", False)):
            writer.add_scalar(name, float(value), it)
            return
        if self._wandb is not None:
            self._wandb.log({name: float(value)}, step=it)

    def _log_callback_scalars(self, it: int, scalars: dict[str, float]) -> None:
        if not scalars:
            return
        for key, value in scalars.items():
            self._log_scalar(key, float(value), it)

    def _ensure_trace_dump_dir(self):
        if self._trace_dump_dir is not None:
            return self._trace_dump_dir
        logger = getattr(self._runner, "logger", None)
        log_dir = getattr(logger, "log_dir", None) if logger is not None else None
        if not log_dir:
            return None
        trace_dir = Path(log_dir) / "diagnostic_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_dump_dir = trace_dir
        return trace_dir

    def _dump_recent_diagnostic_traces(self, it: int, dones: torch.Tensor | None) -> None:
        if dones is None:
            return
        trace_dir = self._ensure_trace_dump_dir()
        if trace_dir is None:
            return
        done_indices = (dones > 0).nonzero(as_tuple=False).view(-1).tolist()
        if not done_indices:
            return
        for env_index in done_indices:
            base_env = self._get_base_env(int(env_index))
            if base_env is None or not hasattr(base_env, "get_recent_diagnostic_traces"):
                continue
            try:
                traces = base_env.get_recent_diagnostic_traces()
            except Exception:
                continue
            for trace in traces:
                if not isinstance(trace, dict):
                    continue
                episode_id = int(trace.get("episode_id", -1))
                if episode_id < 0:
                    continue
                trace_key = (int(env_index), episode_id)
                if trace_key in self._dumped_trace_ids:
                    continue
                dump_path = trace_dir / f"env{int(env_index):02d}_episode{episode_id:06d}.json"
                payload = {
                    "iteration": int(it),
                    "env_index": int(env_index),
                    "episode_id": episode_id,
                    "trace_fields": [
                        "step_index",
                        "reward_total",
                        "reward_coverage_new",
                        "reward_visibility",
                        "reward_overlap",
                        "visible_ratio",
                        "new_visible_ratio",
                        "overlap_ratio",
                        "accumulated_coverage_ratio",
                        "is_collision",
                        "is_out_of_bounds",
                        "crash_reason",
                        "position_x",
                        "position_y",
                        "position_z",
                        "quat_w",
                        "quat_x",
                        "quat_y",
                        "quat_z",
                    ],
                    "trace": trace.get("steps", []),
                }
                dump_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
                self._dumped_trace_ids.add(trace_key)
                print(f"[diagnostics] trace saved: {dump_path}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Inline OnPolicyRunner.learn with an extra image logging hook.
        runner = self._runner
        if init_at_random_ep_len:
            runner.env.episode_length_buf = torch.randint_like(
                runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
            )

        obs = runner.env.get_observations().to(runner.device)
        runner.train_mode()

        if runner.is_distributed:
            print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
            runner.alg.broadcast_parameters()

        start_it = runner.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            rollout_reward_sum = 0.0
            rollout_reward_count = 0
            with torch.inference_mode():
                for step_idx in range(runner.cfg["num_steps_per_env"]):
                    if self._callbacks.enabled:
                        self._callbacks.on_step_begin(iteration=it, rollout_step=step_idx)
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
                    obs, rewards, dones = (
                        obs.to(runner.device),
                        rewards.to(runner.device),
                        dones.to(runner.device),
                    )
                    self._update_episode_end_info(actions, dones, extras, it)
                    self._dump_recent_diagnostic_traces(it, dones)
                    if self._callbacks.enabled:
                        callback_out = self._callbacks.on_step_end(
                            iteration=it,
                            rollout_step=step_idx,
                            rewards=rewards,
                            dones=dones,
                            extras=extras,
                        )
                        if callback_out.reward_delta is not None:
                            rewards = rewards + callback_out.reward_delta
                        self._log_callback_scalars(it, callback_out.scalars)
                    rollout_reward_sum += float(rewards.sum().item())
                    rollout_reward_count += int(rewards.numel())
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = runner.alg.intrinsic_rewards if runner.alg_cfg["rnd_cfg"] else None
                    runner.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                runner.alg.compute_returns(obs)

            loss_dict = runner.alg.update()

            stop = time.time()
            learn_time = stop - start
            runner.current_learning_iteration = it

            runner.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=runner.alg.learning_rate,
                action_std=runner.alg.policy.action_std,
                rnd_weight=runner.alg.rnd.weight if runner.alg_cfg["rnd_cfg"] else None,
            )
            if rollout_reward_count > 0:
                self._log_mean_reward(it, rollout_reward_sum / float(rollout_reward_count))
            self._log_render_images(it)

            if it % runner.cfg["save_interval"] == 0:
                runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if runner.logger.log_dir is not None and not runner.logger.disable_logs:
            runner.save(os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt"))


def main(cfg: TrainConfig) -> None:
    device = _resolve_device(cfg.device)
    env_cfg = _build_env_cfg(cfg)

    env = GymnasiumVecEnv(env_cfg, cfg.env.num_envs, cfg.env.seed, device)
    train_cfg = build_train_cfg(cfg)

    register_custom_models()

    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    if cfg.logger.lower() == "wandb":
        if not cfg.wandb_project:
            raise SystemExit("wandb_project must be set when logger=wandb.")
        try:
            import wandb  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit("wandb is not installed. Install it via `pip install wandb`.") from exc

    log_dir = f"{cfg.log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    wrap_runner = (
        cfg.logger.lower() == "wandb"
        or cfg.pointcloud_eval.enabled
        or (cfg.env.diagnostics.enabled and cfg.env.diagnostics.trace_enabled)
    )
    if wrap_runner:
        runner = InstrumentedOnPolicyRunner(
            runner,
            log_images=cfg.wandb_log_images if cfg.logger.lower() == "wandb" else False,
            log_every=cfg.wandb_log_images_every,
            pointcloud_eval_cfg=cfg.pointcloud_eval,
        )
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    if tyro is None:
        raise SystemExit("tyro is required for CLI parsing. Install it via `pip install tyro`.")
    main(tyro.cli(TrainConfig))
