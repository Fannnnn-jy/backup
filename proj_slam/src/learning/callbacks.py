from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch

from learning.config import PointCloudEvalConfig


@dataclass
class CallbackOutput:
    reward_delta: Optional[torch.Tensor] = None
    scalars: Dict[str, float] = field(default_factory=dict)


class CallbackManager:
    def __init__(self, callbacks: Iterable[object] | None = None) -> None:
        self._callbacks: List[object] = list(callbacks or [])

    @property
    def enabled(self) -> bool:
        return len(self._callbacks) > 0

    def on_step_begin(self, *, iteration: int, rollout_step: int) -> None:
        for callback in self._callbacks:
            fn = getattr(callback, "on_step_begin", None)
            if fn is None:
                continue
            fn(iteration=iteration, rollout_step=rollout_step)

    def on_step_end(
        self,
        *,
        iteration: int,
        rollout_step: int,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> CallbackOutput:
        merged = CallbackOutput()
        for callback in self._callbacks:
            fn = getattr(callback, "on_step_end", None)
            if fn is None:
                continue
            out = fn(
                iteration=iteration,
                rollout_step=rollout_step,
                rewards=rewards,
                dones=dones,
                extras=extras,
            )
            if out is None:
                continue
            if out.reward_delta is not None:
                delta = out.reward_delta.to(device=rewards.device, dtype=rewards.dtype)
                if merged.reward_delta is None:
                    merged.reward_delta = torch.zeros_like(rewards)
                merged.reward_delta = merged.reward_delta + delta
            if out.scalars:
                for key, value in out.scalars.items():
                    merged.scalars[key] = float(value)
        return merged


def build_callbacks(
    pointcloud_cfg: PointCloudEvalConfig,
    *,
    num_envs: int,
    get_base_env: Callable[[int], object | None],
    shared_slam_model=None,
) -> CallbackManager:
    callbacks: List[object] = []
    if pointcloud_cfg.enabled:
        from utils.pointcloud_eval import PointCloudEpisodeCallback

        callbacks.append(
            PointCloudEpisodeCallback(
                pointcloud_cfg,
                num_envs=num_envs,
                get_base_env=get_base_env,
                shared_slam_model=shared_slam_model,
            )
        )
    return CallbackManager(callbacks)
