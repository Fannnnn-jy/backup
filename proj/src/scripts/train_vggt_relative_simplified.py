"""
Simplified VGGT relative-pose baseline training entry.

This script reuses the existing training pipeline and only swaps in the new
VGGTRelativeActorCritic policy:
- layers [4, 11, 17, 23]
- patch mean + max pooling
- relative pose only
- current + mean(history) + max(history)
- relative action output
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import tyro
except ImportError:  # pragma: no cover
    tyro = None

from learning.config import TrainConfig
from scripts.train import main as base_train_main


def main(cfg: TrainConfig) -> None:
    # Dedicated entry point for the simplified VGGT relative baseline.
    cfg.policy.class_name = "VGGTRelativeActorCritic"
    base_train_main(cfg)


if __name__ == "__main__":
    if tyro is None:
        raise SystemExit("tyro is required for CLI parsing. Install it via `pip install tyro`.")
    main(tyro.cli(TrainConfig))
