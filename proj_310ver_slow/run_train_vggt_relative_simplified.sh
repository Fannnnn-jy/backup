#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Default paths can be overridden from the shell, e.g.
#   GLB_DIR=/path/to/glbs MODEL_CACHE_DIR=/path/to/model_weights ./run_train_vggt_relative_simplified.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GLB_DIR="${GLB_DIR:-/home/ghr/fs/Junyi/data/proj/cad}"
GLB_SCENE_GLOB="${GLB_SCENE_GLOB:-*.glb}"
GLB_SCENE_INDEX="${GLB_SCENE_INDEX:--1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/home/ghr/fs/Junyi/data/proj/model_weights}"
LOG_DIR="${LOG_DIR:-runs/vggt_relative_stage1}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt_relative_stage1}"
LOGGER="${LOGGER:-wandb}"
DEVICE="${DEVICE:-auto}"
NUM_ENVS="${NUM_ENVS:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
ACTION_SCALE="${ACTION_SCALE:-0.5}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

if [[ ! -d "$GLB_DIR" ]]; then
    echo "[run_train_vggt_relative_simplified] GLB_DIR does not exist: $GLB_DIR" >&2
    exit 1
fi

if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    echo "[run_train_vggt_relative_simplified] MODEL_CACHE_DIR does not exist: $MODEL_CACHE_DIR" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES

python src/scripts/train_vggt_relative_simplified.py \
    --log-dir "$LOG_DIR" \
    --logger "$LOGGER" \
    --wandb-project "$WANDB_PROJECT" \
    --device "$DEVICE" \
    --env.num-envs "$NUM_ENVS" \
    --env.sim-backend maniskill \
    --env.glb-data-dir "$GLB_DIR" \
    --env.glb-scene-glob "$GLB_SCENE_GLOB" \
    --env.glb-scene-index "$GLB_SCENE_INDEX" \
    --env.no-gleam-use-init-pose \
    --env.agent-init-height 0.5 \
    --env.maniskill-camera-height 0.5 \
    --env.maniskill-depth-buffer DepthLinear \
    --env.single-camera-obs \
    --env.maniskill-camera-angles 0 \
    --env.lock-camera-pitch-roll \
    --env.action-mode xyz_delta_quat \
    --env.max-steps "$MAX_STEPS" \
    --env.action-scale "$ACTION_SCALE" \
    --env.depth-image-shape "$IMAGE_SIZE" "$IMAGE_SIZE" \
    --env.no-reward-grid-3d-enabled \
    --env.reward-grid-3d-weight 0.0 \
    --env.step-reward -0.01 \
    --env.crash-penalty -1.0 \
    --env.reward-w-new 1.0 \
    --env.reward-w-visible 0.1 \
    --env.reward-w-overlap 0.2 \
    --env.reward-overlap-low 0.30 \
    --env.reward-overlap-high 0.50 \
    --env.reward-eps 1e-6 \
    --env.reward-skip-first-frame \
    --policy.init-noise-std 0.3 \
    --policy.vggt-relative.model-id facebook/VGGT-1B \
    --policy.vggt-relative.model-cache-dir "$MODEL_CACHE_DIR" \
    --policy.vggt-relative.model-local-files-only \
    --policy.vggt-relative.selected-layers 4 11 17 23 \
    --policy.vggt-relative.visual-dim 256 \
    --policy.vggt-relative.pose-dim 64 \
    --policy.vggt-relative.fused-dim 256 \
    --runner.max-iterations "$MAX_ITERATIONS" \
    --runner.save-interval "$SAVE_INTERVAL" \
    "$@"
