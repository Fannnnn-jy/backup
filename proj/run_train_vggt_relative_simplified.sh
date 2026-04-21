#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=5

# Default paths can be overridden from the shell, e.g.
#   GLB_DIR=/path/to/glbs MODEL_CACHE_DIR=/path/to/model_weights ./run_train_vggt_relative_simplified.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GLB_DIR="${GLB_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/cad_apt}"
GLB_SCENE_GLOB="${GLB_SCENE_GLOB:-*.glb}"
GLB_SCENE_INDEX="${GLB_SCENE_INDEX:--1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/home/ghr/fs/Junyi/data/proj/model_weights}"
LOG_DIR="${LOG_DIR:-runs/vggt_relative_stage1}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt_relative_stage1}"
LOGGER="${LOGGER:-wandb}"
DEVICE="${DEVICE:-auto}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
ACTION_SCALE="${ACTION_SCALE:-0.5}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
REWARD_COVERAGE_DILATION="${REWARD_COVERAGE_DILATION:-1}"
CAMERA_ROTATION_LOCK="${CAMERA_ROTATION_LOCK:-yaw_pitch}"
INIT_X_MIN="${INIT_X_MIN:--0.5}"
INIT_X_MAX="${INIT_X_MAX:-0.5}"
INIT_Y_MIN="${INIT_Y_MIN:--0.5}"
INIT_Y_MAX="${INIT_Y_MAX:-0.5}"
INIT_Z_MIN="${INIT_Z_MIN:-0.3}"
INIT_Z_MAX="${INIT_Z_MAX:-0.8}"

if [[ ! -d "$GLB_DIR" ]]; then
    echo "[run_train_vggt_relative_simplified] GLB_DIR does not exist: $GLB_DIR" >&2
    exit 1
fi

if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    echo "[run_train_vggt_relative_simplified] MODEL_CACHE_DIR does not exist: $MODEL_CACHE_DIR" >&2
    exit 1
fi

ROTATION_LOCK_ARGS=()
INIT_POSITION_RANGE_ARGS=()
case "$CAMERA_ROTATION_LOCK" in
    none|"")
        ;;
    yaw_pitch)
        ROTATION_LOCK_ARGS+=(--env.lock-camera-roll)
        ;;
    yaw)
        ROTATION_LOCK_ARGS+=(--env.lock-camera-pitch-roll)
        ;;
    *)
        echo "[run_train_vggt_relative_simplified] CAMERA_ROTATION_LOCK must be one of: none, yaw_pitch, yaw" >&2
        exit 1
        ;;
esac

if [[ -n "$INIT_X_MIN" || -n "$INIT_X_MAX" ]]; then
    if [[ -z "$INIT_X_MIN" || -z "$INIT_X_MAX" ]]; then
        echo "[run_train_vggt_relative_simplified] INIT_X_MIN and INIT_X_MAX must be set together" >&2
        exit 1
    fi
    INIT_POSITION_RANGE_ARGS+=(--env.init-position-x-range "$INIT_X_MIN" "$INIT_X_MAX")
fi

if [[ -n "$INIT_Y_MIN" || -n "$INIT_Y_MAX" ]]; then
    if [[ -z "$INIT_Y_MIN" || -z "$INIT_Y_MAX" ]]; then
        echo "[run_train_vggt_relative_simplified] INIT_Y_MIN and INIT_Y_MAX must be set together" >&2
        exit 1
    fi
    INIT_POSITION_RANGE_ARGS+=(--env.init-position-y-range "$INIT_Y_MIN" "$INIT_Y_MAX")
fi

if [[ -n "$INIT_Z_MIN" || -n "$INIT_Z_MAX" ]]; then
    if [[ -z "$INIT_Z_MIN" || -z "$INIT_Z_MAX" ]]; then
        echo "[run_train_vggt_relative_simplified] INIT_Z_MIN and INIT_Z_MAX must be set together" >&2
        exit 1
    fi
    INIT_POSITION_RANGE_ARGS+=(--env.init-position-z-range "$INIT_Z_MIN" "$INIT_Z_MAX")
fi

export CUDA_VISIBLE_DEVICES=7

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
    --env.action-mode xyz_delta_quat \
    --env.max-steps "$MAX_STEPS" \
    --env.action-scale "$ACTION_SCALE" \
    --env.depth-image-shape "$IMAGE_SIZE" "$IMAGE_SIZE" \
    --env.no-reward-grid-3d-enabled \
    --env.reward-grid-3d-weight 0.0 \
    --env.step-reward 0.0 \
    --env.crash-penalty -2.0 \
    --env.reward-w-new 10.0 \
    --env.reward-w-visible 3.0 \
    --env.reward-w-overlap 1.0 \
    --env.reward-overlap-low 0.30 \
    --env.reward-overlap-high 0.50 \
    --env.reward-eps 1e-6 \
    --env.reward-coverage-dilation "$REWARD_COVERAGE_DILATION" \
    --env.reward-skip-first-frame \
    --policy.init-noise-std 0.3 \
    --env.diagnostics.enabled \
    --env.diagnostics.trace_enabled \
    --env.diagnostics.histogram_enabled \
    --policy.vggt-relative.model-id facebook/VGGT-1B \
    --policy.vggt-relative.model-cache-dir "$MODEL_CACHE_DIR" \
    --policy.vggt-relative.model-local-files-only \
    --policy.vggt-relative.selected-layers 4 11 17 23 \
    --policy.vggt-relative.visual-dim 256 \
    --policy.vggt-relative.pose-dim 64 \
    --policy.vggt-relative.fused-dim 256 \
    --runner.max-iterations "$MAX_ITERATIONS" \
    --runner.save-interval "$SAVE_INTERVAL" \
    "${ROTATION_LOCK_ARGS[@]}" \
    "${INIT_POSITION_RANGE_ARGS[@]}" \
    "$@"

# Example overrides:
# REWARD_COVERAGE_DILATION=1 CAMERA_ROTATION_LOCK=yaw_pitch ./run_train_vggt_relative_simplified.sh
# CAMERA_ROTATION_LOCK=yaw ./run_train_vggt_relative_simplified.sh
