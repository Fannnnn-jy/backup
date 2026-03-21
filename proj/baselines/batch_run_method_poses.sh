#!/usr/bin/env bash
set -euo pipefail

# Batch baseline pose generation driven by sampled XYZ + yaw_deg tuples.
#
# The generated ActiveGS init pose keeps roll=0 and pitch=0, and only varies
# yaw around world +Z.
#
# What is implemented now:
# - active-gs + GT depth: runnable
#
# What is intentionally left commented because the code paths are not wired yet:
# - gleam + external sampled init poses
# - gleam depth-mode splits
#
# Usage:
#   bash baselines/batch_run_method_poses.sh
#   bash baselines/batch_run_method_poses.sh replicacad/apt_0 procthor/ProcTHOR-Test-21

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTIVE_GS_ROOT="${ACTIVE_GS_ROOT:-/home/ghr/fs/Junyi/active-gs}"
DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"

# Override these if different environments/interpreters are needed.
PROJ_PYTHON="${PROJ_PYTHON:-python}"
ACTIVE_GS_PYTHON="${ACTIVE_GS_PYTHON:-python}"
ACTIVE_GS_CONDA_SH="${ACTIVE_GS_CONDA_SH:-/home/ghr/anaconda3/etc/profile.d/conda.sh}"
ACTIVE_GS_CONDA_ENV="${ACTIVE_GS_CONDA_ENV:-active-gs}"

ACTIVE_GS_PYGLFW_LIBRARY="${ACTIVE_GS_PYGLFW_LIBRARY:-/usr/lib/x86_64-linux-gnu/libglfw.so.3}"
ACTIVE_GS_EGL_VENDOR_JSON="${ACTIVE_GS_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
ACTIVE_GS_SYSTEM_LIB_DIR="${ACTIVE_GS_SYSTEM_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
ACTIVE_GS_LD_PRELOAD_LIB="${ACTIVE_GS_LD_PRELOAD_LIB:-/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0}"

SAMPLE_DIR="${SCRIPT_DIR}/sampled_inits"
mkdir -p "${SAMPLE_DIR}"

N_SAMPLES="${N_SAMPLES:-3}"
MAX_FRAMES="${MAX_FRAMES:-20}"
ACTIVE_GS_EXP_ID="${ACTIVE_GS_EXP_ID:-batch_xyz_yaw}"
ACTIVE_GS_GT_METHOD_NAME="${ACTIVE_GS_GT_METHOD_NAME:-active-gs-gt-depth}"

# Global sampling box used for every scene.
DEFAULT_X_MIN="${DEFAULT_X_MIN:--0.5}"
DEFAULT_X_MAX="${DEFAULT_X_MAX:-0.5}"
DEFAULT_Y_MIN="${DEFAULT_Y_MIN:--0.5}"
DEFAULT_Y_MAX="${DEFAULT_Y_MAX:-0.5}"
DEFAULT_Z_MIN="${DEFAULT_Z_MIN:-0.3}"
DEFAULT_Z_MAX="${DEFAULT_Z_MAX:-0.8}"
DEFAULT_XY_RADIUS="${DEFAULT_XY_RADIUS:-6}"
DEFAULT_Z_RADIUS="${DEFAULT_Z_RADIUS:-3}"
DEFAULT_BORDER_VOXELS="${DEFAULT_BORDER_VOXELS:-4}"
DEFAULT_YAW_MIN_DEG="${DEFAULT_YAW_MIN_DEG:-0}"
DEFAULT_YAW_MAX_DEG="${DEFAULT_YAW_MAX_DEG:-360}"
DEFAULT_SEED="${DEFAULT_SEED:-0}"
DEFAULT_SAFETY_MARGIN="${DEFAULT_SAFETY_MARGIN:-[0.0,0.0,0.0]}"
DEFAULT_ROBOT_SIZE="${DEFAULT_ROBOT_SIZE:-0.01}"

if (($# > 0)); then
  SCENES=("$@")
else
  SCENES=("replicacad/apt_0")
fi

scene_safe() {
  local scene_id="$1"
  printf '%s' "${scene_id//\//__}"
}

for scene_id in "${SCENES[@]}"; do
  safe_scene="$(scene_safe "${scene_id}")"
  glb_path="${DATA_DIR}/${scene_id}.glb"
  sample_tsv="${SAMPLE_DIR}/${safe_scene}_xyz_yaw.tsv"

  if [[ ! -f "${glb_path}" ]]; then
    echo "[skip] scene not found: ${glb_path}" >&2
    continue
  fi

  echo "==== Sampling init poses for ${scene_id} ===="
  (
    cd "${PROJ_ROOT}"
    "${PROJ_PYTHON}" baselines/sample_xyz_pitch.py \
      --scene_id "${scene_id}" \
      --data_dir "${DATA_DIR}" \
      --n_samples "${N_SAMPLES}" \
      --seed "${DEFAULT_SEED}" \
      --x_min "${DEFAULT_X_MIN}" \
      --x_max "${DEFAULT_X_MAX}" \
      --y_min "${DEFAULT_Y_MIN}" \
      --y_max "${DEFAULT_Y_MAX}" \
      --z_min "${DEFAULT_Z_MIN}" \
      --z_max "${DEFAULT_Z_MAX}" \
      --xy_radius "${DEFAULT_XY_RADIUS}" \
      --z_radius "${DEFAULT_Z_RADIUS}" \
      --border_voxels "${DEFAULT_BORDER_VOXELS}" \
      --yaw_min_deg "${DEFAULT_YAW_MIN_DEG}" \
      --yaw_max_deg "${DEFAULT_YAW_MAX_DEG}" \
      --output "${sample_tsv}"
  )

  while IFS=$'\t' read -r sample_id run_id x y z yaw_deg init_pose_json; do
    echo
    echo "==== Scene ${scene_id} | sample ${sample_id} | run_id ${run_id} ===="
    echo "     xyz=(${x}, ${y}, ${z}) yaw_deg=${yaw_deg}"

    echo "[run] active-gs + gt depth"
    (
      cd "${ACTIVE_GS_ROOT}"
      had_nounset=0
      if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
      fi
      source "${ACTIVE_GS_CONDA_SH}"
      conda activate "${ACTIVE_GS_CONDA_ENV}"
      if [[ "${had_nounset}" -eq 1 ]]; then
        set -u
      fi
      export PYGLFW_LIBRARY="${ACTIVE_GS_PYGLFW_LIBRARY}"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${ACTIVE_GS_EGL_VENDOR_JSON}"
      export LD_LIBRARY_PATH="${ACTIVE_GS_SYSTEM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      export LD_PRELOAD="${ACTIVE_GS_LD_PRELOAD_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
      "${ACTIVE_GS_PYTHON}" main.py \
          "scene_name=${scene_id}.glb" \
          "experiment.max_frames=${MAX_FRAMES}" \
          "experiment.exp_id=${ACTIVE_GS_EXP_ID}" \
          "experiment.run_id=${sample_id}" \
          "experiment.pose_method_name=${ACTIVE_GS_GT_METHOD_NAME}" \
          "planner.init_pose=${init_pose_json}" \
          "mapper.voxel_map.safety_margin=${DEFAULT_SAFETY_MARGIN}" \
          "planner.robot_size=${DEFAULT_ROBOT_SIZE}" \
          use_gui=false
    )

    echo "[run] active-gs + predicted depth"
    (
      cd "${ACTIVE_GS_ROOT}"
      had_nounset=0
      if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
      fi
      source "${ACTIVE_GS_CONDA_SH}"
      conda activate "${ACTIVE_GS_CONDA_ENV}"
      if [[ "${had_nounset}" -eq 1 ]]; then
        set -u
      fi
      export PYGLFW_LIBRARY="${ACTIVE_GS_PYGLFW_LIBRARY}"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${ACTIVE_GS_EGL_VENDOR_JSON}"
      export LD_LIBRARY_PATH="${ACTIVE_GS_SYSTEM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      export LD_PRELOAD="${ACTIVE_GS_LD_PRELOAD_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
      "${ACTIVE_GS_PYTHON}" main.py \
          "scene_name=${scene_id}.glb" \
          "experiment.max_frames=${MAX_FRAMES}" \
          "experiment.exp_id=${ACTIVE_GS_EXP_ID}" \
          "experiment.run_id=${sample_id}" \
          "experiment.pose_method_name=active-gs-pred-depth" \
          "planner.init_pose=${init_pose_json}" \
          "mapper.voxel_map.safety_margin=${DEFAULT_SAFETY_MARGIN}" \
          "planner.robot_size=${DEFAULT_ROBOT_SIZE}" \
          "mapper.depth_mode=pred" \
          use_gui=false
    )

    random_voxel_path="${DATA_DIR}/${scene_id}.glb.voxels.h5"
    if [[ -f "${random_voxel_path}" ]]; then
      random_seed=$((DEFAULT_SEED + sample_id))

      echo "[run] random"
      (
        cd "${PROJ_ROOT}"
        "${PROJ_PYTHON}" baselines/random_pose/generate_poses.py \
          --scene_id "${scene_id}" \
          --data_dir "${DATA_DIR}" \
          --n_views "${MAX_FRAMES}" \
          --run_id "${run_id}" \
          --seed "${random_seed}" \
          --z_min "${DEFAULT_Z_MIN}" \
          --z_max "${DEFAULT_Z_MAX}" \
          --xy_radius "${DEFAULT_XY_RADIUS}" \
          --z_radius "${DEFAULT_Z_RADIUS}" \
          --border_voxels "${DEFAULT_BORDER_VOXELS}" \
          --method "random" \
          --init_pose_json "${init_pose_json}"
      )
    else
      echo "[skip] random baselines require voxel grid: ${random_voxel_path}" >&2
    fi

    # [TODO] gleam + gt depth
    # baselines/gleam_poses/generate.py currently does not accept an external
    # init pose/yaw from the sampled xyz/yaw set.
    #
    # (
    #   cd "${PROJ_ROOT}"
    #   "${PROJ_PYTHON}" baselines/gleam_poses/generate.py \
    #     --glb "${glb_path}" \
    #     --scene_id "${scene_id}" \
    #     --run_id "${run_id}" \
    #     --method "gleam-gt-depth"
    # )

    # [TODO] gleam + predicted depth
    # Same init-pose blocker as above. Also clarify first whether the pred/gt
    # depth split should affect trajectory generation, rendering, or evaluation.

  done < <(tail -n +2 "${sample_tsv}")
done
