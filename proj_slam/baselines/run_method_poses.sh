#!/usr/bin/env bash
set -euo pipefail

# Batch baseline pose generation driven by sampled XYZ + yaw_deg tuples.
#
# The generated ActiveGS init pose keeps roll=0 and pitch=0, and only varies
# yaw around world +Z.
#
# What is implemented now:
# - active-gs + GT depth: runnable
# - gleam: runnable (custom eval env, sampled xyz/yaw TSV)
#
# What is intentionally left out:
# (nothing — both GT-depth and pred-depth variants are now supported)
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
GLEAM_ROOT="${GLEAM_ROOT:-/home/ghr/fs/Junyi/GLEAM}"
GLEAM_PYTHON="${GLEAM_PYTHON:-python}"
GLEAM_CONDA_SH="${GLEAM_CONDA_SH:-/home/ghr/anaconda3/etc/profile.d/conda.sh}"
GLEAM_CONDA_ENV="${GLEAM_CONDA_ENV:-gleam}"
GLEAM_CKPT_PATH="${GLEAM_CKPT_PATH:-/home/ghr/fs/Junyi/data/proj/model_weights/GLEAM/rl_model_40000000_steps.zip}"
GLEAM_GT_ROOT="${GLEAM_GT_ROOT:-data_gleam/gt}"
GLEAM_URDF_ROOT="${GLEAM_URDF_ROOT:-data_gleam/urdf}"
GLEAM_GT_METHOD_NAME="${GLEAM_GT_METHOD_NAME:-gleam-gt-depth}"
GLEAM_PRED_METHOD_NAME="${GLEAM_PRED_METHOD_NAME:-gleam-pred-depth}"
GLEAM_PRED_DEPTH_MODEL="${GLEAM_PRED_DEPTH_MODEL:-depth-anything/da3mono-large}"
GLEAM_PRED_DEPTH_CACHE_DIR="${GLEAM_PRED_DEPTH_CACHE_DIR:-/home/ghr/fs/Junyi/model_weights}"
GLEAM_SCENE_PREFIX_PREFIX="${GLEAM_SCENE_PREFIX_PREFIX:-gt_glb_}"
GLEAM_DRONE_HEIGHT="${GLEAM_DRONE_HEIGHT:-1.5}"
GLEAM_GRID_RESO="${GLEAM_GRID_RESO:-128}"
GLEAM_NUM_ENVS="${GLEAM_NUM_ENVS:-1}"
GLEAM_SIM_DEVICE="${GLEAM_SIM_DEVICE:-cuda:0}"
GLEAM_EVAL_DEVICE="${GLEAM_EVAL_DEVICE:-cuda:0}"
GLEAM_DEBUG_OBS="${GLEAM_DEBUG_OBS:-true}"
GLEAM_DEBUG_OBS_ENV_IDX="${GLEAM_DEBUG_OBS_ENV_IDX:-0}"
GLEAM_DEBUG_OBS_MAX_STEPS="${GLEAM_DEBUG_OBS_MAX_STEPS:-20}"
GLEAM_SAVE_PATH="${GLEAM_SAVE_PATH:-}"
GLEAM_WORK_DIR="${GLEAM_WORK_DIR:-/home/ghr/fs/Junyi/data/proj/baseline_data/tmp}"
GLEAM_INPUT_GLB_MODE="${GLEAM_INPUT_GLB_MODE:-raw}"
GLEAM_PREPROCESS_OVERWRITE="${GLEAM_PREPROCESS_OVERWRITE:-false}"
GLEAM_CC="${GLEAM_CC:-gcc-11}"
GLEAM_CXX="${GLEAM_CXX:-g++-11}"
GLEAM_CUDAHOSTCXX="${GLEAM_CUDAHOSTCXX:-g++-11}"

ACTIVE_GS_PYGLFW_LIBRARY="${ACTIVE_GS_PYGLFW_LIBRARY:-/usr/lib/x86_64-linux-gnu/libglfw.so.3}"
ACTIVE_GS_EGL_VENDOR_JSON="${ACTIVE_GS_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
ACTIVE_GS_SYSTEM_LIB_DIR="${ACTIVE_GS_SYSTEM_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
ACTIVE_GS_LD_PRELOAD_LIB="${ACTIVE_GS_LD_PRELOAD_LIB:-/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0}"

SAMPLE_DIR="${SCRIPT_DIR}/sampled_inits"
mkdir -p "${SAMPLE_DIR}"
BATCH_LOG_DIR="${SCRIPT_DIR}/tmp"
mkdir -p "${BATCH_LOG_DIR}"
BATCH_STATUS_LOG_RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_STATUS_LOG="${BATCH_STATUS_LOG:-${BATCH_LOG_DIR}/run_method_poses_status_${BATCH_STATUS_LOG_RUN_TS}.tsv}"
if [[ ! -f "${BATCH_STATUS_LOG}" ]]; then
  printf 'timestamp_utc\tscene_id\tsample_id\trun_id\tmethod\tstatus\tdetail\n' > "${BATCH_STATUS_LOG}"
fi
echo "[log] batch status: ${BATCH_STATUS_LOG}"

N_SAMPLES="${N_SAMPLES:-3}"
MAX_FRAMES="${MAX_FRAMES:-20}"
ACTIVE_GS_EXP_ID="${ACTIVE_GS_EXP_ID:-batch_xyz_yaw}"
ACTIVE_GS_GT_METHOD_NAME="${ACTIVE_GS_GT_METHOD_NAME:-active-gs-gt-depth}"
GLEAM_NUM_EVAL_ROUNDS="${GLEAM_NUM_EVAL_ROUNDS:-${N_SAMPLES}}"

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

# ── Run options ───────────────────────────────────────────────────────────────
# Scenes to process.  Command-line positional args override this list.
DEFAULT_SCENES=(
  # "replicacad/apt_0"
  "replicacad/apt_1"
  # "replicacad/apt_2"
  # "replicacad/apt_3"
  # "replicacad/apt_4"
  # "replicacad/apt_5"
)

# DEFAULT_SCENES=(
#   "scene_smith/scene_000"
#   "scene_smith/scene_001"
#   "scene_smith/scene_002"
#   "scene_smith/scene_003"
#   "scene_smith/scene_004"
#   "scene_smith/scene_005"
#   "scene_smith/scene_006"
#   "scene_smith/scene_007"
#   "scene_smith/scene_008"
#   "scene_smith/scene_009"
# )

# true  → re-run and overwrite existing outputs
# false → skip any stage whose output already exists
OVERWRITE="${OVERWRITE:-true}"

# true → sample initial XYZ+yaw poses (writes/overwrites the TSV)
# false → reuse the existing TSV in sampled_inits/
DO_SAMPLE="${DO_SAMPLE:-false}"

# true → run the random baseline
DO_RANDOM="${DO_RANDOM:-false}"

# true → run active-gs (both GT-depth and pred-depth variants)
DO_ACTIVEGS="${DO_ACTIVEGS:-false}"

# true → run GLEAM
DO_GLEAM="${DO_GLEAM:-true}"
# ─────────────────────────────────────────────────────────────────────────────

if (($# > 0)); then
  SCENES=("$@")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

scene_safe() {
  local scene_id="$1"
  printf '%s' "${scene_id//\//__}"
}

sanitize_log_field() {
  local text="${1:-}"
  text="${text//$'\t'/ }"
  text="${text//$'\n'/ }"
  printf '%s' "${text}"
}

append_batch_status() {
  local scene_id="$1"
  local sample_id="$2"
  local run_id="$3"
  local method="$4"
  local status="$5"
  local detail="${6:-}"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="$(sanitize_log_field "${detail}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ts}" "${scene_id}" "${sample_id}" "${run_id}" "${method}" "${status}" "${detail}" >> "${BATCH_STATUS_LOG}"
}

sample_seed_path() {
  local safe_scene="$1"
  printf '%s/%s_xyz_yaw.seed.txt' "${SAMPLE_DIR}" "${safe_scene}"
}

generate_scene_sample_seed() {
  local scene_id="$1"
  "${PROJ_PYTHON}" - "${scene_id}" <<'PY'
import sys
import time
import zlib

scene_id = sys.argv[1]
seed = (time.time_ns() ^ zlib.crc32(scene_id.encode("utf-8"))) % (2**31 - 1)
if seed <= 0:
    seed = 1
print(seed)
PY
}

resolve_scene_sample_seed() {
  local scene_id="$1"
  local safe_scene="$2"
  local seed_path
  seed_path="$(sample_seed_path "${safe_scene}")"

  if [[ -f "${seed_path}" ]]; then
    local seed
    seed="$(tr -d '[:space:]' < "${seed_path}")"
    if [[ "${seed}" =~ ^[0-9]+$ ]] && (( seed > 0 )); then
      echo "${seed}"
      return
    fi
    echo "[warn] invalid sample seed file, regenerating: ${seed_path}" >&2
  fi

  local seed
  seed="$(generate_scene_sample_seed "${scene_id}")"
  printf '%s\n' "${seed}" > "${seed_path}"
  echo "[seed] scene ${scene_id}: ${seed} -> ${seed_path}" >&2
  echo "${seed}"
}

activate_gleam_env() {
  local had_nounset=0
  if [[ $- == *u* ]]; then had_nounset=1; set +u; fi
  source "${GLEAM_CONDA_SH}"
  conda activate "${GLEAM_CONDA_ENV}"
  if [[ "${had_nounset}" -eq 1 ]]; then set -u; fi
}

gleam_scene_prefix() {
  local scene_id="$1"
  printf '%s%s' "${GLEAM_SCENE_PREFIX_PREFIX}" "$(scene_safe "${scene_id}")"
}

gleam_dataset_name() {
  local scene_id="$1"
  printf 'glb_%s' "$(scene_safe "${scene_id}")"
}

gleam_height_tag() {
  local height="$1"
  if [[ "${height}" == *.* ]]; then
    height="${height%0}"
    height="${height%.}"
  fi
  printf '%s' "${height//./d}"
}

gleam_scene_preprocessed() {
  local scene_id="$1"
  local dataset_name gt_dir urdf_path height_tag
  dataset_name="$(gleam_dataset_name "${scene_id}")"
  gt_dir="${GLEAM_ROOT}/${GLEAM_GT_ROOT}/gt_${dataset_name}"
  urdf_path="${GLEAM_ROOT}/${GLEAM_URDF_ROOT}/${dataset_name}/scene_0.urdf"
  height_tag="$(gleam_height_tag "${GLEAM_DRONE_HEIGHT}")"

  [[ -f "${gt_dir}/${dataset_name}_${GLEAM_GRID_RESO}_range_gt.pt" ]] &&
  [[ -f "${gt_dir}/${dataset_name}_${GLEAM_GRID_RESO}_voxel_size_gt.pt" ]] &&
  [[ -f "${gt_dir}/${dataset_name}_${GLEAM_GRID_RESO}_occ_map_height_${height_tag}_gt.pt" ]] &&
  [[ -f "${gt_dir}/${dataset_name}_${GLEAM_GRID_RESO}_init_map_${height_tag}.pt" ]] &&
  [[ -f "${urdf_path}" ]]
}

ensure_gleam_scene_preprocessed() {
  local scene_id="$1"
  local preprocess_overwrite_flag=""
  local glb_path="${DATA_DIR}/${scene_id}.glb"

  if gleam_scene_preprocessed "${scene_id}"; then
    echo "[skip] gleam preprocess already exists: ${scene_id}"
    return
  fi

  if [[ ! -f "${glb_path}" ]]; then
    echo "[error] cannot preprocess missing GLB: ${glb_path}" >&2
    return 1
  fi

  if [[ "${GLEAM_PREPROCESS_OVERWRITE}" == true ]]; then
    preprocess_overwrite_flag="--overwrite"
  fi

  echo "[run] gleam preprocess ${scene_id}"
  (
    cd "${GLEAM_ROOT}"
    activate_gleam_env
    export CC="${GLEAM_CC}"
    export CXX="${GLEAM_CXX}"
    export CUDAHOSTCXX="${GLEAM_CUDAHOSTCXX}"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    export PYTHONPATH="${GLEAM_ROOT}/isaacgym/python${PYTHONPATH:+:${PYTHONPATH}}"

    "${GLEAM_PYTHON}" "${PROJ_ROOT}/baselines/gleam_poses/preprocess_normalized_glb.py" \
      --glb "${glb_path}" \
      --scene_id "${scene_id}" \
      --drone_height "${GLEAM_DRONE_HEIGHT}" \
      --grid_reso "${GLEAM_GRID_RESO}" \
      --work_dir "${GLEAM_WORK_DIR}" \
      --input_glb_mode "${GLEAM_INPUT_GLB_MODE}" \
      ${preprocess_overwrite_flag}
  )
}

_run_gleam_variant() {
  local scene_id="$1"
  local safe_scene="$2"
  local method_name="$3"
  local use_pred_depth="$4"        # "true" or "false"
  local gleam_prefix
  local expected_complete=1
  local method_dir="${SCRIPT_DIR}/poses/${method_name}"
  gleam_prefix="$(gleam_scene_prefix "${scene_id}")"

  if [[ "${OVERWRITE}" == false ]]; then
    for ((round_idx = 0; round_idx < GLEAM_NUM_EVAL_ROUNDS; round_idx++)); do
      printf -v run_tag 'run_%03d' "${round_idx}"
      if [[ ! -f "${method_dir}/${safe_scene}_${run_tag}.json" ]]; then
        expected_complete=0
        break
      fi
    done
    if [[ "${expected_complete}" -eq 1 ]]; then
      echo "[skip] gleam ${method_name} already exists (OVERWRITE=false): ${method_dir}/${safe_scene}_run_*.json"
      append_batch_status "${scene_id}" "-" "-" "${method_name}" "skipped" "outputs already exist"
      return
    fi
  fi

  echo "[run] gleam ${method_name}"
  if (
    cd "${GLEAM_ROOT}"
    activate_gleam_env
    export CC="${GLEAM_CC}"
    export CXX="${GLEAM_CXX}"
    export CUDAHOSTCXX="${GLEAM_CUDAHOSTCXX}"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    export PYTHONPATH="${GLEAM_ROOT}/isaacgym/python${PYTHONPATH:+:${PYTHONPATH}}"

    cmd=(
      "${GLEAM_PYTHON}" gleam/test/test_gleam_custom.py
      --custom_scene_prefixes "${gleam_prefix}"
      --custom_gt_root "${GLEAM_GT_ROOT}"
      --custom_urdf_root "${GLEAM_URDF_ROOT}"
      --custom_sampled_init_root "${SAMPLE_DIR}"
      --custom_num_eval_round "${GLEAM_NUM_EVAL_ROUNDS}"
      --custom_debug_obs "${GLEAM_DEBUG_OBS}"
      --custom_debug_obs_env_idx "${GLEAM_DEBUG_OBS_ENV_IDX}"
      --custom_debug_obs_max_steps "${GLEAM_DEBUG_OBS_MAX_STEPS}"
      --custom_baseline_pose_root "${SCRIPT_DIR}/poses"
      --custom_baseline_pose_method "${method_name}"
      --drone_height "${GLEAM_DRONE_HEIGHT}"
      --num_envs "${GLEAM_NUM_ENVS}"
      --sim_device "${GLEAM_SIM_DEVICE}"
      --eval_device "${GLEAM_EVAL_DEVICE}"
      --ckpt_path "${GLEAM_CKPT_PATH}"
      --stop_wandb
      --headless
    )
    if [[ "${use_pred_depth}" == true ]]; then
      cmd+=(
        --custom_pred_depth True
        --custom_pred_depth_model "${GLEAM_PRED_DEPTH_MODEL}"
        --custom_pred_depth_cache_dir "${GLEAM_PRED_DEPTH_CACHE_DIR}"
      )
    fi
    if [[ -n "${GLEAM_SAVE_PATH}" ]]; then
      cmd+=(--custom_save_path "${GLEAM_SAVE_PATH}")
    fi
    "${cmd[@]}"
  ); then
    append_batch_status "${scene_id}" "-" "-" "${method_name}" "completed" "gleam pose export finished"
  else
    local exit_code=$?
    local status="failed"
    if compgen -G "${method_dir}/${safe_scene}_run_*.json" > /dev/null; then
      status="incomplete"
    fi
    append_batch_status "${scene_id}" "-" "-" "${method_name}" "${status}" "exit_code=${exit_code}"
    echo "[warn] gleam ${method_name} failed for ${scene_id}; continuing" >&2
  fi
  return 0
}

run_gleam_scene() {
  local scene_id="$1"
  local safe_scene="$2"

  if ! ensure_gleam_scene_preprocessed "${scene_id}"; then
    append_batch_status "${scene_id}" "-" "-" "${GLEAM_GT_METHOD_NAME}" "failed" "gleam preprocess failed"
    append_batch_status "${scene_id}" "-" "-" "${GLEAM_PRED_METHOD_NAME}" "failed" "gleam preprocess failed"
    echo "[warn] gleam preprocess failed for ${scene_id}; continuing" >&2
    return 0
  fi

  _run_gleam_variant "${scene_id}" "${safe_scene}" "${GLEAM_GT_METHOD_NAME}" false
  _run_gleam_variant "${scene_id}" "${safe_scene}" "${GLEAM_PRED_METHOD_NAME}" true
}

for scene_id in "${SCENES[@]}"; do
  safe_scene="$(scene_safe "${scene_id}")"
  glb_path="${DATA_DIR}/${scene_id}.glb"
  sample_tsv="${SAMPLE_DIR}/${safe_scene}_xyz_yaw.tsv"

  if [[ ! -f "${glb_path}" ]]; then
    echo "[skip] scene not found: ${glb_path}" >&2
    continue
  fi

  scene_sample_seed="$(resolve_scene_sample_seed "${scene_id}" "${safe_scene}")"

  if [[ "${DO_SAMPLE}" == true ]] || [[ ! -f "${sample_tsv}" ]]; then
    if [[ "${DO_SAMPLE}" != true ]]; then
      echo "[warn] DO_SAMPLE=false but TSV not found; sampling anyway: ${sample_tsv}" >&2
    elif [[ "${OVERWRITE}" == false ]] && [[ -f "${sample_tsv}" ]]; then
      echo "[skip] init-pose TSV already exists (OVERWRITE=false): ${sample_tsv}"
      append_batch_status "${scene_id}" "-" "-" "sample-init" "skipped" "sample TSV already exists"
    else
      echo "==== Sampling init poses for ${scene_id} ===="
      echo "[seed] sampling ${scene_id} with seed ${scene_sample_seed}"
      if (
        cd "${PROJ_ROOT}"
        "${PROJ_PYTHON}" baselines/sample_xyz_pitch.py \
          --scene_id "${scene_id}" \
          --data_dir "${DATA_DIR}" \
          --n_samples "${N_SAMPLES}" \
          --seed "${scene_sample_seed}" \
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
      ); then
        append_batch_status "${scene_id}" "-" "-" "sample-init" "completed" "seed=${scene_sample_seed}"
      else
        sample_exit_code=$?
        append_batch_status "${scene_id}" "-" "-" "sample-init" "failed" "exit_code=${sample_exit_code} seed=${scene_sample_seed}"
        echo "[warn] sampling init poses failed for ${scene_id}; continuing" >&2
      fi
    fi
  else
    echo "[skip] DO_SAMPLE=false, reusing: ${sample_tsv}"
    append_batch_status "${scene_id}" "-" "-" "sample-init" "skipped" "reusing existing TSV"
  fi

  if [[ ! -f "${sample_tsv}" ]]; then
    echo "[skip] no TSV found for ${scene_id}, skipping all method runs" >&2
    append_batch_status "${scene_id}" "-" "-" "sample-init" "failed" "no TSV available after sampling stage"
    continue
  fi

  if [[ "${DO_GLEAM}" == true ]]; then
    run_gleam_scene "${scene_id}" "${safe_scene}"
  else
    echo "[skip] gleam (DO_GLEAM=false)"
    append_batch_status "${scene_id}" "-" "-" "${GLEAM_GT_METHOD_NAME}" "skipped" "DO_GLEAM=false"
    append_batch_status "${scene_id}" "-" "-" "${GLEAM_PRED_METHOD_NAME}" "skipped" "DO_GLEAM=false"
  fi

  while IFS=$'\t' read -r sample_id run_id x y z yaw_deg init_pose_json; do
    echo
    echo "==== Scene ${scene_id} | sample ${sample_id} | run_id ${run_id} ===="
    echo "     xyz=(${x}, ${y}, ${z}) yaw_deg=${yaw_deg}"

    if [[ "${DO_ACTIVEGS}" == true ]]; then
      _ags_run() {
        local method_name="$1"; shift
        local extra_args=("$@")
        local out_dir="${ACTIVE_GS_ROOT}/experiments/${ACTIVE_GS_EXP_ID}/${scene_id}/${method_name}/${sample_id}"
        local pose_json="${SCRIPT_DIR}/poses/${method_name}/${safe_scene}_${run_id}.json"
        if [[ "${OVERWRITE}" == false ]] && [[ -d "${out_dir}/map" ]]; then
          echo "[skip] active-gs ${method_name} already exists (OVERWRITE=false): ${out_dir}"
          append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "${method_name}" "skipped" "output map already exists"
          return
        fi
        echo "[run] active-gs ${method_name}"
        if (
          cd "${ACTIVE_GS_ROOT}"
          had_nounset=0
          if [[ $- == *u* ]]; then had_nounset=1; set +u; fi
          source "${ACTIVE_GS_CONDA_SH}"
          conda activate "${ACTIVE_GS_CONDA_ENV}"
          if [[ "${had_nounset}" -eq 1 ]]; then set -u; fi
          export PYGLFW_LIBRARY="${ACTIVE_GS_PYGLFW_LIBRARY}"
          export __EGL_VENDOR_LIBRARY_FILENAMES="${ACTIVE_GS_EGL_VENDOR_JSON}"
          export LD_LIBRARY_PATH="${ACTIVE_GS_SYSTEM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
          export LD_PRELOAD="${ACTIVE_GS_LD_PRELOAD_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
          "${ACTIVE_GS_PYTHON}" main.py \
              "scene_name=${scene_id}.glb" \
              "experiment.max_frames=${MAX_FRAMES}" \
              "experiment.exp_id=${ACTIVE_GS_EXP_ID}" \
              "experiment.run_id=${sample_id}" \
              "experiment.pose_method_name=${method_name}" \
              "planner.init_pose=${init_pose_json}" \
              "mapper.voxel_map.safety_margin=${DEFAULT_SAFETY_MARGIN}" \
              "planner.robot_size=${DEFAULT_ROBOT_SIZE}" \
              "${extra_args[@]}" \
              use_gui=false
        ); then
          append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "${method_name}" "completed" "active-gs finished"
        else
          local exit_code=$?
          local status="failed"
          if [[ -f "${pose_json}" ]]; then
            status="incomplete"
          fi
          append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "${method_name}" "${status}" "exit_code=${exit_code}"
          echo "[warn] active-gs ${method_name} failed for ${scene_id} ${run_id}; continuing" >&2
        fi
      }

      _ags_run "${ACTIVE_GS_GT_METHOD_NAME}"
      _ags_run "active-gs-pred-depth" "mapper.depth_mode=pred"
    else
      echo "[skip] active-gs (DO_ACTIVEGS=false)"
      append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "${ACTIVE_GS_GT_METHOD_NAME}" "skipped" "DO_ACTIVEGS=false"
      append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "active-gs-pred-depth" "skipped" "DO_ACTIVEGS=false"
    fi

    if [[ "${DO_RANDOM}" == true ]]; then
      random_voxel_path="${DATA_DIR}/${scene_id}.glb.voxels.h5"
      if [[ -f "${random_voxel_path}" ]]; then
        random_seed=$((DEFAULT_SEED + sample_id))
        random_pose_json="${SCRIPT_DIR}/poses/random/${safe_scene}_${run_id}.json"
        echo "[run] random"
        if (
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
            --init_x "${x}" --init_y "${y}" --init_z "${z}" --init_yaw_deg "${yaw_deg}"
        ); then
          append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "random" "completed" "seed=${random_seed}"
        else
          random_exit_code=$?
          random_status="failed"
          if [[ -f "${random_pose_json}" ]]; then
            random_status="incomplete"
          fi
          append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "random" "${random_status}" "exit_code=${random_exit_code} seed=${random_seed}"
          echo "[warn] random failed for ${scene_id} ${run_id}; continuing" >&2
        fi
      else
        echo "[skip] random baselines require voxel grid: ${random_voxel_path}" >&2
        append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "random" "skipped" "missing voxel grid"
      fi
    else
      echo "[skip] random (DO_RANDOM=false)"
      append_batch_status "${scene_id}" "${sample_id}" "${run_id}" "random" "skipped" "DO_RANDOM=false"
    fi

  done < <(tail -n +2 "${sample_tsv}")
done
