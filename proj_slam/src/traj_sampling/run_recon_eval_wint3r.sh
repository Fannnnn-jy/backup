#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

# Reconstruction + evaluation with WinT3R for traj_sampling rendered frames.
#
# Reads pose JSONs from poses/ subdirectories, uses rendered frames from
# my_trajectories/rendered/, and writes outputs to
# baselines/reconstructions_traj_sampling and baselines/results_traj_sampling
# with *_wint3r suffixes.
#
# Usage:
#   bash proj/src/traj_sampling/run_recon_eval_wint3r.sh
#   RUNS="manual_run_000 manual_interp5_run_000" bash proj/src/traj_sampling/run_recon_eval_wint3r.sh
#   SCENES="replicacad__apt_0" bash proj/src/traj_sampling/run_recon_eval_wint3r.sh
#   OVERWRITE=true bash proj/src/traj_sampling/run_recon_eval_wint3r.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

TRAJ_DIR="${SCRIPT_DIR}/my_trajectories"
POSES_DIR="${TRAJ_DIR}/poses"
RENDERED_DIR="${TRAJ_DIR}/rendered"
RECON_DIR="${SCRIPT_DIR}/reconstructions_traj_sampling"
RESULTS_DIR="${SCRIPT_DIR}/results_traj_sampling"

WINT3R_REPO="${WINT3R_REPO:-/home/ghr/fs/Junyi/WinT3R}"
MODEL_PATH="${MODEL_PATH:-/home/ghr/fs/Junyi/data/proj/model_weights/WinT3R.bin}"
WINT3R_SIZE="${WINT3R_SIZE:-512}"
WINT3R_INFERENCE_MODE="${WINT3R_INFERENCE_MODE:-online}"

# ── Run options ───────────────────────────────────────────────────────────────
# Space-separated rendered folder names, e.g. "manual_run_000 manual_interp5_run_000".
# Empty = all folders found under rendered/<scene>/.
RUNS="${RUNS:-}"

# Space-separated scene safe-names (e.g. "replicacad__apt_0").
# Empty = all scene dirs found under rendered/.
SCENES="${SCENES:-}"

OVERWRITE="${OVERWRITE:-false}"
SAVE_RECON_EVERY="${SAVE_RECON_EVERY:-0}"
DEVICE="${DEVICE:-cuda}"
COVERAGE_DILATION="${COVERAGE_DILATION:-1}"
MAX_POINTS="${MAX_POINTS:-20000}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -d "${WINT3R_REPO}" ]]; then
  echo "[error] WinT3R repo not found: ${WINT3R_REPO}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] WinT3R weights not found: ${MODEL_PATH}" >&2
  exit 1
fi

# Discover scene directories under rendered/.
if [[ -n "${SCENES}" ]]; then
  scene_dirs=()
  for s in ${SCENES}; do
    d="${RENDERED_DIR}/${s}"
    if [[ -d "${d}" ]]; then
      scene_dirs+=("${d}")
    else
      echo "[warn] scene dir not found: ${d}" >&2
    fi
  done
else
  mapfile -t scene_dirs < <(find "${RENDERED_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#scene_dirs[@]} -eq 0 ]]; then
  echo "[error] no scene directories found under ${RENDERED_DIR}" >&2
  exit 1
fi

n_total=0
n_run=0
n_skip=0

for scene_dir in "${scene_dirs[@]}"; do
  safe_scene="$(basename "${scene_dir}")"

  # Discover run directories under this scene.
  if [[ -n "${RUNS}" ]]; then
    run_dirs=()
    for r in ${RUNS}; do
      d="${scene_dir}/${r}"
      if [[ -d "${d}" ]]; then
        run_dirs+=("${d}")
      else
        echo "[warn] run dir not found: ${d}" >&2
      fi
    done
  else
    mapfile -t run_dirs < <(find "${scene_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
  fi

  for run_dir in "${run_dirs[@]}"; do
    rendered_meta="${run_dir}/meta.json"
    if [[ ! -f "${rendered_meta}" ]]; then
      echo "[skip] meta.json not found: ${rendered_meta}" >&2
      continue
    fi

    # Read scene_id, method, run_id from the rendered meta.json.
    if command -v jq &>/dev/null; then
      scene_id="$(jq -r '.scene_id' "${rendered_meta}")"
      run_id="$(jq -r '.run_id'    "${rendered_meta}")"
      method="$(jq -r '.method'    "${rendered_meta}")"
    else
      scene_id="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['scene_id'])" "${rendered_meta}")"
      run_id="$(  "${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_id'])"    "${rendered_meta}")"
      method="$(  "${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['method'])"    "${rendered_meta}")"
    fi

    # Find the matching pose JSON by searching poses/ recursively.
    pose_json=""
    while IFS= read -r candidate; do
      if command -v jq &>/dev/null; then
        c_sid="$(jq -r '.scene_id' "${candidate}")"
        c_rid="$(jq -r '.run_id'   "${candidate}")"
        c_mtd="$(jq -r '.method'   "${candidate}")"
      else
        c_sid="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['scene_id'])" "${candidate}")"
        c_rid="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_id'])"   "${candidate}")"
        c_mtd="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['method'])"   "${candidate}")"
      fi
      if [[ "${c_sid}" == "${scene_id}" && "${c_rid}" == "${run_id}" && "${c_mtd}" == "${method}" ]]; then
        pose_json="${candidate}"
        break
      fi
    done < <(find "${POSES_DIR}" -name "*.json" | sort)

    if [[ -z "${pose_json}" ]]; then
      echo "[skip] no matching pose JSON for method=${method} scene=${scene_id} run=${run_id}" >&2
      continue
    fi

    result_file="${RESULTS_DIR}/${method}_${safe_scene}_${run_id}_wint3r.json"
    n_total=$((n_total + 1))

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${result_file}" ]]; then
      echo "[skip] result already exists: ${result_file}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== Recon+Eval WinT3R | method=${method} scene=${scene_id} run=${run_id} ===="
    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" baselines/reconstruct_and_eval_wint3r.py \
        --pose_json          "${pose_json}" \
        --rendered_dir       "${RENDERED_DIR}" \
        --recon_dir          "${RECON_DIR}" \
        --results_dir        "${RESULTS_DIR}" \
        --data_dir           "${DATA_DIR}" \
        --wint3r_repo        "${WINT3R_REPO}" \
        --model_path         "${MODEL_PATH}" \
        --size               "${WINT3R_SIZE}" \
        --inference_mode     "${WINT3R_INFERENCE_MODE}" \
        --device             "${DEVICE}" \
        --coverage_dilation  "${COVERAGE_DILATION}" \
        --max_points         "${MAX_POINTS}" \
        --depth_max          "${DEPTH_MAX}" \
        --save_recon_every "${SAVE_RECON_EVERY}"
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_recon_eval_wint3r (traj_sampling) done: ${n_run} run, ${n_skip} skipped, ${n_total} total ===="
