#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# Batch reconstruction + evaluation: for every pose JSON under baselines/poses/,
# call reconstruct_and_eval_cut3r.py to run CUT3R incrementally and compute metrics.
#
# Prerequisites: rendered frames must already exist (run run_render.sh first).
#
# Output: baselines/reconstructions/ and baselines/results/ with *_cut3r suffixes.
#
# Usage:
#   bash baselines/run_recon_eval_cut3r.sh
#   bash baselines/run_recon_eval_cut3r.sh                          # all methods, all scenes
#   METHODS="gleam active-gs-gt-depth" bash baselines/run_recon_eval_cut3r.sh
#   SCENES="replicacad/apt_0" bash baselines/run_recon_eval_cut3r.sh
#   OVERWRITE=true bash baselines/run_recon_eval_cut3r.sh
#   SAVE_RECON_ARTIFACTS=false bash baselines/run_recon_eval_cut3r.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

POSES_DIR="${SCRIPT_DIR}/poses"
RENDERED_DIR="${SCRIPT_DIR}/rendered"
RECON_DIR="${SCRIPT_DIR}/reconstructions"
RESULTS_DIR="${SCRIPT_DIR}/results"

CUT3R_REPO="${CUT3R_REPO:-/home/ghr/fs/Junyi/CUT3R}"
MODEL_PATH="${MODEL_PATH:-/home/ghr/fs/Junyi/data/proj/model_weights/cut3r/cut3r_512_dpt_4_64.pth}"
CUT3R_SIZE="${CUT3R_SIZE:-512}"

# ── Run options ───────────────────────────────────────────────────────────────
# Space-separated list of method names (subdirectory names under poses/).
# Empty = all methods found.
METHODS="${METHODS:-}"

# Space-separated list of scene_ids to process (e.g. "replicacad/apt_0").
# Empty = all scenes found in the pose files.
SCENES="${SCENES:-}"

# true → re-run and overwrite existing results
OVERWRITE="${OVERWRITE:-false}"

# true → save per-k reconstruction artifacts (npy/ply) under baselines/reconstructions
# false → do not write npy/ply, only metrics JSON under baselines/results
SAVE_RECON_ARTIFACTS="${SAVE_RECON_ARTIFACTS:-true}"

# CUT3R inference device
DEVICE="${DEVICE:-cuda}"

# Voxel coverage dilation in voxels (0 = exact hits only)
COVERAGE_DILATION="${COVERAGE_DILATION:-1}"

# Maximum number of points used for Chamfer distance
MAX_POINTS="${MAX_POINTS:-20000}"

# Maximum valid GT depth in metres
DEPTH_MAX="${DEPTH_MAX:-10.0}"
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -d "${CUT3R_REPO}" ]]; then
  echo "[error] CUT3R repo not found: ${CUT3R_REPO}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] CUT3R weights not found: ${MODEL_PATH}" >&2
  exit 1
fi

# Build the list of method directories to scan.
if [[ -n "${METHODS}" ]]; then
  method_dirs=()
  for m in ${METHODS}; do
    d="${POSES_DIR}/${m}"
    if [[ -d "${d}" ]]; then
      method_dirs+=("${d}")
    else
      echo "[warn] method dir not found: ${d}" >&2
    fi
  done
else
  mapfile -t method_dirs < <(find "${POSES_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#method_dirs[@]} -eq 0 ]]; then
  echo "[error] no method directories found under ${POSES_DIR}" >&2
  exit 1
fi

n_total=0
n_run=0
n_skip=0
n_missing_render=0

for method_dir in "${method_dirs[@]}"; do
  method="$(basename "${method_dir}")"

  mapfile -t json_files < <(find "${method_dir}" -maxdepth 1 -name "*.json" | sort)

  for pose_json in "${json_files[@]}"; do
    # Parse scene_id and run_id from the JSON.
    if command -v jq &>/dev/null; then
      scene_id="$(jq -r '.scene_id' "${pose_json}")"
      run_id="$(jq -r '.run_id'    "${pose_json}")"
    else
      scene_id="$("${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['scene_id'])" "${pose_json}")"
      run_id="$(  "${PROJ_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_id'])"    "${pose_json}")"
    fi

    # Optional scene filter.
    if [[ -n "${SCENES}" ]]; then
      match=0
      for s in ${SCENES}; do
        if [[ "${scene_id}" == "${s}" ]]; then match=1; break; fi
      done
      if [[ "${match}" -eq 0 ]]; then
        continue
      fi
    fi

    safe_scene="${scene_id//\//__}"
    result_file="${RESULTS_DIR}/${method}_${safe_scene}_${run_id}_cut3r.json"
    rendered_meta="${RENDERED_DIR}/${safe_scene}/${method}_${run_id}/meta.json"

    n_total=$((n_total + 1))

    # Skip if rendered frames are missing (prerequisite).
    if [[ ! -f "${rendered_meta}" ]]; then
      echo "[skip] rendered frames not found (run run_render.sh first): ${rendered_meta}" >&2
      n_missing_render=$((n_missing_render + 1))
      continue
    fi

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${result_file}" ]]; then
      echo "[skip] result already exists: ${result_file}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== Recon+Eval CUT3R | method=${method} scene=${scene_id} run=${run_id} ===="
    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" baselines/reconstruct_and_eval_cut3r.py \
        --pose_json          "${pose_json}" \
        --rendered_dir       baselines/rendered \
        --recon_dir          baselines/reconstructions \
        --results_dir        baselines/results \
        --data_dir           "${DATA_DIR}" \
        --cut3r_repo         "${CUT3R_REPO}" \
        --model_path         "${MODEL_PATH}" \
        --size               "${CUT3R_SIZE}" \
        --device             "${DEVICE}" \
        --coverage_dilation  "${COVERAGE_DILATION}" \
        --max_points         "${MAX_POINTS}" \
        --depth_max          "${DEPTH_MAX}" \
        --save_recon_artifacts "${SAVE_RECON_ARTIFACTS}"
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_recon_eval_cut3r done: ${n_run} run, ${n_skip} skipped, ${n_missing_render} missing render, ${n_total} total ===="
