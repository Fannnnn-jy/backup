#!/usr/bin/env bash
set -euo pipefail

# Batch rendering: for every pose JSON under baselines/poses/, call
# render_from_poses.py to produce RGB + depth frames.
#
# Output: baselines/rendered/{scene_id}/{method}_{run_id}/
#
# Usage:
#   bash baselines/run_render.sh
#   bash baselines/run_render.sh                          # all methods, all scenes
#   METHODS="gleam active-gs-gt-depth" bash baselines/run_render.sh
#   SCENES="replicacad/apt_0" bash baselines/run_render.sh
#   OVERWRITE=true bash baselines/run_render.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data}"
PROJ_PYTHON="${PROJ_PYTHON:-python}"

POSES_DIR="${SCRIPT_DIR}/poses"
RENDERED_DIR="${SCRIPT_DIR}/rendered"

# ── Run options ───────────────────────────────────────────────────────────────
# Space-separated list of method names (subdirectory names under poses/).
# Empty = all methods found.
# METHODS="${METHODS:-}"
METHODS="${METHODS:-random}"

# Space-separated list of scene_ids to process (e.g. "replicacad/apt_0").
# Empty = all scenes found in the pose files.
# SCENES="${SCENES:-}"
SCENES="${SCENES:-replicacad/apt_0}"

# true → re-render and overwrite existing outputs
OVERWRITE="${OVERWRITE:-true}"

# SAPIEN render backend: "gpu" or "cpu"
RENDER_BACKEND="${RENDER_BACKEND:-gpu}"
# Skip a rendered frame when invalid depth ratio is greater than this threshold.
INVALID_DEPTH_SKIP_RATIO="${INVALID_DEPTH_SKIP_RATIO:-0.25}"
# ─────────────────────────────────────────────────────────────────────────────

overwrite_flag=""
if [[ "${OVERWRITE}" == true ]]; then
  overwrite_flag="--overwrite"
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

for method_dir in "${method_dirs[@]}"; do
  method="$(basename "${method_dir}")"

  mapfile -t json_files < <(find "${method_dir}" -maxdepth 1 -name "*.json" | sort)

  for pose_json in "${json_files[@]}"; do
    # Parse scene_id and run_id from the JSON (fast jq if available, else python).
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
    out_dir="${RENDERED_DIR}/${safe_scene}/${method}_${run_id}"

    n_total=$((n_total + 1))

    if [[ "${OVERWRITE}" == false ]] && [[ -f "${out_dir}/meta.json" ]]; then
      echo "[skip] already rendered: ${out_dir}"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "==== Render | method=${method} scene=${scene_id} run=${run_id} ===="
    (
      cd "${PROJ_ROOT}"
      "${PROJ_PYTHON}" baselines/render_from_poses.py \
        --pose_json      "${pose_json}" \
        --data_dir       "${DATA_DIR}" \
        --output_dir     baselines/rendered \
        --render_backend "${RENDER_BACKEND}" \
        --invalid_depth_skip_ratio "${INVALID_DEPTH_SKIP_RATIO}" \
        ${overwrite_flag}
    )
    n_run=$((n_run + 1))
  done
done

echo
echo "==== run_render done: ${n_run} rendered, ${n_skip} skipped, ${n_total} total ===="
