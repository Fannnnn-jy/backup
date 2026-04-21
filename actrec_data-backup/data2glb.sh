#!/bin/bash
# ==============================================================================
# data2glb.sh - Convert datasets to y-up textured GLB format
#
# Usage:
#   ./data2glb.sh [dataset] [options]
#
# Datasets:
#   all          Convert all datasets (default)
#   procthor     Convert ProcTHOR scenes
#   replicacad   Convert ReplicaCAD scenes
#   replicav1    Convert Replica v1 scenes
#
# Options:
#   --overwrite  Overwrite existing GLB files
#   --dry-run    Print commands without executing
# ==============================================================================

set -euo pipefail

# ---- Path Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_DIR="${SCRIPT_DIR}/preprocess"
SRC_DATA_DIR="${SCRIPT_DIR}/src_data"
DST_DATA_DIR="${SCRIPT_DIR}/dst_data"

# ProcTHOR paths
PROCTHOR_DATASET_ROOT="${SRC_DATA_DIR}/ProcTHOR/ai2thor-hab"
PROCTHOR_OBJ_ASSET_DIR="${SRC_DATA_DIR}/ProcTHOR/ai2thorhab-uncompressed/assets/objects"
PROCTHOR_SCENE_LIST="/home/ghr/fs/Junyi/procthor_training_set.txt"
PROCTHOR_OUTPUT_DIR="${DST_DATA_DIR}/procthor"

# ReplicaCAD paths
REPLICACAD_DATASET_ROOT="${SRC_DATA_DIR}/ReplicaCAD"
REPLICACAD_SCENES_DIR="${REPLICACAD_DATASET_ROOT}/configs/scenes"
REPLICACAD_OUTPUT_DIR="${DST_DATA_DIR}/replicacad"

# Replica v1 paths
REPLICAV1_DATASET_ROOT="${SRC_DATA_DIR}/replica_v1"
REPLICAV1_OUTPUT_DIR="${DST_DATA_DIR}/replicav1"

# ---- Options ----
OVERWRITE=""
DRY_RUN=false
DATASET="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        procthor|replicacad|replicav1|all)
            DATASET="$1"; shift ;;
        --overwrite)
            OVERWRITE="--overwrite"; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            head -14 "$0" | tail -13; exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

run_cmd() {
    echo "[CMD] $*"
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

# ==============================================================================
# ProcTHOR conversion
# ==============================================================================
convert_procthor() {
    echo "=============================================="
    echo " Converting ProcTHOR scenes"
    echo "=============================================="

    mkdir -p "${PROCTHOR_OUTPUT_DIR}"

    if [ ! -f "${PROCTHOR_SCENE_LIST}" ]; then
        echo "Warning: Scene list not found: ${PROCTHOR_SCENE_LIST}"
        echo "Converting all ProcTHOR scenes instead..."
        run_cmd python "${PREPROCESS_DIR}/procthor2mesh.py" \
            --dataset-root "${PROCTHOR_DATASET_ROOT}" \
            --object-asset-dir "${PROCTHOR_OBJ_ASSET_DIR}" \
            --output-dir "${PROCTHOR_OUTPUT_DIR}" \
            --export-y-up \
            ${OVERWRITE}
        return
    fi

    local count=0
    local total
    total=$(grep -c '[^[:space:]]' "${PROCTHOR_SCENE_LIST}" || true)
    echo "Found ${total} scenes in scene list"

    while IFS= read -r line || [ -n "$line" ]; do
        clean_id=$(echo "$line" | tr -d '\r' | xargs)
        [ -z "$clean_id" ] && continue

        if [[ $clean_id == ProcTHOR-* ]]; then
            full_scene_id="$clean_id"
        else
            full_scene_id="ProcTHOR-$clean_id"
        fi

        count=$((count + 1))
        echo "------------------------------------------------"
        echo "[${count}/${total}] Processing: ${full_scene_id}"

        run_cmd python "${PREPROCESS_DIR}/procthor2mesh.py" \
            --dataset-root "${PROCTHOR_DATASET_ROOT}" \
            --object-asset-dir "${PROCTHOR_OBJ_ASSET_DIR}" \
            --scene-id "${full_scene_id}" \
            --output-dir "${PROCTHOR_OUTPUT_DIR}" \
            --export-y-up \
            ${OVERWRITE}
    done < "${PROCTHOR_SCENE_LIST}"

    echo "ProcTHOR: ${count} scenes processed"
}

# ==============================================================================
# ReplicaCAD conversion
# ==============================================================================
convert_replicacad() {
    echo "=============================================="
    echo " Converting ReplicaCAD scenes"
    echo "=============================================="

    mkdir -p "${REPLICACAD_OUTPUT_DIR}"

    local count=0
    for scene_cfg in "${REPLICACAD_SCENES_DIR}"/*.scene_instance.json; do
        [ -f "$scene_cfg" ] || continue

        local scene_name
        scene_name=$(basename "$scene_cfg" .scene_instance.json)

        # Skip empty_stage
        [[ "$scene_name" == "empty_stage" ]] && continue

        local output_file="${REPLICACAD_OUTPUT_DIR}/${scene_name}.glb"
        if [ -f "$output_file" ] && [ -z "$OVERWRITE" ]; then
            echo "Skipping ${scene_name} (already exists)"
            continue
        fi

        count=$((count + 1))
        echo "------------------------------------------------"
        echo "[${count}] Processing: ${scene_name}"

        run_cmd python "${PREPROCESS_DIR}/replicaCAD2mesh.py" \
            --scene-cfg "${scene_cfg}" \
            --asset-dir "${REPLICACAD_DATASET_ROOT}" \
            --save-path "${output_file}" \
            --export-textures \
            --export-y-up \
            --normalize-axes \
            # --skip-doors

    done

    echo "ReplicaCAD: ${count} scenes processed"
}

# ==============================================================================
# Replica v1 conversion
# ==============================================================================
convert_replicav1() {
    echo "=============================================="
    echo " Converting Replica v1 scenes"
    echo "=============================================="

    mkdir -p "${REPLICAV1_OUTPUT_DIR}"

    run_cmd python "${PREPROCESS_DIR}/replicav12mesh.py" \
        --dataset-root "${REPLICAV1_DATASET_ROOT}" \
        --output-dir "${REPLICAV1_OUTPUT_DIR}" \
        --output-ext ".glb" \
        --export-y-up \
        --normalize-axes \
        ${OVERWRITE}

    echo "Replica v1: conversion complete"
}

# ==============================================================================
# Main
# ==============================================================================
echo "=============================================="
echo " data2glb - Dataset to GLB Converter"
echo " Output: ${DST_DATA_DIR}"
echo "=============================================="

case "$DATASET" in
    all)
        convert_procthor
        convert_replicacad
        convert_replicav1
        ;;
    procthor)
        convert_procthor
        ;;
    replicacad)
        convert_replicacad
        ;;
    replicav1)
        convert_replicav1
        ;;
esac

echo ""
echo "=============================================="
echo " Done! GLB files are in: ${DST_DATA_DIR}"
echo "=============================================="
