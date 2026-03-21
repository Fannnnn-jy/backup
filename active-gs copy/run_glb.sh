#!/usr/bin/env bash
# Run active-GS on a custom GLB scene.
# Usage: bash run_glb.sh [run_id]
#
# Before first run, inspect the scene to set the correct init_pose:
#   python inspect_scene.py scene=glb/custom
# Then update init_pose in config/planner/confidence.yaml

set -e
RUN_ID=${1:-0}
EXP_ID=$(date "+%Y%m%d-%H%M")

python main.py \
    scene=glb/custom \
    planner=confidence \
    mapper=incremental \
    experiment.exp_id=${EXP_ID} \
    experiment.run_id=${RUN_ID} \
    use_gui=true
