#!/bin/bash
echo "start running"
export WANDB_USERNAME=fannnnn-jy-peking-university
python scripts/train.py --logger wandb --wandb-project gleam_baseline --env.gleam_scene_index=6 --env.no-gleam-resample-scene --env.no-gleam-use-init-pose --wandb-log-images-every 100
echo "finish running"

