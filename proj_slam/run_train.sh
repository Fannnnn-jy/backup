#!/bin/bash
# proj_slam training script — SLAM-Former reconstruction backend
#
# Reconstruction model options (--pointcloud_eval.reconstruction_model):
#   "slamformer"   — incremental SLAM-Former (default here)
#   "vggt"         — batch VGGT-1B
#   "placeholder"  — stub for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/scripts/train.py \
    --policy.class_name SLAMFormerRelativeActorCritic \
    --policy.slamformer_relative.slam_root /home/ghr/fs/Junyi/SLAM-Former \
    --policy.slamformer_relative.slam_ckpt_path /home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model \
    --policy.slamformer_relative.slamformer_input_size 518 \
    --policy.slamformer_relative.visual_dim 256 \
    --policy.slamformer_relative.backend_every 10 \
    --env.action_mode xyz_delta_quat \
    --policy.init_noise_std 0.2 \
    --env.sim_backend maniskill \
    --env.glb_data_dir /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/cad_apt \
    --env.glb_scene_index 0 \
    --env.no-gleam-use-init-pose \
    --env.agent-init-height 0.5 \
    --env.lock-camera-pitch-roll \
    --env.reward_by location \
    --env.no-reward-grid-3d-enabled \
    --env.reward_grid_3d_weight 0.0 \
    --env.maniskill_depth_buffer DepthLinear \
    --env.num_envs 1 \
    --env.max_steps 64 \
    --env.action_scale 0.3 \
    --env.depth-image-shape 512 512 \
    --env.lock-camera-pitch-roll \
    --runner.max_iterations 20000 \
    --pointcloud_eval.enabled \
    --pointcloud_eval.reconstruction_model slamformer \
    --pointcloud_eval.slam_root /home/ghr/fs/Junyi/SLAM-Former \
    --pointcloud_eval.slam_ckpt_path /home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model \
    --pointcloud_eval.slam_target_size 518 \
    --pointcloud_eval.slam_kf_th 0.0 \
    --pointcloud_eval.slam_retention_ratio 0.5 \
    --pointcloud_eval.slam_bn_every 10 \
    --pointcloud_eval.slam_conf_percentile 15 \
    --pointcloud_eval.min_frames 1 \
    --pointcloud_eval.frame_stride 1 \
    --pointcloud_eval.align_mode similarity \
    --pointcloud_eval.distance_mode chamfer \
    --pointcloud_eval.reward_enabled \
    --pointcloud_eval.reward_weight 2.0 \
    --pointcloud_eval.reward_clip 5.0 \
    --pointcloud_eval.debug_print_buffers_once \
    --pointcloud_eval.debug_dump_first_episode \
    --pointcloud_eval.debug_dump_dir debug/slam_first_episode \
    --log_dir /home/ghr/fs/Junyi/data/proj/runs/slam_former \
    --runner.save_interval 10 \
    --pointcloud_eval.no-verbose
