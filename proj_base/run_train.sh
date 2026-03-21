# python src/scripts/train.py \
#     --env.sim_backend maniskill \
#     --env.glb_data_dir /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb \
#     --env.glb_scene_index 2 \
#     --env.reward_by location

# python src/scripts/train.py \
#     --env.sim_backend maniskill \
#     --env.glb_data_dir data/textured_mesh/procthor_glb \
#     --env.glb_scene_index 0 \
#     --env.reward_by location

export CUDA_VISIBLE_DEVICES=1
python src/scripts/train.py \
    --policy.class_name ActorCriticRGBCNNRecurrent \
    --env.action_mode xyz_delta_quat \
    --policy.init_noise_std 0.8 \
    --env.sim_backend maniskill \
    --env.glb_data_dir /home/ghr/fs/Junyi/data/proj/cad \
    --env.glb_scene_index 0 \
    --env.no-gleam-use-init-pose \
    --env.agent-init-height 0.5 \
    --env.lock-camera-pitch-roll \
    --env.reward_by location \
    --env.no-reward-grid-3d-enabled \
    --env.reward_grid_3d_weight 0.0 \
    --env.maniskill_depth_buffer DepthLinear \
    --env.num_envs 1 \
    --env.max_steps 20 \
    --env.action_scale 0.3 \
    --env.depth-image-shape 512 512 \
    --env.lock-camera-pitch-roll \
    --runner.max_iterations 20000 \
    --pointcloud_eval.enabled \
    --pointcloud_eval.model_id facebook/VGGT-1B \
    --pointcloud_eval.model_cache_dir /home/ghr/fs/Junyi/data/proj/model_weights \
    --pointcloud_eval.model_local_files_only \
    --pointcloud_eval.min_frames 1 \
    --pointcloud_eval.frame_stride 1 \
    --pointcloud_eval.align_mode notebook \
    --pointcloud_eval.distance_mode chamfer \
    --pointcloud_eval.reward_enabled \
    --pointcloud_eval.reward_weight 2.0 \
    --pointcloud_eval.reward_clip 5.0 \
    --pointcloud_eval.debug_print_buffers_once \
    --pointcloud_eval.debug_dump_first_episode \
    --pointcloud_eval.debug_dump_dir debug/vggt_first_episode \
    --runner.save_interval 20
