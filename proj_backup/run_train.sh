# python src/scripts/train.py \
#     --env.sim_backend maniskill \
#     --env.glb_data_dir /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb \
#     --env.glb_scene_index 2 \
#     --env.reward_by location

python src/scripts/train.py \
    --env.sim_backend maniskill \
    --env.glb_data_dir data/textured_mesh/procthor_glb \
    --env.glb_scene_index 0 \
    --env.reward_by location
