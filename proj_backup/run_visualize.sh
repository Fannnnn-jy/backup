# /home/ghr/anaconda3/envs/src-new/bin/python src/scripts/visualize_glb.py \
#     --glb-path /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/apt_1.scene_instance.glb \
#     --no-y-up

# python src/scripts/dev_ms_glb_scene.py /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb/apt_2.scene_instance_y_up_.glb --no-live
# echo "Visualization complete."

# python src/scripts/dev_ms_glb_scene.py /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/HSSD_glb/102343992_closed.glb --no-live
# echo "Visualization complete."

echo "Starting visualization of .glb scene."
# python -m src.scripts.visualize_active /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/procthor_glb/ProcTHOR-Test-0.glb
# python -m src.scripts.visualize_active /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb/apt_2.scene_instance_y_up_.glb


# python src/scripts/visualize_active.py  /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb/apt_2.scene_instance_y_up_.glb \
#     --env-mode active_mapping \
#     --glb-dir  /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb \
#     --glb-glob "*.glb" \
#     --topdown \
#     --height 256 --width 256 \
#     --am-image-mode rgb \
#     --am-single-camera \
#     --am-action-mode xy_cos_sin --am-camera-angles 0 \
#     --rerun --rerun-spawn
#     # --rerun-capture

echo "Visualization"
python src/scripts/visualize_active.py  data/textured_mesh/training_set/ProcTHOR-Val-960.glb \
    --glb-dir  data/textured_mesh/training_set \
    --glb-glob "*.glb" \
    --topdown \
    --height 256 --width 256 \
    --am-image-mode rgb \
    --am-single-camera \
    --am-action-mode xyz_quat --am-camera-angles 0 \
    # --rerun --rerun-spawn \
    # --rerun-mesh \
    # --rerun-mesh-mode both \
    # --rerun-mesh-alpha 0.1 \
    # --rerun-asset-rotate y_up


    # --rerun-capture
