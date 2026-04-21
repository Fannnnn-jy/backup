# #!/bin/bash
# echo "start running"
# python scripts/visualize_depth.py --load-model /data/gleam_baseline/runs/active_mapping/20260125_214849/model_1000.pt --width 64 --height 64 --scene-index 6 --first-camera-only --no-gleam-use-init-pose
# echo "finish running"

# python -m actrec.scripts.visualize_active /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/HSSD_glb/102343992.glb
python -m actrec.scripts.visualize_active /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb/apt_2.scene_instance_y_up_.glb