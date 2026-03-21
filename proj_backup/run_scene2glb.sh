#!/usr/bin/env bash
# python  actrec/scripts/replicaCAD2mesh.py \
#     --scene-cfg /datasets/v2p/current/fs/Junyi/data/scene_datasets/replica_cad_dataset/configs/scenes/apt_2.scene_instance.json \
#     --asset-dir /datasets/v2p/current/fs/Junyi/data/scene_datasets/replica_cad_dataset \
#     --save-path /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/apt_2.scene_instance_y_up_.glb \
#     --export-textures \
#     --export-y-up

# SCENE_ROOT=/datasets/v2p/current/fs/Junyi/data/scene_datasets/replica_cad_dataset/configs/scenes
# ASSET_DIR=/datasets/v2p/current/fs/Junyi/data/scene_datasets/replica_cad_dataset
# OUT_DIR=/datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/replica_cad_glb
# mkdir -p "$OUT_DIR"

# find "$SCENE_ROOT" -name "*.scene_instance.json" -print0 | while IFS= read -r -d '' f; do
#     scene=$(basename "$f" .scene_instance.json)
#     python actrec/scripts/replicaCAD2mesh.py \
#         --scene-cfg "$f" \
#         --asset-dir "$ASSET_DIR" \
#         --save-path "$OUT_DIR/${scene}.glb" \
#         --export-textures \
#         --export-y-up
# done
# echo "Conversion of ReplicaCAD scenes to .glb format complete."



# python -m actrec.scripts.HSSD2mesh \
#     --scene-id 102343992 \
#     --overwrite \
#     --log-removed



# python -m src.scripts.procthor2glb \
#     --dataset-root /datasets/v2p/current/fs/Junyi/data/scene_datasets/ai2thor_hab/ai2thor-hab \
#     --object-asset-dir /datasets/v2p/current/fs/Junyi/data/scene_datasets/ai2thor_hab/ai2thorhab-uncompressed/assets/objects \
#     --scene-id ProcTHOR-Test-0 \
#     --output-dir /datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/procthor_glb \
#     --overwrite \
#     --export-y-up

# 定义路径变量
ID_FILE="/home/ghr/fs/Junyi/proj/data/ai2thor_hab/selected_1room_ids.txt"
DATASET_ROOT="/datasets/v2p/current/fs/Junyi/data/scene_datasets/ai2thor_hab/ai2thor-hab"
OBJ_ASSET_DIR="/datasets/v2p/current/fs/Junyi/data/scene_datasets/ai2thor_hab/ai2thorhab-uncompressed/assets/objects"
# 建议确认此路径是否有空间，或者先输出到 Home 目录
OUTPUT_DIR="/datasets/v2p/current/fs/Junyi/data/scene_datasets/textured_mesh/training_set"

# 检查文件是否存在
if [ ! -f "$ID_FILE" ]; then
    echo "Error: $ID_FILE not found!"
    exit 1
fi

# 逐行读取 ID
while IFS= read -r line || [ -n "$line" ]; do
    # 1. 去除行尾回车符（防止 Windows 换行符导致的路径错误）并跳过空行
    clean_id=$(echo "$line" | tr -d '\r' | xargs)
    [ -z "$clean_id" ] && continue
    
    # 2. 拼接完整的 Scene ID
    # 如果 ID 已经是 ProcTHOR- 开头则不加，否则加上前缀
    if [[ $clean_id == ProcTHOR-* ]]; then
        full_scene_id="$clean_id"
    else
        full_scene_id="ProcTHOR-$clean_id"
    fi
    
    echo "------------------------------------------------"
    echo "Processing: $full_scene_id"
    
    # 3. 执行 Python 命令
    python -m src.scripts.procthor2glb \
        --dataset-root "$DATASET_ROOT" \
        --object-asset-dir "$OBJ_ASSET_DIR" \
        --scene-id "$full_scene_id" \
        --output-dir "$OUTPUT_DIR" \
        --overwrite \
        --export-y-up
        
done < "$ID_FILE"