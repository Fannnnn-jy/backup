import trimesh
import numpy as np

def convert_y_up_to_z_up(glb_path, output_path):
    # 1. 读入 GLB 模型
    # force='mesh' 会将场景中的所有几何体合并为一个单体 mesh
    mesh = trimesh.load(glb_path, force='mesh')
    
    # 2. 构造变换矩阵：Y-up -> Z-up
    # 逻辑：新x=旧x, 新y=-旧z, 新z=旧y
    transform = [
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ]
    
    # 3. 应用变换
    mesh.apply_transform(transform)
    
    # 4. 存储为新的 GLB
    mesh.export(output_path)
    print(f"Successfully converted {glb_path} to Z-up and saved to {output_path}")

# 使用示例
convert_y_up_to_z_up("/home/ghr/fs/Junyi/ProcTHOR-Val-960.glb", "/home/ghr/fs/Junyi/ProcTHOR-Val-960_z_up.glb")