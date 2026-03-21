# actrec_data

Active Reconstruction 项目的数据管理目录，负责将多种开源 3D 场景数据集统一转换为 **y-up、带纹理的 `.glb` 格式**。

## 目录结构

```
actrec_data/
├── data2glb.sh          # 一键转换脚本
├── README.md
├── src_data/            # 原始开源数据集（只读，不修改）
│   ├── ProcTHOR/        # AI2-THOR Habitat 格式场景
│   ├── ReplicaCAD/      # ReplicaCAD 场景（含 URDF 铰接物体）
│   └── replica_v1/      # Replica v1 单网格场景
├── preprocess/          # 数据转换脚本
│   ├── procthor2mesh.py
│   ├── replicaCAD2mesh.py
│   ├── replicav12mesh.py
│   ├── HSSD2mesh.py
│   ├── glb_baking.py
│   └── procthor_room_count.py
└── dst_data/            # 转换输出（y-up textured GLB）
    ├── procthor/
    ├── replicacad/
    └── replicav1/
```

## 数据集概览

| 数据集 | 场景数 | 原始格式 | 转换脚本 | 说明 |
|--------|--------|----------|----------|------|
| ProcTHOR | 120 (training set) | `scene_instance.json` + 资产 | `procthor2mesh.py` | 程序化生成的室内场景，支持高清物体资产替换 |
| ReplicaCAD | ~90 | `scene_instance.json` + URDF | `replicaCAD2mesh.py` | 真实扫描场景 + 可交互物体，支持铰接物体导出 |
| Replica v1 | 18 | `mesh.ply`（顶点色） | `replicav12mesh.py` | 真实扫描的单网格场景，含顶点颜色 |

## 转换流程

所有转换统一执行以下处理：
1. 加载原始场景配置和资产
2. 坐标系转换至 **glTF y-up** 约定
3. 场景归一化：Y 轴 [0,1]，XZ 轴 [-1,1]
4. 保留/烘焙纹理和材质
5. 导出为 `.glb` 格式

## 使用方法

```bash
# 转换所有数据集
./data2glb.sh

# 转换单个数据集
./data2glb.sh procthor
./data2glb.sh replicacad
./data2glb.sh replicav1

# 覆盖已有文件
./data2glb.sh all --overwrite

# 预览命令（不实际执行）
./data2glb.sh all --dry-run
```

## 转换脚本说明

| 脚本 | 用途 |
|------|------|
| `procthor2mesh.py` | 读取 ProcTHOR 的 `scene_instance.json`，组装 stage + objects，处理 ProcTHOR 特有的 90° 旋转，输出 GLB |
| `replicaCAD2mesh.py` | 读取 ReplicaCAD 场景配置，支持 URDF 铰接物体加载，可跳过门物体，输出带纹理 GLB |
| `replicav12mesh.py` | 读取 Replica v1 的 `mesh.ply`（顶点色），批量转换为 GLB，可选添加白色天花板 |
| `HSSD2mesh.py` | 转换 HSSD 数据集场景，支持语义过滤和门物体移除 |
| `glb_baking.py` | 通过 Blender 将 GLB 模型的顶点色烘焙为纹理贴图（需要 Blender 环境） |
| `procthor_room_count.py` | 分析 ProcTHOR 场景的房间数量（辅助工具） |

## 依赖

- Python 3.9+
- trimesh, numpy, open3d
- Blender（仅 `glb_baking.py` 需要）
