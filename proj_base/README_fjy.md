# 环境定义与训练说明（src）

本文档说明 `/home/ghr/fs/Junyi/proj/src` 中环境定义、体素化方式、坐标约定、GLB 处理、以及训练与 policy 的输入输出，并在每一部分标注相关函数与脚本。

## 1. 环境入口与核心类

- 核心环境：`envs/active_mapping_env.py`
  - 直接从 `gleam_baseline` 抄过来的
- ManiSkill 版本：`envs/maniskill_active_mapping_env.py`
- 训练入口：`src/scripts/train.py`

## 2. 体素定义（Voxel Definition）

### 2.1 GLB 路径（glb_data_dir 非空）

- 体素化入口函数：
  - `ActiveMappingEnv._set_glb_scene` (`envs/active_mapping_env.py`)
  - `ActiveMappingEnv._load_glb_voxel_cache`
  - `ActiveMappingEnv._build_glb_voxel_cache`
- 体素化实现：`trimesh.Trimesh.voxelized(pitch)`
- 体素是**立方体**，尺寸统一为 `pitch`（voxel size = `[pitch, pitch, pitch]`）。
- `pitch` 计算函数：`ActiveMappingEnv._compute_voxel_pitch`
  - 当 `cfg.glb_voxel_pitch > 0` 时：直接使用该值。
  - 否则：`pitch = max(bounds_size) / max_dim`，其中：
    - `bounds_size = bounds[1] - bounds[0]`
    - `max_dim = max(8, cfg.glb_voxel_max_dim)`
- 网格缓存：`ActiveMappingEnv._save_glb_voxel_cache` / `ActiveMappingEnv._load_glb_voxel_cache`

### 2.2 GLEAM 数据路径（gleam_data_dir）

- 数据加载入口：`ActiveMappingEnv._init_gleam_dataset` / `ActiveMappingEnv._set_gleam_scene`
- 数据集类：`GleamDataset`（`envs/gleam_dataset.py`）
  - `GleamDataset._load` 读取 `range_gt` 与 `voxel_size`
  - `GleamDataset.get_scene` 返回 `GleamScene`
- 环境直接使用 `scene.voxel_size`，不在运行时重体素化。

## 3. Reward Grid 说明

Reward grid 是用于统计探索覆盖率与奖励发放的 2D 网格，通常比 voxel 的二维投影更大。

### 3.1 形状与初始化

- 相关字段：
  - `ActiveMappingConfig.reward_grid_size`
  - `ActiveMappingEnv._reward_grid_shape`
  - `ActiveMappingEnv._reward_visited`
- 形状解析函数：`ActiveMappingEnv._resolve_reward_grid_shape`
  - `reward_grid_size = None` → 与 `_grid_shape` 相同
  - `reward_grid_size = int` → `(int, int)`
  - `reward_grid_size = (h, w)` → 按给定值
  - 注意这里的 size 是指整体网格尺寸而不是单个小格子，也就是说这个值越大分的越碎

### 3.2 valid mask 与布局约束

- 当存在布局/障碍 mask 时：
  - `ActiveMappingEnv._set_gleam_scene` 里生成 `_valid_mask`
  - `_update_reward_valid_mask` 将 valid 区域下采样/匹配到 reward grid
- 用于限制奖励只在可行区域发放。

### 3.3 patch → reward grid 映射

- 映射函数：`ActiveMappingEnv._patch_to_reward_idx`
  - 当 reward grid 与导航 grid 形状不同，按比例缩放行列索引。

### 3.4 奖励发放模式

奖励逻辑在 `ManiSkillActiveMappingEnv.step` 中：

- `reward_by = "location"`（默认）：
  - 通过 `mark_patch_visited` 标记探索位置
  - 然后映射到 reward grid，首次访问奖励 `grid_reward`
- `reward_by = "sight"`：
  - 留给用视线覆盖面积计算奖励（还没 debug 完）



## 4. 三维 Reward Grid

三维 Reward Grid 用于统计 3D 空间覆盖情况，并可给出额外奖励。默认关闭，需要显式开启。

### 4.1 配置项

- `reward_grid_3d_enabled`：是否启用 3D reward grid（默认 `False`）
- `reward_grid_3d_size`：3D reward grid 尺寸
  - `None` → 使用体素网格尺寸 `_grid_shape_3d`
  - `N` → 使用 `(N, N, N)`
  - 这里的尺寸和二维一样，是指把场景分割的尺寸，N 越大分的越碎
- `reward_grid_3d_weight`：每个新访问 3D cell 的奖励权重
- `reward_grid_3d_traverse`：是否沿路径采样多个位置更新 3D grid （指比如一步走不止 1 格，是否计算途径）


### 4.2 更新与奖励逻辑

- 位置映射：`_position_to_voxel` → `_voxel_to_reward3d_idx`
- 更新逻辑在：`ManiSkillActiveMappingEnv.step`
  - `reward_grid_3d_traverse=True` 时沿路径采样
  - `reward_grid_3d_traverse=False` 时只统计终点
  - 奖励增量：`newly_3d * reward_grid_3d_weight`

## 5. 世界坐标与体素坐标定义

### 5.1 世界坐标 → 体素索引

函数：
- `ActiveMappingEnv._position_to_voxel`（`envs/active_mapping_env.py`）
- `voxel_carving._world_to_voxel`（`envs/voxel_carving.py`）

映射规则 `voxel_size = [vx, vy, vz]`

```
x_min_voxel = x_min - 0.5 * vx
ix = floor((x - x_min_voxel) / vx)

同理:
- y_min_voxel = y_min - 0.5 * vy
- z_min_voxel = z_min - 0.5 * vz
```

索引会被裁剪到 `[0, grid_size-1]`。

### 5.2 体素索引 → 世界坐标

相关函数：
- `ActiveMappingEnv._compute_obstacle_centers`（将占据体素索引转换为世界坐标中心）

映射规则（以 `voxel_size = [vx, vy, vz]` 为例）：

```
x = x_min + ix * vx
```

对应 `y/z` 同理。

### 5.3 体素网格索引顺序

GLB 体素网格来自 `trimesh`：

- `grid.shape = (nx, ny, nz)`，索引顺序为 `[x, y, z]`
- 环境内部：
  - `_grid_shape_3d = (nx, ny, nz)`
  - `_grid_shape = (ny, nx)`（用于 2D patch/topdown）

相关函数：
- `ActiveMappingEnv._set_glb_scene`
- `ActiveMappingEnv._finalize_scene`

## 6. .glb 坐标与变换

GLB 加载与坐标变换在：
- `ActiveMappingEnv._load_glb_mesh_for_voxels`

规则：
- 默认假设 GLB 为 **Y‑up** 坐标系。
- 若 `cfg.glb_y_up = True`，对 mesh 施加**绕 X 轴 +90°** 旋转。
- `cfg.glb_scale` 提供全局缩放。
