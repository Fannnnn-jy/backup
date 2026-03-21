# Baseline Comparison for Active Reconstruction

对不同 active reconstruction 模型给出的相机轨迹，统一通过 VGGT 重建并与 GT 对比，
输出三项逐步增加视角的量化指标：`voxel_coverage`、`chamfer_distance`、`depth_l2_voxel`。

---

## 目录结构

```
baselines/
├── poses/                              # 输入：位姿 JSON（每个模型/场景一个文件）
│   └── example_pose.json
├── rendered/                           # 中间结果：GLB 渲染的 RGB + 深度图
│   └── {scene_id}/{method}_{run_id}/
│       ├── view_0000_rgb.png
│       ├── view_0000_depth.npy
│       └── meta.json
├── reconstructions/                    # 中间结果：VGGT 增量重建
│   └── {method}_{scene_id}_{run_id}/
│       └── views_{k:04d}/              # k = 1, 2, ..., N
│           ├── pred_points_raw.npy
│           ├── pred_points_aligned.npy
│           ├── pred_depth.npy          # (k, H, W) VGGT 预测深度
│           ├── pred_extrinsics.npy     # (k, 4, 4)
│           ├── pred_intrinsics.npy     # (k, 3, 3)
│           ├── gt_points_k.npy        # 前 k 帧 GT 点云
│           └── voxel_coverage.ply     # 绿=已覆盖表面体素，红=未覆盖表面体素
├── results/                            # 输出：指标 JSON
│   └── {method}_{scene_id}_{run_id}.json
├── render_from_poses.py
├── reconstruct_and_eval.py
└── BASELINE.md
```

---

## 位姿 JSON 格式（poses/）

```json
{
  "scene_id":   "procthor/ProcTHOR-Test-21",
  "method":     "active-gs",
  "run_id":     "run_000",
  "max_views":  10,
  "fov_deg":    90.0,
  "image_width":  256,
  "image_height": 256,
  "camera_near":  0.01,
  "camera_far":   10.0,
  "quaternion_convention": "wxyz",
  "coordinate_frame":      "camera_to_world",
  "views": [
    {
      "view_idx":        0,
      "position":        [x, y, z],
      "quaternion_wxyz": [w, x, y, z]
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `scene_id` | 相对于 `actrec_data/dst_data/` 的路径，不含 `.glb` 后缀 |
| `method` | 生成该轨迹的模型名称，用于区分不同 baseline |
| `run_id` | 同一方法多次运行的编号（如 `run_000`） |
| `max_views` | 实际使用的视角上限；`views` 数组可以更长 |
| `fov_deg` | 垂直方向 FOV（度），用于渲染和计算 intrinsics |
| `image_width/height` | 渲染分辨率 |
| `camera_near/far` | 近/远裁切面（米） |
| `quaternion_convention` | 固定为 `wxyz` |
| `coordinate_frame` | 固定为 `camera_to_world`（相机到世界，与 SAPIEN 约定一致） |
| `position` | 相机在世界坐标系下的位置（米） |
| `quaternion_wxyz` | 相机到世界的旋转四元数 |

---

## 使用流程

### Active-GS 轨迹导出

`active-gs` 默认使用 `confidence` planner，默认 `use_gui=true`，但以下两个参数现在必须显式传入：

- `scene_name`
- `experiment.max_frames`

示例：

```bash
cd /home/ghr/fs/Junyi/active-gs
python main.py \
    scene_name=replicacad/apt_0.glb \
    experiment.max_frames=20
```

默认会从 `/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/` 下解析场景文件，因此上面的输入会读取：

```text
/home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/procthor/ProcTHOR-Test-21.glb
```

并导出到：

```text
/home/ghr/fs/Junyi/proj/baselines/poses/active-gs/procthor__ProcTHOR-Test-21_run_000.json
```

对应 JSON metadata 中：

- `method = active-gs`
- `scene_id = procthor/ProcTHOR-Test-21`

### GLEAM 轨迹生成

GLEAM 使用基于 Isaac Gym 的预训练 PPO 策略（LocoTransformer encoder），在体素化场景中规划无人机探索轨迹。
入口脚本：`baselines/gleam_poses/generate.py`，完整流程：GLB 预处理 → Isaac Gym 仿真 → 位姿 JSON。

**依赖**：需要 Isaac Gym 环境，以及固定路径的 GLEAM 预训练权重文件
`/home/ghr/fs/Junyi/data/proj/model_weights/GLEAM/rl_model_40000000_steps.zip`。

```bash
cd /home/ghr/fs/Junyi/proj
python baselines/gleam_poses/generate.py \
    --glb      /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data/replicacad/apt_0.glb \
    --scene_id replicacad/apt_0 \
    --n_views  20 \
    --drone_height 0.5 \
    --run_id   run_000 \
    --output   baselines/poses
```

输出写入：

```text
baselines/poses/gleam/replicacad__apt_0_run_001.json
```

**主要参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--glb` | 必填 | Y-up GLB 场景文件路径 |
| `--scene_id` | 必填 | 场景标识符，如 `procthor/ProcTHOR-Test-21` |
| `--n_views` | 30 | 采集的视角数 |
| `--drone_height` | 0.5 | 无人机飞行高度（Z-up 坐标，米） |
| `--run_id` | `run_000` | 多次运行的编号 |
| `--output` | `baselines/poses` | 位姿 JSON 输出根目录 |
| `--device` | `cuda:0` | CUDA 设备 |

脚本会硬编码加载：
`/home/ghr/fs/Junyi/data/proj/model_weights/GLEAM/rl_model_40000000_steps.zip`
如果该文件不存在，会直接报错退出。
| `--buffer_size` | 30 | 历史状态缓冲帧数（需与训练时一致） |
| `--work_dir` | 系统临时目录 | 预处理中间文件目录，默认用完自动删除 |
| `--keep_work_dir` | false | 保留预处理中间文件（调试用） |

**坐标系说明**：输入 GLB 为 Y-up（GLTF 标准），内部转换为 Z-up（Isaac Gym），输出 JSON 转回 Y-up，与其他 baseline 格式一致。

---

### 步骤 1：渲染

从 GLB 场景按给定位姿渲染 RGB 图和深度图：

```bash
python baselines/render_from_poses.py \
    --pose_json /home/ghr/fs/Junyi/proj/baselines/poses/active-gs/replicacad__apt_0_run_000.json \
    --data_dir  /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data \
    --output_dir baselines/rendered \
    --overwrite
```

渲染结果存到 `baselines/rendered/{scene_id}/{method}_{run_id}/`，
同时生成 `meta.json`（包含 intrinsics 和每帧位姿）。

### 步骤 2：增量重建 + 评估

对前 1 帧、前 2 帧……全部帧，依次运行 VGGT 重建并计算三项指标：

```bash
python baselines/reconstruct_and_eval.py \
    --pose_json    /home/ghr/fs/Junyi/proj/baselines/poses/active-gs/procthor__ProcTHOR-Test-21_run_000.json \
    --align_method first_frame \
    --rendered_dir baselines/rendered \
    --recon_dir    baselines/reconstructions \
    --results_dir  baselines/results \
    --data_dir     /home/ghr/fs/Junyi/data/proj/actrec_data/dst_data \
    --model_cache_dir /home/ghr/fs/Junyi/data/proj/model_weights
```

结果写入 `baselines/results/{method}_{scene_id}_{run_id}.json`。

---

## 输出指标说明

`metrics_by_k` 数组中每条记录对应使用前 k 个视角的结果：

| 字段 | 说明 |
|---|---|
| `k` | 使用的视角数 |
| `voxel_coverage` | GT 深度反投影点云覆盖的**表面**体素比例（0~1）；分母为 surface voxels（见下文），无 `.voxels.h5` 则 fallback 到全 N 帧 GT 体素 |
| `chamfer_distance` | 双向 Chamfer distance（米），对称定义：0.5×(mean A→B + mean B→A) |
| `depth_l2_voxel` | 按 voxel 归一化的深度平方误差：先将预测深度缩放并重采样到 GT 分辨率，再对每个 voxel 内的 `(pred−gt)^2` 在单帧内求平均、在覆盖该 voxel 的多帧间求平均，最后对所有被观测 voxel 求平均 |
| `scale` | Umeyama 估计的各向同性缩放因子（=1 表示无尺度偏差） |
| `pred_points` | 对齐后预测点云点数 |
| `gt_points_k` | 前 k 帧 GT 点云点数 |
| `n_surface_voxels` | 覆盖率分母：GT 网格中暴露于自由空间的表面体素数 |
| `n_gt_voxels` | GT 网格中全部占据体素数（含实心内部体素，仅供参考） |

---

## 对齐方式

通过 `--align_method` 参数选择，默认 `umeyama`。

### `umeyama`（默认）

使用 Umeyama 相似变换对齐（`src/utils/pointcloud_eval.py:_umeyama_alignment`）：
以 VGGT 预测的相机中心 vs GT 相机位置作为对应点对，求最优旋转 R、平移 t 和各向同性缩放 s，
然后将预测点云整体变换 `s·(P @ Rᵀ) + t`。

- 需要 ≥ 3 帧；k < 3 时退化为纯平移对齐
- 深度误差使用同一个 scale s 缩放 VGGT 深度后，进入 voxel-normalized L2 统计
- 优点：自动补偿 VGGT 的尺度偏差

### `first_frame`

利用第 0 帧预测位姿与 GT 位姿之间的刚体变换对齐：

```
T_align = T_gt0_c2w @ T_pred0_w2c
```

即 VGGT世界 → VGGT相机0 → GT世界，等价于"把 VGGT 坐标系的原点和朝向强制对齐到 GT 第一帧相机"。

- k = 1 也可用，无帧数限制
- 固定 scale = 1.0（假设 VGGT 输出 metric 深度，无需缩放）
- 优点：对齐精度不受后续帧预测质量影响，适合研究 k=1 时的重建效果

---

## Voxel Coverage 计算细节

### 指标语义分工

| 指标 | 数据来源 | 衡量内容 |
|---|---|---|
| `voxel_coverage` | GT 深度反投影点云 | agent **探索**了多少场景 |
| `chamfer_distance` | VGGT 预测点云 | 3D **重建**几何精度 |
| `depth_l2_voxel` | VGGT 预测深度 + GT voxel 网格 | **深度估计**精度，且对像素密度与重复观测做 voxel 归一化 |

Coverage 使用 GT 深度而非 VGGT 预测，确保它只反映轨迹探索质量，不受重建误差干扰。

### 深度指标：Voxel-Normalized L2

深度误差不再直接做逐像素 `AbsRel / RMSE / δ1` 汇总，而是按 GT 世界坐标下的 voxel 做归一化统计。
这样可以避免：

- 同一平面上高分辨率区域因像素更多而在指标里占比过大
- 同一个 voxel 被多张相邻视角重复观测时，被简单逐像素平均重复计权

记某个被观测到的 voxel 为 `j`，覆盖它的图像集合为 `I(j)`，图像 `i` 中落入该 voxel 的有效像素集合为 `P(i, j)`。
则指标定义为：

```
per_image_mean(i, j) = mean_{p in P(i, j)} (pred_p - gt_p)^2
voxel_error(j)       = mean_{i in I(j)} per_image_mean(i, j)
depth_l2_voxel       = mean_{j in covered_voxels} voxel_error(j)
```

对应实现流程：

1. 将 VGGT 预测深度 resize 到 GT 深度分辨率。
2. 用对齐得到的全局 `scale` 对预测深度做尺度缩放。
3. 将 GT 有效深度像素反投影到世界坐标，并映射到 GT voxel 网格。
4. 在每张图内，对同一 voxel 的所有有效像素先求一次均方误差。
5. 对覆盖同一 voxel 的多张图再做平均。
6. 最后对所有至少被一个有效像素命中的 voxel 求平均。

这个指标本质上是 voxel-normalized mean squared depth error。
如果后续更希望名字强调这是 MSE 而不是开方后的 L2，可以改名为 `depth_mse_voxel`；当前文档先保留 `depth_l2_voxel` 这个更直观的名字。

### 表面体素筛选

`.voxels.h5` 标记了所有占据体素，包括墙体/地板内部的实心体素，这些体素从任何室内视角都无法被射线击中。
评测使用**表面体素**（6-连通邻域中至少有一个自由体素的占据体素）作为分母：

```
observable = gt_grid & binary_dilation(~gt_grid, structure=6-conn)
coverage   = (covered & observable).sum() / observable.sum()
```

### 膨胀容忍（`--coverage_dilation`）

点云像素密度有限，远处体素可能漏击。`--coverage_dilation N`（默认 `1`）在计算指标前
将已击中的 `covered_mask` 向外膨胀 N 层体素（6-连通），相当于 N × pitch 米的容忍半径。

- `covered_mask` 自身**不做**膨胀，始终记录精确击中，避免累积膨胀滚雪球
- 膨胀只在**计算指标和生成 `voxel_coverage.ply` 时**临时应用
- `--coverage_dilation 0` 恢复严格模式（仅精确击中计数）

### 可视化

每个 `views_{k:04d}/voxel_coverage.ply` 记录当前累积覆盖状态：

- **绿色**：已覆盖的表面体素
- **红色**：尚未覆盖的表面体素

在 MeshLab 中调大点径（`Render → Point size ≥ 5`）可获得"有体积"的方块感。

---

## 复用的现有组件

| 功能 | 文件 |
|---|---|
| GLB 场景渲染 | `src/envs/glb_camera_env.py` — `GLBCameraAgentEnv` |
| 相机内参计算 | `src/envs/voxel_carving.py` — `intrinsics_from_fov` |
| GT 深度反投影 | `src/envs/voxel_carving.py` — `depth_to_world_points` |
| Chamfer distance | `src/utils/pointcloud_eval.py` — `chamfer_distance` |
| Umeyama 对齐 | `src/utils/pointcloud_eval.py` — `_umeyama_alignment` |
| 相机中心提取 | `src/utils/pointcloud_eval.py` — `_camera_centers_from_extrinsics` |
