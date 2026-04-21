# proj_slam 主动重建项目工作文档
> 本文档使用简体中文维护

## 项目概述

基于 SLAM-Former 的主动三维重建项目，使用 PPO 强化学习方法训练智能体在未知室内场景中自主探索，最大化三维覆盖率。核心思路：冻结预训练 SLAM-Former 视觉骨干作为流式特征提取器，在其上接轻量 Actor-Critic 头进行策略学习。

## 目录结构

```
proj_slam/
├── src/
│   ├── scripts/
│   │   ├── train.py                              # 主训练入口（PPO + rsl_rl）
│   │   ├── train_vggt_relative_simplified.py      # VGGT baseline 入口
│   │   ├── visualize_active.py                    # 可视化脚本
│   │   └── vggt_reconstruct_raw_frames.py         # VGGT 重建脚本
│   ├── learning/
│   │   ├── slamformer_relative_actor_critic.py    # ★ 核心：SLAM-Former Actor-Critic
│   │   ├── vggt_relative_actor_critic.py          # VGGT baseline Actor-Critic
│   │   ├── config.py                              # 全部配置 dataclass
│   │   ├── policy.py                              # 策略注册 & CNN baseline
│   │   ├── utils.py                               # GymnasiumVecEnv 向量化环境包装
│   │   └── callbacks.py                           # 训练回调
│   ├── envs/
│   │   ├── active_mapping_env.py                  # 环境基类 ActiveMappingEnv
│   │   ├── maniskill_active_mapping_env.py        # ★ ManiSkill/SAPIEN 渲染环境实现
│   │   ├── gleam_dataset.py                       # GLEAM 数据集加载
│   │   ├── glb_camera_env.py                      # GLB 场景相机环境
│   │   ├── voxel_carving.py                       # 深度图 → 体素雕刻
│   │   └── patchify.py                            # 网格 patch 访问标记
│   ├── utils/
│   │   ├── camera_pose_update.py                  # 四元数/位姿变换工具
│   │   ├── rendering.py                           # 可视化渲染工具
│   │   └── math.py                                # 数学工具（四元数↔旋转矩阵）
│   └── preprocess/                                # 数据预处理
├── tests/                                         # 单元测试
├── debug/                                         # 调试输出
└── wandb/                                         # W&B 实验日志
```

## 核心架构

### 1. 网络结构 (`slamformer_relative_actor_critic.py`)

#### 1.1 SLAMFormerFeatureExtractor — 流式视觉骨干

```
输入: RGB 图像 [B, S, 3, H, W]
  ↓ resize 到 518×518
  ↓ ImageNet 归一化
  ↓ 冻结 DINOv2 Encoder → patch tokens                    ← 冻结
  ↓ 36 层 BlockRope Decoder（前端/后端双分支）               ← 冻结
  ↓ 取最新帧的 5 个 register tokens [B, 5, 2048]
  ─────────── 以下为可训练层（Actor/Critic 共享） ───────────
  ↓ LayerNorm (reg_norm, 2048)                             ← 可训练
  ↓ Flatten → [B, 10240]
  ↓ MLP(10240 → 512 → 256) with GELU (visual_proj)        ← 可训练
输出: visual_feat [B, 256]
```

> **注意：** `reg_norm` 和 `visual_proj` 虽然在 `SLAMFormerFeatureExtractor` 类内，
> 但它们**不属于冻结范围**，是 Actor 和 Critic 共享的可训练投影层。
> 冻结仅作用于 `encoder`、`decoder_blocks` 和 `register_token`。

**前端 (Branch 1) — 每帧增量处理：**
- 偶数层：局部自注意力，每帧独立处理 `(B*N, hw, C)`
- 奇数层：全局跨帧注意力，所有帧展开 `(B, N*hw, C)`，带 KV cache
- KV cache 实现跨帧记忆，新帧只需增量计算

**后端 (Branch 2) — 全局精炼：**
- 每 `backend_every=10` 帧触发一次
- 对累积的 token_map 全量重新走一遍 decoder (branch=2)
- 刷新 KV cache，提升长程一致性

**关键设计：无显式位姿输入。** SLAM-Former 的跨帧注意力 + RoPE 位置编码隐式学习了帧间空间关系。

#### 1.2 SLAMFormerRelativeActorCritic — Actor-Critic 头

```
visual_feat [B, 256]
  ├── Actor (GaussianActorHead):
  │     Linear(256→256) → GELU → Linear(256→128) → GELU → Linear(128→num_actions*2)
  │     输出: mean + log_std (log_std clamp [-5, 2])
  │     分布: Normal(mean, exp(log_std))
  └── Critic (ValueHead):
        Linear(256→256) → GELU → Linear(256→128) → GELU → Linear(128→1)
        输出: state value V(s)
```

#### 1.3 Per-Environment 流式状态 (`_EnvStreamState`)

每个环境实例独立维护：
```python
@dataclass
class _EnvStreamState:
    token_map: Tensor | None    # [N_acc, hw, 2048] 累积 token，存 CPU
    kv_cache: list | None       # 每层全局注意力 (K, V)，存 CPU
    n_frames: int               # 已处理帧数
    frames_since_backend: int   # 距上次后端精炼的帧数
    H: int, W: int              # 图像尺寸
```

- `reset_envs(indices)`: episode 结束时清空对应环境的 token_map 和 kv_cache
- `_truncate_map()`: 当 `max_map_frames > 0` 且帧数超限时，只保留最近 N 帧

### 2. Warmup 阶段 (`slamformer_relative_actor_critic.py:495-505`)

每个 episode 开始的前 **4 步**使用固定动作（不走策略网络）：
- 动作：零平移 + 绕 Z 轴旋转 90° 的 delta 四元数 `[0, 0, 0, cos(π/4), 0, 0, sin(π/4)]`
- 效果：原地旋转 4×90°=360°，让 SLAM-Former 建立初始全景认知
- 虽然动作被覆盖，但 SLAM-Former 的 forward 仍然执行（积累 KV cache）
- 通过 `_env_step_counts` 字典跟踪每个环境的步数

### 3. 环境 (`maniskill_active_mapping_env.py`)

#### 3.1 观测空间 (Dict)

| Key | Shape | 说明 |
|-----|-------|------|
| `obs_rgb_tensor` | `[max_depth_frames, 64, 64, 3]` | RGB 图像序列 |
| `obs_depth_tensor` | `[max_depth_frames, 64, 64]` | 深度图序列 |
| `obs_embedding` | `[64, 64]` | 深度编码嵌入 |
| `agent_state` | `[3]` | 归一化绝对位置 `(pos - center) / scale` |
| `obs_pose` | `[7]` | 相对位姿 `[rel_x, rel_y, rel_z, qw, qx, qy, qz]` |

**注意：** SLAMFormerRelativeActorCritic 只使用 `obs_rgb_tensor`，不使用 `agent_state` 和 `obs_pose`。

#### 3.2 动作空间

当前使用 `action_mode = "xyz_delta_quat"` (7D)：
```
action = [tx, ty, tz, dqw, dqx, dqy, dqz]
```
- `tx, ty, tz`: 相机局部坐标系下的平移，乘以 `action_scale=0.5`
- `dqw, dqx, dqy, dqz`: 局部坐标系下的 delta 四元数旋转
- 通过 `apply_local_camera_delta_pose()` 应用到当前位姿
- 可选约束：`lock_camera_pitch_roll` 只保留 yaw，`lock_camera_roll` 保留 yaw+pitch

#### 3.3 位姿表示与更新

环境内部维护：
- `_position`: 世界坐标系下的 3D 位置 `[x, y, z]`
- `_camera_quat_wxyz`: 相机朝向四元数 (w, x, y, z)，世界到相机
- `_init_position`: episode 起始位置（用于计算相对位姿观测）

位姿更新流程（`xyz_delta_quat` 模式）：
```python
current_quat = normalize_quat_wxyz(self._camera_quat_wxyz)
translation_local = action[:3] * action_scale
next_pos, next_quat = apply_local_camera_delta_pose(
    prev_position, current_quat, translation_local, action[3:7]
)
```

### 4. 奖励设计 (`maniskill_active_mapping_env.py:677-744`)

```
reward = step_penalty + crash_penalty + coverage_reward + visibility_reward + overlap_reward
```

| 分项 | 权重 | 计算方式 |
|------|------|----------|
| **Step Penalty** | `step_reward = -0.01` | 每步固定惩罚 |
| **Crash Penalty** | `crash_penalty = -1.0` | 碰撞或出界时触发 |
| **Coverage (新覆盖)** | `reward_w_new = 1.0` | `w × new_visible_count / total_target` |
| **Visibility (可见性)** | `reward_w_visible = 0.1` | `w × visible_count / total_target` |
| **Overlap (重叠控制)** | `reward_w_overlap = 0.2` | `w × overlap_band_score(overlap_ratio)` |

**Overlap Band Score 函数：**
```
overlap_ratio < 0.30 → -(0.30 - ratio)²   (太少重叠，惩罚)
0.30 ≤ ratio ≤ 0.50  → +1.0               (目标区间，奖励)
overlap_ratio > 0.50 → -(ratio - 0.50)²   (太多重叠，惩罚)
```

**覆盖率追踪：**
- `_covered_target_voxels`: 累积已覆盖的 3D 体素 mask (union)
- `_target_voxel_mask()`: 场景的 ground truth 可占据体素
- 当 `accumulated_coverage_ratio >= 1.0` 时 episode 终止
- 体素来源：GLB 场景体素化 或 GLEAM 数据集

### 5. PPO 训练流程 (`train.py`)

#### 5.1 训练循环

```
for iteration in range(max_iterations=20000):
    # Rollout 阶段
    for step in range(num_steps_per_env=128):
        actions = policy.act(obs)              # 含 warmup 覆盖
        obs, rewards, dones, extras = env.step(actions)
        policy.process_env_step(...)           # 存储 transition
    
    # 计算优势
    policy.compute_returns(obs)                # GAE-λ (γ=0.99, λ=0.95)
    
    # PPO 更新
    for epoch in range(num_learning_epochs=10):
        for mini_batch in range(num_mini_batches=4):
            # L = L_clip + 0.5 * L_value - 0.01 * H(π)
            update_policy(clip_param=0.2, lr=3e-4)
```

#### 5.2 PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `clip_param` | 0.2 | PPO 裁剪范围 |
| `value_loss_coef` | 0.5 | 价值函数损失权重 |
| `entropy_coef` | 0.01 | 熵正则权重 |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `schedule` | adaptive | 基于 KL 散度自适应调整 |
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `max_grad_norm` | 0.5 | 梯度裁剪 |

#### 5.3 可训练参数

冻结 SLAM-Former encoder + decoder，只训练：
- `visual_proj`: MLP(10240 → 512 → 256)
- `reg_norm`: LayerNorm(2048)
- `actor_head`: GaussianActorHead MLP
- `value_head`: ValueHead MLP

### 6. 配置系统 (`config.py`)

采用 `dataclass` + `tyro` CLI 解析：

```python
@dataclass
class TrainConfig:
    env: EnvConfig                    # 环境配置
    policy: PolicyConfig              # 策略配置
    algorithm: AlgorithmConfig        # PPO 超参数
    runner: RunnerConfig              # 训练循环配置
    pointcloud_eval: PointCloudEvalConfig  # 未启用，stage1 不使用
    log_dir: str = "runs/active_mapping"
    logger: str = "wandb"
    wandb_project: str = "slamformer_stage1"
```

关键子配置：

```python
@dataclass
class SLAMFormerRelativePolicyConfig:
    slam_root: str              # SLAM-Former 仓库路径
    slam_ckpt_path: str         # 预训练权重路径
    slamformer_input_size: int = 518
    visual_dim: int = 256       # 视觉特征维度
    actor_hidden_dims: [256, 128]
    critic_hidden_dims: [256, 128]
    freeze_backbone: bool = True
    use_layernorm: bool = True
    backend_every: int = 10     # 后端精炼间隔
    max_map_frames: int = 0     # 0=无限制
```

### 7. VGGT Baseline (`vggt_relative_actor_critic.py`)

对比方案，使用冻结 VGGT 作为视觉骨干：
- 提取层 [4, 11, 17, 23] 的 aggregator features
- Patch mean + max pooling
- **显式使用相对位姿输入** (`RelativePoseEmbedder`)：当前帧相对历史帧的 SE(3) 变换 → 6D rotation + 3D translation → MLP embedding
- `HistoryPooling`: current + mean(history) + max(history) 聚合
- 与 SLAMFormer 方案的关键区别：VGGT 方案需要显式位姿输入

### 8. 环境类继承关系

```
gym.Env
  └── ActiveMappingEnv (active_mapping_env.py)
        ├── 定义观测/动作空间、碰撞检测、网格标记
        ├── _build_observation() → dict obs
        └── ManiSkillActiveMappingEnv (maniskill_active_mapping_env.py)
              ├── SAPIEN 场景渲染（GLB/URDF 加载）
              ├── reset() / step() 完整实现
              ├── _compute_reward_breakdown() 奖励计算
              ├── 3D 体素覆盖率追踪
              └── 碰撞/出界检测 via _sweep_for_collision()
```

向量化包装：
```
GymnasiumVecEnv (learning/utils.py)
  └── gym.vector.SyncVectorEnv([ManiSkillActiveMappingEnv × num_envs])
  └── TensorDict 接口适配 rsl_rl
```

## 代码约定

- 四元数统一使用 **wxyz** 格式
- 位姿 7D 向量格式：`[tx, ty, tz, qw, qx, qy, qz]`
- 坐标系：Z-up 世界坐标系，GLB 场景加载时从 Y-up 旋转
- 所有 SLAM-Former 状态存储在 CPU，推理时临时移到 GPU
- 使用 `bfloat16` 混合精度推理
