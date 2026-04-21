# PROJECT_STATE — 项目当前状态

> 更新日期: 2026-04-13

## 当前开发阶段

**阶段：核心训练框架已搭建完成，进入调参/实验阶段。**

## 已完成的模块

### 1. SLAM-Former 流式 Actor-Critic (`slamformer_relative_actor_critic.py`)

- [x] 冻结 DINOv2 encoder + 36 层 BlockRope decoder
- [x] 前端增量处理：每帧更新 KV cache，单帧推理
- [x] 后端全局精炼：每 10 帧重跑 decoder branch=2
- [x] Register token 提取 → LayerNorm → MLP 投影到 256 维
- [x] GaussianActorHead (256→256→128→num_actions×2)
- [x] ValueHead (256→256→128→1)
- [x] Warmup 阶段：前 4 步固定 90° 旋转建立全景认知
- [x] Per-env 流式状态管理（token_map + kv_cache 存 CPU）
- [x] Episode 结束自动清空状态
- [x] `max_map_frames` 截断支持
- [x] bfloat16 混合精度推理

### 2. ManiSkill 环境 (`maniskill_active_mapping_env.py`)

- [x] SAPIEN 渲染后端：GLB / URDF 场景加载
- [x] `xyz_delta_quat` 动作模式：局部坐标系平移 + delta 四元数
- [x] 3D 体素覆盖率追踪（depth → voxel carving → union mask）
- [x] 碰撞/出界检测：`_sweep_for_collision()` 逐步采样碰撞
- [x] 多相机支持（可配置角度列表）
- [x] 奖励函数：step_penalty + crash + coverage + visibility + overlap_band
- [x] 场景随机重采样 (`gleam_resample_scene`)
- [x] 初始位姿随机化 (`init_pose_resample`)
- [x] 位姿约束选项：lock_pitch_roll / lock_roll
- [x] Episode 诊断系统：trace dump、histogram、episode summary

### 3. 训练基础设施 (`train.py`)

- [x] `InstrumentedOnPolicyRunner`: 包装 rsl_rl 的 OnPolicyRunner
- [x] W&B 日志：图像渲染、奖励曲线、episode 结束原因
- [x] 诊断 trace 自动保存到 `diagnostic_traces/`
- [x] 回调系统 (`callbacks.py`)
- [x] tyro CLI 配置解析
- [x] 自适应学习率调度（基于 KL 散度）

### 4. VGGT Baseline (`vggt_relative_actor_critic.py`)

- [x] 冻结 VGGT aggregator 特征提取
- [x] 显式相对位姿嵌入 (`RelativePoseEmbedder`)
- [x] 帧融合：current + mean(history) + max(history)
- [x] 独立训练入口 `train_vggt_relative_simplified.py`

### 5. 辅助模块

- [x] `GymnasiumVecEnv`: SyncVectorEnv + TensorDict 适配 rsl_rl
- [x] `camera_pose_update.py`: 四元数归一化、局部 delta 位姿应用、yaw/pitch 投影
- [x] `voxel_carving.py`: 深度图 → 世界坐标点 → 体素占据
- [x] `gleam_dataset.py`: GLEAM 数据集加载器
- [x] `rendering.py`: 深度图 + 俯视图可视化

## 关键实现细节

### Pose 流转

```
环境内部:
  _position [3]            世界坐标 (Z-up)
  _camera_quat_wxyz [4]    相机朝向四元数 (w,x,y,z)

观测输出 obs_pose [7]:
  rel_pos = (position - init_position) / state_scale
  obs_pose = [rel_x, rel_y, rel_z, qw, qx, qy, qz]

动作输入 (xyz_delta_quat) [7]:
  action = [tx_local, ty_local, tz_local, dqw, dqx, dqy, dqz]
  translation_world = rotate(current_quat, action[:3] * action_scale)
  next_pos = prev_pos + translation_world
  next_quat = current_quat ⊗ delta_quat
```

### SLAMFormer 与环境交互

```
每步:
  1. env.step(action) → obs (含 obs_rgb_tensor)
  2. policy.act(obs):
     a. feature_extractor.forward(images)   # 流式更新 token_map + kv_cache
     b. actor_head(visual_feat) → sample action
     c. 若在 warmup 阶段 → 覆盖为固定旋转动作
  3. env 检测碰撞 → 计算体素覆盖 → 返回 reward

episode 结束:
  1. policy.reset(dones) → 清空对应环境的 _EnvStreamState
  2. env.reset() → 重新加载场景 / 随机初始位姿
```

### 梯度流向

```
冻结 (requires_grad=False):
  ├── DINOv2 encoder
  ├── BlockRope decoder (36 层)
  └── register_token

可训练:
  ├── reg_norm (LayerNorm, 2048)
  ├── visual_proj (MLP: 10240→512→256)
  ├── actor_head (MLP: 256→256→128→num_actions×2)
  └── value_head (MLP: 256→256→128→1)
```

## 配置默认值速查

| 配置项 | 默认值 | 位置 |
|--------|--------|------|
| `num_envs` | 1 | `EnvConfig` |
| `max_steps` | 100 | `EnvConfig` |
| `action_mode` | `xyz_delta_quat` | `EnvConfig` |
| `action_scale` | 0.5 | `EnvConfig` |
| `step_reward` | -0.01 | `EnvConfig` |
| `crash_penalty` | -1.0 | `EnvConfig` |
| `reward_w_new` | 1.0 | `EnvConfig` |
| `reward_w_visible` | 0.1 | `EnvConfig` |
| `reward_w_overlap` | 0.2 | `EnvConfig` |
| `overlap_band` | [0.30, 0.50] | `EnvConfig` |
| `visual_dim` | 256 | `SLAMFormerRelativePolicyConfig` |
| `backend_every` | 10 | `SLAMFormerRelativePolicyConfig` |
| `freeze_backbone` | True | `SLAMFormerRelativePolicyConfig` |
| `slamformer_input_size` | 518 | `SLAMFormerRelativePolicyConfig` |
| `warmup_steps` | 4 | `SLAMFormerRelativeActorCritic` (硬编码) |
| `clip_param` | 0.2 | `AlgorithmConfig` |
| `learning_rate` | 3e-4 | `AlgorithmConfig` |
| `num_steps_per_env` | 128 | `RunnerConfig` |
| `max_iterations` | 20000 | `RunnerConfig` |

## 已知限制与注意事项

1. **单进程环境**: 使用 `SyncVectorEnv`，多环境串行执行，`num_envs` 较大时瓶颈在环境渲染
2. **显存消耗**: SLAM-Former decoder 36 层 + DINOv2 encoder 即使冻结仍占用大量显存，token_map/kv_cache 存 CPU 缓解
3. **Warmup 步数硬编码**: `warmup_steps=4` 写死在 `__init__` 中，未暴露为配置项
4. **Backend 精炼开销**: 每 10 帧全量重跑 decoder 对长 episode 有显著时间开销
5. **SLAMFormer 不使用位姿观测**: 依赖隐式跨帧注意力学习空间关系，与 VGGT baseline 的显式位姿方案形成对比
