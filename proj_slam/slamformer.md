# SLAM-Former 流式 Actor-Critic 集成

## 概述

用冻结的 SLAM-Former（DINOv2 ViT-L/14 编码器 + 36层 BlockRope 解码器）作为流式视觉骨干。

**纯视觉状态**——不使用显式位姿编码。SLAM-Former 的 cross-frame attention 从纯视觉 token 中隐式学习帧间空间关系（这也是它能从 decoder token 预测相机位姿的原因），所以 register tokens 已同时包含"看到了什么"和"帧间相对位置"信息。

## 架构

```text
流式 SLAM-Former（冻结）:
  Step 0: 所有 S 帧 → DINOv2 → 36层 decode → KV cache + token_map
  Step t: 1 帧新帧  → DINOv2 → 增量 decode（KV cache）→ append token_map
  每 backend_every 步: backendT 重新 decode（双向 attention 全局优化）

状态提取:
  latest frame register tokens [5, 2048]
    → LayerNorm → mean+max pool → visual_proj MLP
    → [B, visual_dim=256]

Actor/Critic:
  [B, 256] → GaussianActorHead → (delta_pos, delta_yaw, delta_pitch)
  [B, 256] → ValueHead → scalar
```

## 设计决策

### 为什么不用 HistoryPooling

VGGT 版本需要 HistoryPooling 因为 VGGT 对每帧独立编码，无跨帧上下文。SLAM-Former 的 decoder 奇数层做 cross-frame global attention，最新帧的 token 已 attend 到所有历史帧，不需要额外时序聚合。

### 为什么不用显式位姿编码

SLAM-Former 不接收任何位姿输入，却能从 decoder token 预测相机位姿（CameraHead）。说明 cross-frame attention 已从视觉对应中隐式学到帧间空间关系。register tokens 作为帧级摘要 token，已编码这些信息。

### 为什么用 register tokens 而非 patch tokens

- Register tokens（5个/帧）是模型内置的帧级聚合 token
- 经过 backend 双向 attention 后携带全局上下文
- 比 patch tokens（1369个/帧）紧凑得多
- 避免了额外的 spatial pooling

## 流式 Pipeline 细节

```text
前端（frontendT, branch=1）:
  新帧 encoder output → 36层 decode → 因果掩码（帧 i 只看 <= i）
  KV cache 累积历史帧的 Key/Value

后端（backendT, branch=2）:
  token_map[:, :, :1024]（前半维）→ 36层 decode → 无掩码（双向 attention）
  所有帧互相可见 → 全局一致的精炼特征

Episode 结束:
  reset(dones) → 清除 token_map + KV cache
```

## 文件变更

| 文件 | 变更 |
| ---- | ---- |
| `src/learning/slamformer_relative_actor_critic.py` | **新建。** 流式特征提取器 + 纯视觉 actor-critic |
| `src/learning/config.py` | 新增 `SLAMFormerRelativePolicyConfig` + `build_train_cfg` 分支 |
| `src/learning/policy.py` | 导入 + 注册 `SLAMFormerRelativeActorCritic` |
| `src/scripts/train.py` | 添加到 `policies_need_xyz_delta_quat` 集合 |

## 配置

```yaml
policy:
  class_name: "SLAMFormerRelativeActorCritic"
  slamformer_relative:
    slam_root: "/path/to/SLAM-Former"
    slam_ckpt_path: "/path/to/checkpoint.ckpt"
    slamformer_input_size: 518
    visual_dim: 256
    freeze_backbone: true
    use_patch_mean_pool: true
    use_patch_max_pool: true
    backend_every: 10         # 0 = 禁用后端
    max_map_frames: 0         # 0 = 无限制累积
```

## 梯度流

```text
冻结（no_grad）: DINOv2 → 36层 decoder → token_map（detached）
可训练（grads）: reg_norm（LayerNorm）→ pool → visual_proj（MLP）→ actor/critic heads
```

## 内存

Per-env CPU 内存随累积帧数增长（token_map + KV cache）。用 `max_map_frames` 限制，超限时截断并重建 cache。
