Task: implement a simplified VGGT-based RL actor-critic baseline for active 3D reconstruction, while reusing the current project’s existing environment definition, observation reading, rollout/training loop, logging, checkpointing, and config style as much as possible.

Very important constraints
1. Do NOT rewrite or modify the existing environment logic unless absolutely necessary.
2. Do NOT rewrite the existing overall training pipeline unless absolutely necessary.
3. Reuse current env creation, observation reading/parsing, trainer entry points, logging, checkpoint saving, and evaluation flow.
4. Create a new simplified script and new model module, with minimal integration changes.
5. Keep the trainable model lightweight.
6. Freeze the VGGT aggregator completely.

Goal
Implement a new simplified actor-critic model that uses frozen VGGT aggregator features as visual backbone.

Final design choices (fixed for this implementation)
1. Use VGGT aggregator outputs from layers [4, 11, 17, 23].
2. Use patch tokens only. Ignore non-patch tokens.
3. For each selected layer and each frame:
   - patch mean pooling
   - patch max pooling
4. Pose embedding uses relative pose only:
   - each frame pose is represented relative to the latest frame in the sequence
   - no absolute pose embedding in this implementation
5. History fusion is:
   - current token
   - mean(history tokens)
   - max(history tokens)
6. Action output is relative move in the environment’s native action space.
7. Keep everything simple: no transformer in the new head, no DPT dense decoder, no attention pooling.

High-level implementation plan
Create a new lightweight model that:
- calls the frozen VGGT aggregator once on the full image sequence
- reads layers [4, 11, 17, 23]
- extracts patch tokens
- pools patch tokens per frame using mean and max
- concatenates features across layers
- embeds relative pose per frame
- fuses visual and pose features per frame
- summarizes history using current + mean(history) + max(history)
- outputs actor distribution parameters and critic value

Files to add
Please add at least these files:
1. a new model file, for example:
   - models/vggt_relative_actor_critic.py
2. a new simplified training entry script, for example:
   - train_vggt_relative_simplified.py

If the repo structure suggests slightly different paths, follow the existing style.

Do not delete or replace existing files.
Keep changes localized.

==================================================
MODEL SPECIFICATION
==================================================

Implement these modules cleanly and separately.

1. class VGGTFeatureExtractor(nn.Module)
Purpose:
- wrap the frozen VGGT aggregator
- extract pooled per-frame visual features from layers [4,11,17,23]

Behavior:
- input:
  images: [B, S, 3, H, W]
- output:
  visual_feats: [B, S, visual_dim]

Implementation details:
- call the existing VGGT aggregator from the current codebase
- do not use any VGGT head (no DPT head, no camera head, etc.)
- use aggregated_tokens_list from the aggregator
- use the project’s existing way to get patch_start_idx
  - do not hardcode if the repo already has this information available
  - if no helper exists, add a very small adapter utility and document it clearly
- for each selected layer idx in [4, 11, 17, 23]:
  - x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
  - shape expected: [B, S, N_patch, D]
  - apply LayerNorm if helpful / consistent with current code
  - compute:
      mean_pool = mean over patch dimension -> [B, S, D]
      max_pool  = max over patch dimension  -> [B, S, D]
  - concat mean and max -> [B, S, 2D]
- concat the 4 layer features along the last dimension
- project with a small MLP:
  - Linear -> GELU -> Linear
  - output dimension = 256
- freeze all aggregator parameters
- ensure no gradients flow into aggregator
- aggregator should remain in eval mode unless the project already has a specific preferred frozen-backbone handling approach

Suggested constructor args:
- selected_layers=(4,11,17,23)
- visual_dim=256
- use_layernorm=True

2. class RelativePoseEmbedder(nn.Module)
Purpose:
- embed per-frame pose relative to the last frame in the sequence

Behavior:
- input:
  poses: per-frame camera pose from existing env/obs
- output:
  pose_feats: [B, S, pose_dim]

Important:
- adapt to the current project’s pose format instead of forcing a new one
- if the current project uses 4x4 extrinsics, use that
- if it uses R,t separately, adapt accordingly
- keep the rest of the pipeline unchanged

Definition of relative pose:
- let the latest frame index be t = S-1
- for each frame i:
    T_rel_i = inverse(T_t) @ T_i
- for the last frame itself, relative pose should be identity / zero translation

Pose representation:
- relative translation: 3 dims
- relative rotation: 6D rotation representation
- recency / frame index feature:
  - include one simple scalar or embedding that indicates how far frame i is from the latest frame
  - simplest version:
      recency = (i - (S-1)) / max(S-1,1)
  - shape 1 dim
- final raw pose feature per frame:
  [rel_translation(3), rel_rotation_6d(6), recency(1)] = 10 dims

Pose MLP:
- Linear -> GELU -> Linear
- output pose_dim = 64

Please implement a small helper for converting rotation matrices to 6D representation.
If the existing codebase already has such a helper, reuse it.

3. class FrameFusion(nn.Module)
Purpose:
- fuse per-frame visual and pose embeddings

Behavior:
- input:
  visual_feats: [B, S, 256]
  pose_feats:   [B, S, 64]
- output:
  fused_feats:  [B, S, 256]

Implementation:
- concat [visual, pose] -> [B, S, 320]
- Linear -> GELU -> Linear
- output 256 dims

Keep this very small.

4. class HistoryPooling(nn.Module)
Purpose:
- summarize the current state using current + mean(history) + max(history)

Behavior:
- input:
  fused_feats: [B, S, 256]
  optional valid_mask: [B, S] if available in current pipeline
- output:
  state_feat: [B, 256 * 3]

Definition:
- current = fused_feats[:, last_valid_index]
  For the simplest baseline, if all sequences are full and ordered, use the last frame S-1.
  If the current pipeline has a valid_mask or variable-length sequence handling, support it properly.
- history = frames before current
- hist_mean = mean(history)
- hist_max = max(history)

Edge case:
- if history length is 0 (for example S=1):
  - hist_mean = zeros_like(current)
  - hist_max = zeros_like(current)

Important:
- keep this module simple and deterministic
- if padding is used in the current pipeline, respect valid_mask and do not pool padded frames

5. class GaussianActorHead(nn.Module)
Purpose:
- produce relative action distribution

Behavior:
- input:
  state_feat: [B, 768]
- output:
  action distribution parameters

Implementation:
- MLP:
  Linear(768 -> 256)
  GELU
  Linear(256 -> 128)
  GELU
  Linear(128 -> 2 * action_dim)
- split into mean and log_std

Important:
- action_dim must match the existing environment / training pipeline
- do NOT change the environment action space
- this model predicts relative move in the environment’s existing action parameterization
- if the current project already has a standard Gaussian policy head utility, reuse it
- if the current project expects tanh-squashed Gaussian, follow the existing project convention
- if the current project already has action scaling / clipping utilities, reuse them

6. class ValueHead(nn.Module)
Purpose:
- critic value prediction

Behavior:
- input:
  state_feat: [B, 768]
- output:
  value: [B, 1]

Implementation:
- MLP:
  Linear(768 -> 256)
  GELU
  Linear(256 -> 128)
  GELU
  Linear(128 -> 1)

7. class VGGTRelativeActorCritic(nn.Module)
Purpose:
- compose everything together

Forward behavior:
- input: observation dict from current pipeline
- extract images and poses from the existing observation format
- call frozen VGGTFeatureExtractor
- call RelativePoseEmbedder
- fuse frame features
- pool history
- produce actor outputs and value

Please make this class compatible with the current training pipeline’s expected model API.
If the current trainer expects specific method names like:
- forward()
- act()
- evaluate_actions()
- get_value()
please follow the existing project conventions exactly.

==================================================
OBSERVATION / DATA ADAPTER REQUIREMENTS
==================================================

Reuse current env and observation reading.
Do not redefine the env.

Implement only a thin adapter layer inside the new model or training script.

The model should locate from the current observation structure:
- image sequence tensor
- pose sequence tensor
- optional valid mask if present

Expected default image shape:
- [B, S, 3, H, W]

Expected default pose format:
- ideally [B, S, 4, 4]
but adapt to the actual current project format.

If the current repo uses a different field name or nested dict path for images / poses, detect and document it.
Keep naming aligned with the rest of the codebase.

Do not introduce a new dataset format.

==================================================
TRAINING SCRIPT REQUIREMENTS
==================================================

Create a new simplified training entry script that reuses the existing training flow.

Suggested filename:
- train_vggt_relative_simplified.py

Requirements:
1. Reuse current environment factory / env registration
2. Reuse current trainer / PPO or SAC runner / replay buffer logic
3. Reuse current checkpointing and logging
4. Only swap in the new model class and any minimal config needed
5. Keep the script minimal and easy to read

The script should:
- build envs using existing code
- instantiate the new VGGTRelativeActorCritic model
- plug it into the existing trainer
- run training with the existing config mechanism if possible

If the current project uses yaml / argparse / dataclasses for config, follow the existing style.
Do not invent a completely separate config system unless absolutely necessary.

==================================================
CONFIG REQUIREMENTS
==================================================

Add a small config block or dataclass for the new model.

Include at least:
- selected_layers = [4,11,17,23]
- visual_dim = 256
- pose_dim = 64
- fused_dim = 256
- actor_hidden_dims = [256,128]
- critic_hidden_dims = [256,128]
- freeze_aggregator = True
- use_patch_mean_pool = True
- use_patch_max_pool = True

If the project already has a central config system, integrate these settings there minimally.

==================================================
IMPLEMENTATION DETAILS / SHAPE COMMENTS
==================================================

Please include clear tensor shape comments in code, especially:
- aggregator outputs
- patch token extraction
- mean/max pooling over patches
- relative pose construction
- history pooling
- actor/value head outputs

Also add assertions where useful:
- selected layer indices valid
- image tensor rank correct
- pose tensor format valid or converted
- sequence length S >= 1

==================================================
IMPORTANT DESIGN CHOICES TO PRESERVE
==================================================

Please do NOT add the following in this simplified baseline:
- no transformer on top of frame tokens
- no DPT decoder
- no attention pooling over patches
- no absolute pose embedding
- no mixed absolute+relative pose mode
- no camera head / depth head / point head usage
- no finetuning of the VGGT aggregator

This baseline should stay simple.

==================================================
MASK / VARIABLE LENGTH HANDLING
==================================================

If the existing pipeline supports variable-length frame sequences or padded sequences:
- support a valid_mask
- use it for:
  - selecting the last valid frame as current
  - computing mean(history) over valid history only
  - computing max(history) over valid history only
- do not pool padded frames

If the current pipeline always provides fixed S with no padding, keep implementation simple.

==================================================
SMOKE TESTS
==================================================

Add a small smoke test function or minimal unit-like test that:
1. creates fake images [B,S,3,H,W]
2. creates fake poses [B,S,4,4]
3. runs the model forward
4. prints shapes of:
   - visual_feats
   - pose_feats
   - fused_feats
   - state_feat
   - actor outputs
   - value outputs

If the codebase already has a preferred test location, place it there.
Otherwise a small debug function in the new model file is acceptable.

==================================================
CODE QUALITY REQUIREMENTS
==================================================

1. Keep code modular and readable
2. Do not over-engineer
3. Reuse existing utilities wherever possible
4. Add short docstrings
5. Keep changes localized
6. Preserve compatibility with the existing pipeline

==================================================
DELIVERABLES
==================================================

Please produce:
1. the new model module
2. the new simplified training entry script
3. any minimal config integration needed
4. a short note in comments at the top of the new files explaining:
   - this is a simplified VGGT relative-pose baseline
   - it uses layers [4,11,17,23]
   - patch mean+max pooling
   - relative pose only
   - current + mean(history) + max(history)
   - relative action output

==================================================
OPTIONAL TODO COMMENTS FOR FUTURE ABLATIONS
==================================================

Please leave TODO comments, but do not implement them now:
- compare max-only vs mean-only vs mean+max patch pooling
- add small transformer over frame tokens
- compare relative-only vs absolute+relative pose input
- compare separate actor/critic frame fusion vs shared frame fusion