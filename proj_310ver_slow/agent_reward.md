Modify the existing reward code for the current active 3D reconstruction RL setup.

Important:
- Reuse the current environment, observation pipeline, and training loop.
- Only modify the reward computation and any minimal config plumbing needed.
- Do NOT add reconstruction reward yet.
- This change is for the first-stage training objective only:
  safe movement + useful coverage + reasonable overlap.

==================================================
REWARD DESIGN TO IMPLEMENT
==================================================

Replace the current reward with the following structure:

reward_t =
    step_penalty
  + collision_penalty
  + coverage_reward
  + visibility_reward
  + overlap_reward

where:

1. step_penalty
- a small negative constant every step
- purpose: encourage finishing in fewer views
- default:
  step_penalty = -0.01

2. collision_penalty
- a large negative penalty if collision happens
- default:
  collision_penalty = -1.0 if collision else 0.0
- if the current environment already terminates on collision, keep that behavior
- if collision termination is optional in the current codebase, keep existing termination logic and only modify reward
- do NOT invent new collision detection; reuse the existing collision flag

3. coverage_reward  (main reward)
Use NEW voxel coverage, not total accumulated coverage.

Definitions:
- V_all = all target voxels in the scene/object
- V_t = voxels visible/covered by current privileged depth observation
- U_prev = union of all previously covered voxels before current step

Compute:
- visible_ratio_t = |V_t| / |V_all|
- new_visible_ratio_t = |V_t \ U_prev| / |V_all|

Use:
- coverage_reward = w_new * new_visible_ratio_t

Default:
- w_new = 1.0

Important:
- use incremental new coverage only for the main coverage reward
- do NOT reward total accumulated coverage directly each step

4. visibility_reward  (small auxiliary reward)
Use the current view's visible voxel ratio as a weak positive term.

Compute:
- visible_ratio_t = |V_t| / |V_all|

Use:
- visibility_reward = w_visible * visible_ratio_t

Default:
- w_visible = 0.1

Purpose:
- encourage views that see a reasonably large amount of the scene
- but keep this term much smaller than new coverage reward

5. overlap_reward  (band-pass shaping)
We want overlap that is neither too low nor too high.

Definition:
- overlap_ratio_t = |V_t ∩ U_prev| / max(|V_t|, eps)

Interpretation:
- among voxels covered by the current view, what fraction has already been covered before

Implement a band-pass reward function:

if overlap_ratio_t < overlap_low:
    overlap_score = - (overlap_low - overlap_ratio_t)^2
elif overlap_ratio_t <= overlap_high:
    overlap_score = +1.0
else:
    overlap_score = - (overlap_ratio_t - overlap_high)^2

Then:
- overlap_reward = w_overlap * overlap_score

Default:
- overlap_low = 0.30
- overlap_high = 0.50
- w_overlap = 0.20

Notes:
- this overlap term should be separate from coverage_reward
- do NOT hard-multiply coverage_reward by overlap_ratio
- do NOT zero out coverage reward when overlap is bad
- keep overlap as a shaping term

==================================================
EDGE CASES
==================================================

1. First step
If there is no previous coverage yet:
- U_prev is empty
- overlap_ratio_t should be defined as 0.0
- new_visible_ratio_t should equal visible_ratio_t

2. Empty current visible set
If |V_t| == 0:
- visible_ratio_t = 0.0
- new_visible_ratio_t = 0.0
- overlap_ratio_t = 0.0

3. Numerical stability
- use a small eps for denominators
- avoid NaNs

==================================================
REWARD BREAKDOWN LOGGING
==================================================

Please add logging / info dict entries for:
- reward_total
- reward_step_penalty
- reward_collision
- reward_coverage_new
- reward_visibility
- reward_overlap
- visible_ratio
- new_visible_ratio
- overlap_ratio
- accumulated_coverage_ratio

If the current codebase already has an "info" dict or logging hooks, reuse them.
Do not create a new logging system.

==================================================
CONFIG CHANGES
==================================================

Add minimal config options for reward weights and thresholds.

Suggested config fields:
- reward.step_penalty = -0.01
- reward.collision_penalty = -1.0
- reward.w_new = 1.0
- reward.w_visible = 0.1
- reward.w_overlap = 0.2
- reward.overlap_low = 0.30
- reward.overlap_high = 0.50
- reward.eps = 1e-6

If the current config system already has a reward section, integrate there.
If not, add the smallest possible extension consistent with the existing config style.

==================================================
IMPLEMENTATION NOTES
==================================================

1. Reuse existing voxel carving / point cloud back-projection / voxel coverage code.
2. Do NOT rewrite the voxelization logic.
3. Use the existing privileged depth-based visibility computation already available in the project.
4. Update the reward function only.
5. Keep code clean and localized.

==================================================
PSEUDOCODE
==================================================

Please implement logic equivalent to:

V_t = current_visible_voxels
U_prev = previous_union_of_visible_voxels
V_all = all_target_voxels

visible_ratio_t = len(V_t) / max(len(V_all), eps)
new_visible_ratio_t = len(V_t - U_prev) / max(len(V_all), eps)

if len(V_t) > 0:
    overlap_ratio_t = len(V_t & U_prev) / max(len(V_t), eps)
else:
    overlap_ratio_t = 0.0

if overlap_ratio_t < overlap_low:
    overlap_score = - (overlap_low - overlap_ratio_t) ** 2
elif overlap_ratio_t <= overlap_high:
    overlap_score = 1.0
else:
    overlap_score = - (overlap_ratio_t - overlap_high) ** 2

reward =
    step_penalty
  + (collision_penalty if collision else 0.0)
  + w_new * new_visible_ratio_t
  + w_visible * visible_ratio_t
  + w_overlap * overlap_score

==================================================
VERY IMPORTANT
==================================================

Do not add reconstruction reward in this change.
Do not add any learned reward model.
Do not change the action space.
Do not change the training loop.
Do not change observation definitions.
Do not refactor unrelated files.

Only make the minimum code changes needed to:
- compute the new reward terms
- expose the new config values
- log the reward breakdown

==================================================
OPTIONAL TODO COMMENTS
==================================================

Leave TODO comments only, do not implement:
- stage 2: add low-frequency reconstruction bonus
- compare overlap band [0.30, 0.50] vs [0.40, 0.50]
- compare visibility auxiliary on/off
- compare max_step = 10 vs 20