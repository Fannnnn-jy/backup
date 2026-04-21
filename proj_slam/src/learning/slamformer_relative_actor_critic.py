"""
SLAM-Former streaming actor-critic (pure visual state).

Uses frozen SLAM-Former (DINOv2 + 36-layer BlockRope decoder) as streaming
visual backbone.  State = latest frame's register tokens only — no explicit
pose embedding, since SLAM-Former's cross-frame attention implicitly learns
spatial relationships between frames.

Pipeline:
  Streaming decoder → latest frame register tokens [5, 2048]
    → LayerNorm → mean/max pool → visual_proj → [B, visual_dim]
    → GaussianActorHead → action
    → ValueHead → value
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


def _build_gelu_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianActorHead(nn.Module):
    """Small Gaussian policy head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        hidden_dims = list(int(v) for v in hidden_dims)
        if len(hidden_dims) != 2:
            raise ValueError(f"GaussianActorHead expects exactly 2 hidden dims, got {hidden_dims}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], int(action_dim) * 2),
        )

    def forward(self, state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state_feat)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return mean, log_std


class ValueHead(nn.Module):
    """Small critic head."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        hidden_dims = list(int(v) for v in hidden_dims)
        if len(hidden_dims) != 2:
            raise ValueError(f"ValueHead expects exactly 2 hidden dims, got {hidden_dims}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.net(state_feat)


# ---------------------------------------------------------------------------
#  Per-environment streaming state
# ---------------------------------------------------------------------------

def _kv_to_device(kv_cache: list, device) -> list:
    """Move KV cache entries to device. Each entry is a (K, V) tuple or single tensor or None."""
    out = []
    for kv in kv_cache:
        if kv is None:
            out.append(None)
        elif isinstance(kv, tuple):
            out.append(tuple(t.to(device) for t in kv))
        else:
            out.append(kv.to(device))
    return out


@dataclass
class _EnvStreamState:
    token_map: torch.Tensor | None = None   # [N_acc, hw, 2048] on CPU
    kv_cache: list | None = None             # per-global-layer KV on CPU
    n_frames: int = 0
    frames_since_backend: int = 0
    H: int = 0
    W: int = 0


# ---------------------------------------------------------------------------
#  Streaming feature extractor
# ---------------------------------------------------------------------------

class SLAMFormerFeatureExtractor(nn.Module):
    """Streaming SLAM-Former visual backbone.

    Returns the latest frame's register tokens (5 per frame, 2048-dim),
    pooled and projected to ``visual_dim``.
    """

    def __init__(
        self,
        *,
        visual_dim: int = 256,
        use_layernorm: bool = True,
        freeze_backbone: bool = True,
        slam_root: str = "",
        slam_ckpt_path: str = "",
        slamformer_input_size: int = 518,
        backend_every: int = 10,
        max_map_frames: int = 0,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.slamformer_input_size = int(slamformer_input_size)
        self.backend_every = int(backend_every)
        self.max_map_frames = int(max_map_frames)
        self.forward_call_count = 0

        # --- Load SLAM-Former ---
        slam_root = os.path.expanduser(slam_root)
        if slam_root and slam_root not in sys.path:
            sys.path.insert(0, slam_root)
        slam_src = os.path.join(slam_root, "src")
        if os.path.isdir(slam_src) and slam_src not in sys.path:
            sys.path.insert(0, slam_src)
        croco = os.path.join(slam_src, "croco")
        if os.path.isdir(croco) and croco not in sys.path:
            sys.path.insert(0, croco)

        from slamformer.models.slamformer import SLAMFormer

        torch.backends.cuda.enable_flash_sdp(False)

        model = SLAMFormer()
        if slam_ckpt_path:
            ckpt_raw = torch.load(slam_ckpt_path, map_location="cpu", weights_only=False)
            ckpt = ckpt_raw.get("model", ckpt_raw)
            model.load_state_dict(ckpt, strict=False)

        self.encoder = model.encoder
        self.decoder_blocks = model.decoder
        self.register_token = model.register_token
        self.patch_start_idx = model.patch_start_idx  # 5
        self.patch_size = model.patch_size
        self.pos_type = model.pos_type
        self.dec_embed_dim = model.dec_embed_dim
        self.rope = model.rope
        self.position_getter = model.position_getter
        # Keep reference for sharing with reconstruction model
        self._slamformer_model = model

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # Projection: register tokens flatten → visual_dim
        decoder_out_dim = 2 * self.dec_embed_dim  # 2048
        flatten_dim = self.patch_start_idx * decoder_out_dim  # 5 * 2048 = 10240
        self.reg_norm = nn.LayerNorm(decoder_out_dim) if use_layernorm else nn.Identity()
        self.visual_proj = _build_gelu_mlp(flatten_dim, max(512, self.visual_dim * 2), self.visual_dim)

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            for p in self.decoder_blocks.parameters():
                p.requires_grad_(False)
            self.decoder_blocks.eval()
            self.register_token.requires_grad_(False)
            for p in self.reg_norm.parameters():
                p.requires_grad_(False)
            for p in self.visual_proj.parameters():
                p.requires_grad_(False)

        self._env_states: dict[int, _EnvStreamState] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
            self.decoder_blocks.eval()
        return self

    def reset_envs(self, env_indices: Sequence[int]) -> None:
        for idx in env_indices:
            state = self._env_states.pop(idx, None)
            if state is not None:
                state.token_map = None
                state.kv_cache = None

    # ------------------------------------------------------------------
    #  Image helpers
    # ------------------------------------------------------------------

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.slamformer_input_size <= 0:
            return images
        _, _, _, h, w = images.shape
        if h == self.slamformer_input_size and w == self.slamformer_input_size:
            return images
        bsz, seq = images.shape[:2]
        flat = images.reshape(bsz * seq, *images.shape[-3:])
        flat = F.interpolate(
            flat,
            size=(self.slamformer_input_size, self.slamformer_input_size),
            mode="bilinear", align_corners=False,
        )
        return flat.reshape(bsz, seq, 3, self.slamformer_input_size, self.slamformer_input_size)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames = (frames - self.image_mean.to(frames.device)) / self.image_std.to(frames.device)
        hidden = self.encoder(frames, is_training=False)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        return hidden

    # ------------------------------------------------------------------
    #  Stateless decode helpers
    # ------------------------------------------------------------------

    def _make_pos(self, BN: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        pos = self.position_getter(BN, H // self.patch_size, W // self.patch_size, device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(BN, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        return pos

    def _decode_frontend(self, hidden_I, N, H, W, kv_cache=None):
        BN, _, C = hidden_I.shape
        B = BN // N

        reg = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([reg, hidden_I], dim=1)
        hw = hidden.shape[1]
        pos = self._make_pos(B * N, H, W, hidden.device)

        n_global = len(self.decoder_blocks) // 2
        if kv_cache is None:
            kv_cache = [None] * n_global

        final_output: list[torch.Tensor] = []
        n_blocks = len(self.decoder_blocks)

        for i, blk in enumerate(self.decoder_blocks):
            if i % 2 == 0:
                hidden = hidden.reshape(B * N, hw, -1)
                pos_in = pos.reshape(B * N, hw, -1)
                hidden = blk(hidden, xpos=pos_in, N=N, branch=1)
            else:
                hidden = hidden.reshape(B, N * hw, -1)
                pos_in = pos.reshape(B, N * hw, -1)
                kv_idx = i // 2
                hidden, kv_new = blk(
                    hidden, xpos=pos_in, N=N, branch=1,
                    global_=True, kvcache=kv_cache[kv_idx],
                    use_cache=True, idx=None,
                )
                kv_cache[kv_idx] = kv_new

            if i + 1 in (n_blocks - 1, n_blocks):
                final_output.append(hidden.reshape(B * N, hw, -1))

        decoder_out = torch.cat([final_output[0], final_output[1]], dim=-1)
        return decoder_out, kv_cache

    def _decode_backend(self, hidden_F, N, H, W):
        BN, hw, C2 = hidden_F.shape
        B = 1
        hidden = hidden_F[:, :, : C2 // 2]
        pos = self._make_pos(B * N, H, W, hidden.device)

        n_global = len(self.decoder_blocks) // 2
        kv_cache: list[torch.Tensor | None] = [None] * n_global

        final_output: list[torch.Tensor] = []
        n_blocks = len(self.decoder_blocks)

        for i, blk in enumerate(self.decoder_blocks):
            if i % 2 == 0:
                hidden = hidden.reshape(B * N, hw, -1)
                pos_in = pos.reshape(B * N, hw, -1)
                hidden = blk(hidden, xpos=pos_in, N=N, branch=2)
            else:
                hidden = hidden.reshape(B, N * hw, -1)
                pos_in = pos.reshape(B, N * hw, -1)
                kv_idx = i // 2
                hidden, kv_new = blk(
                    hidden, xpos=pos_in, N=N, branch=2,
                    global_=True, kvcache=kv_cache[kv_idx],
                    use_cache=True, idx=None,
                )
                kv_cache[kv_idx] = kv_new

            if i + 1 in (n_blocks - 1, n_blocks):
                final_output.append(hidden.reshape(B * N, hw, -1))

        refined = torch.cat([final_output[0], final_output[1]], dim=-1)
        return refined, kv_cache

    # ------------------------------------------------------------------
    #  Per-env streaming
    # ------------------------------------------------------------------

    def _process_env_initial(self, frames, env_idx, H, W):
        S = frames.shape[0]
        state = _EnvStreamState(H=H, W=W)
        self._env_states[env_idx] = state

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hidden_I = self._encode_frames(frames)
            decoder_out, kv_cache = self._decode_frontend(hidden_I, N=S, H=H, W=W)

        state.token_map = decoder_out
        state.kv_cache = kv_cache
        state.n_frames = S

    def _process_env_incremental(self, new_frame, env_idx):
        state = self._env_states[env_idx]

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hidden_I = self._encode_frames(new_frame.unsqueeze(0))
            new_out, kv_cache = self._decode_frontend(
                hidden_I, N=1, H=state.H, W=state.W, kv_cache=state.kv_cache
            )

        state.token_map = torch.cat([state.token_map, new_out], dim=0)
        state.kv_cache = kv_cache
        state.n_frames += 1
        state.frames_since_backend += 1

        if self.backend_every > 0 and state.frames_since_backend >= self.backend_every:
            self._run_backend(state)

        if self.max_map_frames > 0 and state.n_frames > self.max_map_frames:
            self._truncate_map(state)

    def _run_backend(self, state):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            refined, kv_cache = self._decode_backend(
                state.token_map, N=state.n_frames, H=state.H, W=state.W
            )
        state.token_map = refined
        state.kv_cache = kv_cache
        state.frames_since_backend = 0

    def _truncate_map(self, state):
        keep = self.max_map_frames
        state.token_map = state.token_map[-keep:]
        state.n_frames = keep
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, kv_cache = self._decode_backend(
                state.token_map, N=keep, H=state.H, W=state.W
            )
        state.kv_cache = kv_cache
        state.frames_since_backend = 0

    # ------------------------------------------------------------------
    #  Forward — [B, visual_dim] from latest frame register tokens
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, S, 3, H, W]

        Returns:
            [B, visual_dim]
        """
        if images.dim() != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")

        self.forward_call_count += 1
        B, S = images.shape[:2]
        images = self._resize_images(images)
        _, _, _, H, W = images.shape
        device = images.device

        if self.freeze_backbone:
            self.encoder.eval()
            self.decoder_blocks.eval()

        for b in range(B):
            state = self._env_states.get(b)
            if state is None:
                self._process_env_initial(images[b], env_idx=b, H=H, W=W)
            else:
                self._process_env_incremental(images[b, -1], env_idx=b)

        # Latest frame register tokens: [B, 5, 2048]
        all_reg = []
        for b in range(B):
            reg = self._env_states[b].token_map[-1, :self.patch_start_idx]
            all_reg.append(reg)

        all_reg = torch.stack(all_reg).float()  # [B, 5, 2048]
        all_reg = self.reg_norm(all_reg)         # [B, 5, 2048]
        flat = all_reg.reshape(all_reg.shape[0], -1)  # [B, 5*2048=10240]

        return self.visual_proj(flat)  # [B, visual_dim]


# ---------------------------------------------------------------------------
#  Actor-Critic: pure visual state → actor / critic
# ---------------------------------------------------------------------------

class SLAMFormerRelativeActorCritic(nn.Module):
    """Actor-critic with pure visual state from streaming SLAM-Former.

    State = visual_proj(register_tokens) of shape [B, visual_dim].
    No explicit pose embedding — SLAM-Former's cross-frame attention
    implicitly encodes spatial relationships between frames.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (256, 128),
        init_noise_std: float = 0.2,
        visual_dim: int = 256,
        freeze_backbone: bool = True,
        use_layernorm: bool = True,
        slam_root: str = "",
        slam_ckpt_path: str = "",
        slamformer_input_size: int = 518,
        backend_every: int = 10,
        max_map_frames: int = 0,
        image_obs_key: str = "obs_rgb_tensor",
        activation: str = "gelu",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "SLAMFormerRelativeActorCritic.__init__ got unexpected arguments: "
                + str(kwargs.keys())
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.num_actions = int(num_actions)
        self.image_obs_key = self._resolve_image_key(obs, obs_groups.get("policy", []), image_obs_key)
        self.activation = activation
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)

        self.feature_extractor = SLAMFormerFeatureExtractor(
            visual_dim=visual_dim,
            use_layernorm=use_layernorm,
            freeze_backbone=freeze_backbone,
            slam_root=slam_root,
            slam_ckpt_path=slam_ckpt_path,
            slamformer_input_size=slamformer_input_size,
            backend_every=backend_every,
            max_map_frames=max_map_frames,
        )

        # State = visual_dim directly
        state_dim = int(visual_dim)
        self.actor_head = GaussianActorHead(state_dim, self.num_actions, actor_hidden_dims)
        self.value_head = ValueHead(state_dim, critic_hidden_dims)
        self.distribution: Normal | None = None

        final_linear = self.actor_head.net[-1]
        if isinstance(final_linear, nn.Linear):
            with torch.no_grad():
                final_linear.bias[self.num_actions:] = torch.log(
                    torch.full((self.num_actions,), float(init_noise_std), dtype=final_linear.bias.dtype)
                )

        # Warmup: first 4 steps per episode are predefined 90° rotations
        self.warmup_steps = 4
        self._env_step_counts: dict[int, int] = {}
        # Each warmup action: no translation + 90° yaw rotation (local z-axis)
        # delta_quat for 90° around z: [cos(45°), 0, 0, sin(45°)]
        c = math.cos(math.pi / 4)  # √2/2
        s = math.sin(math.pi / 4)  # √2/2
        # 4 identical actions: rotate 90° yaw each step (right, back, left, back to front)
        self._warmup_action = torch.tensor(
            [0.0, 0.0, 0.0, c, 0.0, 0.0, s], dtype=torch.float32
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.train(mode)
        return self

    @staticmethod
    def _resolve_image_key(obs, obs_groups, preferred_key):
        if preferred_key in obs:
            return preferred_key
        for key in obs_groups:
            if key in obs and obs[key].dim() >= 5:
                return key
        raise KeyError(f"Could not find image observation key. Preferred={preferred_key}")

    @staticmethod
    def _prepare_image_sequence(images):
        if images.dim() == 5:
            if images.shape[-1] == 3:
                images = images.permute(0, 1, 4, 2, 3)
            elif images.shape[2] != 3:
                raise ValueError(f"Unsupported 5D image shape {tuple(images.shape)}")
        elif images.dim() == 6:
            if images.shape[-1] == 3:
                images = images.permute(0, 1, 2, 5, 3, 4)
            elif images.shape[3] != 3:
                raise ValueError(f"Unsupported 6D image shape {tuple(images.shape)}")
        else:
            raise ValueError(f"Expected image tensor rank 5 or 6, got {tuple(images.shape)}")
        lead_shape = tuple(images.shape[:-4])
        flat = images.reshape(-1, *images.shape[len(lead_shape):])
        return flat, lead_shape

    @staticmethod
    def _restore_leading(x, lead_shape):
        return x.reshape(*lead_shape, *x.shape[1:])

    # ---- Core encoding ----

    def _encode_observation(self, obs, *, return_debug=False):
        images, lead_shape = self._prepare_image_sequence(obs[self.image_obs_key])
        state_feat = self.feature_extractor(images)  # [flat_B, visual_dim]

        if not return_debug:
            return state_feat, lead_shape
        debug = {"visual_feats": self._restore_leading(state_feat, lead_shape)}
        return state_feat, lead_shape, debug

    # ---- Actor / Critic ----

    def actor_from_state(self, state_feat):
        lead_shape = tuple(state_feat.shape[:-1])
        flat = state_feat.reshape(-1, state_feat.shape[-1])
        mean, log_std = self.actor_head(flat)
        return mean.reshape(*lead_shape, -1), log_std.reshape(*lead_shape, -1)

    def value_from_state(self, state_feat):
        lead_shape = tuple(state_feat.shape[:-1])
        flat = state_feat.reshape(-1, state_feat.shape[-1])
        value = self.value_head(flat)
        return value.reshape(*lead_shape, -1)

    def _update_distribution_from_state(self, state_feat):
        mean, log_std = self.actor_from_state(state_feat)
        self.distribution = Normal(mean, torch.exp(log_std))

    # ---- External interface ----

    def forward(self, obs, return_debug=False):
        if return_debug:
            state_feat, lead_shape, debug = self._encode_observation(obs, return_debug=True)
        else:
            state_feat, lead_shape = self._encode_observation(obs, return_debug=False)

        state = self._restore_leading(state_feat, lead_shape)
        action_mean, action_log_std = self.actor_from_state(state)
        value = self.value_from_state(state)
        outputs = {"action_mean": action_mean, "action_log_std": action_log_std, "value": value}
        if return_debug:
            outputs.update(debug)
        return outputs

    def _get_warmup_actions(self, obs) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return warmup actions for envs in warmup phase, None mask for others.

        Returns:
            actions: [B, num_actions] with warmup actions filled in
            warmup_mask: [B] bool tensor, True for envs still in warmup
        """
        images = obs[self.image_obs_key]
        B = images.shape[0]
        device = images.device

        warmup_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            step = self._env_step_counts.get(b, 0)
            if step < self.warmup_steps:
                warmup_mask[b] = True
            self._env_step_counts[b] = step + 1

        if not warmup_mask.any():
            return None, None

        warmup_action = self._warmup_action.to(device=device, dtype=images.dtype)
        actions = warmup_action.unsqueeze(0).expand(B, -1).clone()
        return actions, warmup_mask

    def act(self, obs, **kwargs):
        # Encode observation (always, to feed SLAM-Former streaming)
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        state = self._restore_leading(state_feat, lead_shape)
        self._update_distribution_from_state(state)
        policy_actions = self.distribution.sample()

        # Override with warmup actions for envs in warmup phase
        warmup_actions, warmup_mask = self._get_warmup_actions(obs)
        if warmup_actions is not None:
            flat_policy = policy_actions.reshape(-1, self.num_actions)
            flat_warmup = warmup_actions.reshape(-1, self.num_actions)
            flat_policy[warmup_mask] = flat_warmup[warmup_mask]
            policy_actions = flat_policy.reshape(policy_actions.shape)

        return policy_actions

    def act_inference(self, obs):
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        state = self._restore_leading(state_feat, lead_shape)
        mean, _ = self.actor_from_state(state)

        warmup_actions, warmup_mask = self._get_warmup_actions(obs)
        if warmup_actions is not None:
            flat_mean = mean.reshape(-1, self.num_actions)
            flat_warmup = warmup_actions.reshape(-1, self.num_actions)
            flat_mean[warmup_mask] = flat_warmup[warmup_mask]
            mean = flat_mean.reshape(mean.shape)

        return mean

    def evaluate(self, obs, **kwargs):
        state_feat, lead_shape = self._encode_observation(obs, return_debug=False)
        state = self._restore_leading(state_feat, lead_shape)
        return self.value_from_state(state)

    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @torch.no_grad()
    def encode_visual(self, obs):
        images, lead_shape = self._prepare_image_sequence(obs[self.image_obs_key])
        visual_feats = self.feature_extractor(images)
        return self._restore_leading(visual_feats, lead_shape)

    def encode_trainable_state_from_cached_visual(self, visual_feats, obs):
        """Use cached visual features as state directly (no re-encoding)."""
        return visual_feats

    def act_from_state(self, state_feat, obs=None):
        """Sample action from cached state features."""
        self._update_distribution_from_state(state_feat)
        policy_actions = self.distribution.sample()

        if obs is not None:
            warmup_actions, warmup_mask = self._get_warmup_actions(obs)
            if warmup_actions is not None:
                flat_policy = policy_actions.reshape(-1, self.num_actions)
                flat_warmup = warmup_actions.reshape(-1, self.num_actions)
                flat_policy[warmup_mask] = flat_warmup[warmup_mask]
                policy_actions = flat_policy.reshape(policy_actions.shape)

        return policy_actions

    def evaluate_from_state(self, state_feat):
        """Compute value from cached state features."""
        return self.value_from_state(state_feat)

    def get_visual_encode_call_count(self, reset=False):
        count = int(self.feature_extractor.forward_call_count)
        if reset:
            self.feature_extractor.forward_call_count = 0
        return count

    def update_normalization(self, obs):
        return None

    def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
        if dones is not None and dones.any():
            done_indices = dones.nonzero(as_tuple=True)[0].tolist()
            self.feature_extractor.reset_envs(done_indices)
            for idx in done_indices:
                self._env_step_counts[idx] = 0

    def get_hidden_states(self):
        return None
