from __future__ import annotations

from typing import Any, Callable, List

import torch
import torch.nn.functional as torch_f
from torch import nn
from torch.distributions import Normal

try:
    from rsl_rl.modules import ActorCriticCNN
    from rsl_rl.networks import CNN, EmpiricalNormalization, Memory
except ImportError:  # pragma: no cover
    ActorCriticCNN = None
    CNN = None
    EmpiricalNormalization = None
    Memory = None


def _activation_from_name(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], activation: str) -> nn.Sequential:
    act = _activation_from_name(activation)
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class _LightResidualConvUnit(nn.Module):
    """Lightweight residual conv unit used by DPT-like fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


class _LightFeatureFusionBlock(nn.Module):
    """Lightweight feature fusion block inspired by DPT FeatureFusionBlock."""

    def __init__(self, channels: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.res1 = _LightResidualConvUnit(channels)
        else:
            self.res1 = None
        self.res2 = _LightResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, size: tuple[int, int] | None = None):
        out = x
        if self.has_residual and residual is not None and self.res1 is not None:
            out = out + self.res1(residual)
        out = self.res2(out)
        if size is None:
            out = torch_f.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        else:
            out = torch_f.interpolate(out, size=size, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        return out


if ActorCriticCNN is not None:

    class DepthActorCriticCNN(ActorCriticCNN):
        """Custom Actor-Critic that consumes depth stacks as CNN inputs."""

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            **kwargs,
        ):
            super().__init__(
                obs=obs,
                obs_groups=obs_groups,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
                init_noise_std=init_noise_std,
                state_dependent_std=state_dependent_std,
                **kwargs,
            )

            if self.actor_cnns is None or self.critic_cnns is None:
                raise ValueError("CNN configs must be provided for depth inputs.")

            actor_input_dim = sum(cnn.output_dim for cnn in self.actor_cnns.values())
            critic_input_dim = sum(cnn.output_dim for cnn in self.critic_cnns.values())

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(critic_input_dim, 1, critic_hidden_dims, activation)

        def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)

            if self.state_dependent_std:
                mean_and_std = self.actor(cnn_enc)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.exp(log_std)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            else:
                mean = self.actor(cnn_enc)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

            self.distribution = Normal(mean, std)

        def get_actor_obs(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            obs_dict_2d = {group: obs[group] for group in self.actor_obs_groups_2d}
            return None, obs_dict_2d

        def get_critic_obs(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            obs_dict_2d = {group: obs[group] for group in self.critic_obs_groups_2d}
            return None, obs_dict_2d

        def evaluate(self, obs, **kwargs: dict[str, Any]) -> torch.Tensor:
            _, cnn_obs = self.get_critic_obs(obs)
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            return self.critic(cnn_enc)

        def act_inference(self, obs) -> torch.Tensor:
            _, cnn_obs = self.get_actor_obs(obs)
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)

            if self.state_dependent_std:
                return self.actor(cnn_enc)[..., 0, :]
            return self.actor(cnn_enc)

    class DepthActorCriticCNNRecurrent(nn.Module):
        """Recurrent Actor-Critic with CNN depth encoder + RNN memory."""

        is_recurrent: bool = True

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            rnn_type: str = "lstm",
            rnn_hidden_dim: int = 256,
            rnn_num_layers: int = 1,
            **kwargs,
        ):
            if kwargs:
                print(
                    "DepthActorCriticCNNRecurrent.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if Memory is None or CNN is None or EmpiricalNormalization is None:
                raise ImportError("rsl_rl is required to use DepthActorCriticCNNRecurrent")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            (
                self.actor_obs_groups_1d,
                self.actor_obs_groups_2d,
                num_actor_obs_1d,
                actor_in_dims_2d,
                actor_in_channels_2d,
            ) = self._split_obs_groups(obs, obs_groups["policy"])
            (
                self.critic_obs_groups_1d,
                self.critic_obs_groups_2d,
                num_critic_obs_1d,
                critic_in_dims_2d,
                critic_in_channels_2d,
            ) = self._split_obs_groups(obs, obs_groups["critic"])

            self.actor_cnns, actor_enc_dim = self._build_cnns(
                self.actor_obs_groups_2d, actor_in_dims_2d, actor_in_channels_2d, actor_cnn_cfg
            )
            self.critic_cnns, critic_enc_dim = self._build_cnns(
                self.critic_obs_groups_2d, critic_in_dims_2d, critic_in_channels_2d, critic_cnn_cfg
            )

            actor_input_dim = num_actor_obs_1d + actor_enc_dim
            critic_input_dim = num_critic_obs_1d + critic_enc_dim

            self.memory_a = Memory(actor_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
            self.memory_c = Memory(critic_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(rnn_hidden_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(rnn_hidden_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and num_actor_obs_1d > 0
            self.critic_obs_normalization = critic_obs_normalization and num_critic_obs_1d > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]):
            obs_groups_1d = []
            obs_groups_2d = []
            in_dims_2d = []
            in_channels_2d = []
            num_obs_1d = 0
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 4:  # B, C, H, W
                    obs_groups_2d.append(obs_group)
                    in_dims_2d.append(obs_shape[2:4])
                    in_channels_2d.append(obs_shape[1])
                elif len(obs_shape) == 2:  # B, C
                    obs_groups_1d.append(obs_group)
                    num_obs_1d += obs_shape[-1]
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            return obs_groups_1d, obs_groups_2d, num_obs_1d, in_dims_2d, in_channels_2d

        @staticmethod
        def _resolve_cnn_cfg(obs_groups_2d: List[str], cnn_cfg: dict | None):
            if not obs_groups_2d:
                return {}
            if cnn_cfg is None:
                raise ValueError("CNN configs must be provided for depth inputs.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in obs_groups_2d}
            if len(cnn_cfg) != len(obs_groups_2d):
                raise ValueError("The number of CNN configs must match the number of 2D observations.")
            return cnn_cfg

        def _build_cnns(
            self,
            obs_groups_2d: List[str],
            in_dims_2d: List[tuple[int, int]],
            in_channels_2d: List[int],
            cnn_cfg: dict | None,
        ):
            if not obs_groups_2d:
                return None, 0
            resolved_cfg = self._resolve_cnn_cfg(obs_groups_2d, cnn_cfg)
            cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=in_dims_2d[idx],
                    input_channels=in_channels_2d[idx],
                    **resolved_cfg[obs_group],
                )
                if cnns[obs_group].output_channels is None:
                    encoding_dim += int(cnns[obs_group].output_dim)
                else:
                    raise ValueError("CNN outputs must be flattened before passing to the RNN.")
            return cnns, encoding_dim

        def reset(self, dones: torch.Tensor | None = None) -> None:
            self.memory_a.reset(dones)
            self.memory_c.reset(dones)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def _encode_1d_obs(self, obs, obs_groups_1d: List[str]) -> torch.Tensor | None:
            if not obs_groups_1d:
                return None
            obs_list = [obs[group] for group in obs_groups_1d]
            if len(obs_list) == 1:
                return obs_list[0]
            return torch.cat(obs_list, dim=-1)

        def _encode_2d_obs(self, obs, obs_groups_2d: List[str], cnns: nn.ModuleDict | None) -> torch.Tensor | None:
            if not obs_groups_2d or cnns is None:
                return None
            encodings = []
            for obs_group in obs_groups_2d:
                obs_tensor = obs[obs_group]
                if obs_tensor.dim() == 4:
                    enc = cnns[obs_group](obs_tensor)
                elif obs_tensor.dim() == 5:
                    time_steps, batch = obs_tensor.shape[:2]
                    flat = obs_tensor.reshape(time_steps * batch, *obs_tensor.shape[2:])
                    enc = cnns[obs_group](flat)
                    enc = enc.reshape(time_steps, batch, -1)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_tensor.shape}")
                encodings.append(enc)
            return torch.cat(encodings, dim=-1)

        def _encode_obs(
            self,
            obs,
            obs_groups_1d: List[str],
            obs_groups_2d: List[str],
            cnns: nn.ModuleDict | None,
            normalizer: nn.Module,
        ) -> torch.Tensor:
            mlp_obs = self._encode_1d_obs(obs, obs_groups_1d)
            if mlp_obs is not None:
                mlp_obs = normalizer(mlp_obs)
            cnn_obs = self._encode_2d_obs(obs, obs_groups_2d, cnns)
            if mlp_obs is None:
                if cnn_obs is None:
                    raise ValueError("No observations provided for the policy.")
                return cnn_obs
            if cnn_obs is None:
                return mlp_obs
            return torch.cat([mlp_obs, cnn_obs], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        def _update_distribution(self, rnn_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(rnn_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(rnn_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
            obs_encoded = self._encode_obs(
                obs,
                self.actor_obs_groups_1d,
                self.actor_obs_groups_2d,
                self.actor_cnns,
                self.actor_obs_normalizer,
            )
            out_mem = self.memory_a(obs_encoded, masks, hidden_state).squeeze(0)
            self._update_distribution(out_mem)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            obs_encoded = self._encode_obs(
                obs,
                self.actor_obs_groups_1d,
                self.actor_obs_groups_2d,
                self.actor_cnns,
                self.actor_obs_normalizer,
            )
            out_mem = self.memory_a(obs_encoded).squeeze(0)
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(out_mem))
                return mean
            return self.actor(out_mem)

        def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
            obs_encoded = self._encode_obs(
                obs,
                self.critic_obs_groups_1d,
                self.critic_obs_groups_2d,
                self.critic_cnns,
                self.critic_obs_normalizer,
            )
            out_mem = self.memory_c(obs_encoded, masks, hidden_state).squeeze(0)
            return self.critic(out_mem)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        def get_hidden_states(self):
            return self.memory_a.hidden_state, self.memory_c.hidden_state

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_obs_groups_1d:
                actor_obs = self._encode_1d_obs(obs, self.actor_obs_groups_1d)
                if actor_obs is not None:
                    self.actor_obs_normalizer.update(actor_obs)
            if self.critic_obs_normalization and self.critic_obs_groups_1d:
                critic_obs = self._encode_1d_obs(obs, self.critic_obs_groups_1d)
                if critic_obs is not None:
                    self.critic_obs_normalizer.update(critic_obs)

    class ActorCriticCNNMaxpooling(nn.Module):
        """Actor-Critic that encodes depth frames with a shared CNN and max-pools over frames."""

        is_recurrent: bool = False

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            **kwargs,
        ):
            if kwargs:
                print(
                    "ActorCriticCNNMaxpooling.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if CNN is None or EmpiricalNormalization is None:
                raise ImportError("rsl_rl is required to use ActorCriticCNNMaxpooling")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            self.actor_pose_groups, self.actor_depth_group = self._split_obs_groups(obs, obs_groups["policy"])
            self.critic_pose_groups, self.critic_depth_group = self._split_obs_groups(obs, obs_groups["critic"])

            self.actor_cnn, actor_enc_dim = self._build_cnn(self.actor_depth_group, obs, actor_cnn_cfg)
            self.critic_cnn, critic_enc_dim = self._build_cnn(self.critic_depth_group, obs, critic_cnn_cfg)

            actor_pose_dim = self._pose_dim(obs, self.actor_pose_groups)
            critic_pose_dim = self._pose_dim(obs, self.critic_pose_groups)

            actor_input_dim = actor_pose_dim + actor_enc_dim
            critic_input_dim = critic_pose_dim + critic_enc_dim

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(critic_input_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and actor_pose_dim > 0
            self.critic_obs_normalization = critic_obs_normalization and critic_pose_dim > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(actor_pose_dim)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(critic_pose_dim)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]) -> tuple[List[str], str]:
            pose_groups: List[str] = []
            depth_group: str | None = None
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 4:
                    depth_group = obs_group
                elif len(obs_shape) == 2:
                    pose_groups.append(obs_group)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            if depth_group is None:
                raise ValueError("ActorCriticCNNMaxpooling requires a 4D depth observation.")
            return pose_groups, depth_group

        @staticmethod
        def _resolve_cnn_cfg(depth_group: str, cnn_cfg: dict | None):
            if cnn_cfg is None:
                raise ValueError("CNN config must be provided for depth inputs.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                return cnn_cfg
            if depth_group not in cnn_cfg:
                raise ValueError(f"Missing CNN config for {depth_group}.")
            return cnn_cfg[depth_group]

        def _build_cnn(self, depth_group: str, obs, cnn_cfg: dict | None):
            cfg = self._resolve_cnn_cfg(depth_group, cnn_cfg)
            in_dims = obs[depth_group].shape[-2:]
            cnn = CNN(input_dim=in_dims, input_channels=1, **cfg)
            if cnn.output_channels is not None:
                raise ValueError("CNN outputs must be flattened for the MLP.")
            return cnn, int(cnn.output_dim)

        @staticmethod
        def _pose_dim(obs, pose_groups: List[str]) -> int:
            if not pose_groups:
                return 0
            return int(sum(obs[group].shape[-1] for group in pose_groups))

        def _encode_pose(self, obs, pose_groups: List[str], normalizer: nn.Module) -> torch.Tensor | None:
            if not pose_groups:
                return None
            obs_list = [obs[group] for group in pose_groups]
            pose = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
            return normalizer(pose)

        def _encode_depth(self, depth: torch.Tensor, cnn: nn.Module) -> torch.Tensor:
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)
            if depth.dim() != 4:
                raise ValueError(f"Depth tensor must be 4D (B,T,H,W), got {depth.shape}.")
            batch, frames, height, width = depth.shape
            flat = depth.reshape(batch * frames, 1, height, width)
            enc = cnn(flat)
            enc = enc.reshape(batch, frames, -1)
            return torch.amax(enc, dim=1)

        def _encode(self, obs, pose_groups: List[str], depth_group: str, cnn: nn.Module, normalizer: nn.Module):
            pose = self._encode_pose(obs, pose_groups, normalizer)
            depth = self._encode_depth(obs[depth_group], cnn)
            if pose is None:
                return depth
            return torch.cat([pose, depth], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        def _update_distribution(self, mlp_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(mlp_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(mlp_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.actor_pose_groups, self.actor_depth_group, self.actor_cnn, self.actor_obs_normalizer
            )
            self._update_distribution(mlp_obs)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.actor_pose_groups, self.actor_depth_group, self.actor_cnn, self.actor_obs_normalizer
            )
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(mlp_obs))
                return mean
            return self.actor(mlp_obs)

        def evaluate(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.critic_pose_groups, self.critic_depth_group, self.critic_cnn, self.critic_obs_normalizer
            )
            return self.critic(mlp_obs)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_pose_groups:
                actor_pose = self._encode_pose(obs, self.actor_pose_groups, self.actor_obs_normalizer)
                if actor_pose is not None:
                    self.actor_obs_normalizer.update(actor_pose)
            if self.critic_obs_normalization and self.critic_pose_groups:
                critic_pose = self._encode_pose(obs, self.critic_pose_groups, self.critic_obs_normalizer)
                if critic_pose is not None:
                    self.critic_obs_normalizer.update(critic_pose)

        def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
            # Non-recurrent policy: no hidden state to reset.
            return None

        def get_hidden_states(self):
            return None

    class ActorCriticRGBCNNMaxpooling_2D(nn.Module):
        """Actor-Critic that encodes RGB frames with a shared CNN and max-pools over frames."""

        is_recurrent: bool = False

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            **kwargs,
        ):
            if kwargs:
                print(
                    "ActorCriticRGBCNNMaxpooling_2D.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if CNN is None or EmpiricalNormalization is None:
                raise ImportError("rsl_rl is required to use ActorCriticRGBCNNMaxpooling_2D")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            self.actor_pose_groups, self.actor_rgb_group = self._split_obs_groups(obs, obs_groups["policy"])
            self.critic_pose_groups, self.critic_rgb_group = self._split_obs_groups(obs, obs_groups["critic"])

            self.actor_cnn, actor_enc_dim = self._build_cnn(self.actor_rgb_group, obs, actor_cnn_cfg)
            self.critic_cnn, critic_enc_dim = self._build_cnn(self.critic_rgb_group, obs, critic_cnn_cfg)

            actor_pose_dim = self._pose_dim(obs, self.actor_pose_groups)
            critic_pose_dim = self._pose_dim(obs, self.critic_pose_groups)

            actor_input_dim = actor_pose_dim + actor_enc_dim
            critic_input_dim = critic_pose_dim + critic_enc_dim

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(critic_input_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and actor_pose_dim > 0
            self.critic_obs_normalization = critic_obs_normalization and critic_pose_dim > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(actor_pose_dim)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(critic_pose_dim)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]) -> tuple[List[str], str]:
            pose_groups: List[str] = []
            rgb_group: str | None = None
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 5:
                    rgb_group = obs_group
                elif len(obs_shape) == 2:
                    pose_groups.append(obs_group)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            if rgb_group is None:
                raise ValueError("ActorCriticRGBCNNMaxpooling_2D requires a 5D RGB observation.")
            return pose_groups, rgb_group

        @staticmethod
        def _resolve_cnn_cfg(rgb_group: str, cnn_cfg: dict | None):
            if cnn_cfg is None:
                raise ValueError("CNN config must be provided for RGB inputs.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                return cnn_cfg
            if rgb_group not in cnn_cfg:
                raise ValueError(f"Missing CNN config for {rgb_group}.")
            return cnn_cfg[rgb_group]

        def _build_cnn(self, rgb_group: str, obs, cnn_cfg: dict | None):
            cfg = self._resolve_cnn_cfg(rgb_group, cnn_cfg)
            in_dims = obs[rgb_group].shape[-3:-1] if obs[rgb_group].shape[-1] == 3 else obs[rgb_group].shape[-2:]
            cnn = CNN(input_dim=in_dims, input_channels=3, **cfg)
            if cnn.output_channels is not None:
                raise ValueError("CNN outputs must be flattened for the MLP.")
            return cnn, int(cnn.output_dim)

        @staticmethod
        def _pose_dim(obs, pose_groups: List[str]) -> int:
            if not pose_groups:
                return 0
            return int(sum(obs[group].shape[-1] for group in pose_groups))

        def _encode_pose(self, obs, pose_groups: List[str], normalizer: nn.Module) -> torch.Tensor | None:
            if not pose_groups:
                return None
            obs_list = [obs[group] for group in pose_groups]
            pose = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
            return normalizer(pose)

        def _encode_rgb(self, rgb: torch.Tensor, cnn: nn.Module) -> torch.Tensor:
            squeeze_time = False
            if rgb.dim() == 5:
                rgb = rgb.unsqueeze(0)
                squeeze_time = True
            if rgb.dim() != 6:
                raise ValueError(
                    f"RGB tensor must be 5D (B,F,H,W,3)/(B,F,3,H,W) or 6D (T,B,F,H,W,3)/(T,B,F,3,H,W), got {rgb.shape}."
                )
            if rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 1, 2, 5, 3, 4)
            elif rgb.shape[3] != 3:
                raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}.")
            time_steps, batch, frames, channels, height, width = rgb.shape
            flat = rgb.reshape(time_steps * batch * frames, channels, height, width)
            enc = cnn(flat)
            enc = enc.reshape(time_steps, batch, frames, -1)
            enc = torch.amax(enc, dim=2)
            if squeeze_time:
                enc = enc.squeeze(0)
            return enc

        def _encode(self, obs, pose_groups: List[str], rgb_group: str, cnn: nn.Module, normalizer: nn.Module):
            pose = self._encode_pose(obs, pose_groups, normalizer)
            rgb = self._encode_rgb(obs[rgb_group], cnn)
            if pose is None:
                return rgb
            return torch.cat([pose, rgb], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

    class ActorCriticRGBCNNMaxpooling_3D(ActorCriticRGBCNNMaxpooling_2D):
        """RGB maxpool policy intended for 3D control (e.g., 7-DoF actions)."""

        pass

        def _update_distribution(self, mlp_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(mlp_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(mlp_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.actor_pose_groups, self.actor_rgb_group, self.actor_cnn, self.actor_obs_normalizer
            )
            self._update_distribution(mlp_obs)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.actor_pose_groups, self.actor_rgb_group, self.actor_cnn, self.actor_obs_normalizer
            )
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(mlp_obs))
                return mean
            return self.actor(mlp_obs)

        def evaluate(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs, self.critic_pose_groups, self.critic_rgb_group, self.critic_cnn, self.critic_obs_normalizer
            )
            return self.critic(mlp_obs)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_pose_groups:
                actor_pose = self._encode_pose(obs, self.actor_pose_groups, self.actor_obs_normalizer)
                if actor_pose is not None:
                    self.actor_obs_normalizer.update(actor_pose)
            if self.critic_obs_normalization and self.critic_pose_groups:
                critic_pose = self._encode_pose(obs, self.critic_pose_groups, self.critic_obs_normalizer)
                if critic_pose is not None:
                    self.critic_obs_normalizer.update(critic_pose)

        def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
            return None

        def get_hidden_states(self):
            return None

    class ActorCriticRGBCNNRecurrent(nn.Module):
        """Recurrent Actor-Critic with RGB encoder + pose/state features."""

        is_recurrent: bool = True

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            rnn_type: str = "lstm",
            rnn_hidden_dim: int = 256,
            rnn_num_layers: int = 1,
            **kwargs,
        ):
            if kwargs:
                print(
                    "ActorCriticRGBCNNRecurrent.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if CNN is None or EmpiricalNormalization is None or Memory is None:
                raise ImportError("rsl_rl is required to use ActorCriticRGBCNNRecurrent")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            self.actor_pose_groups, self.actor_rgb_group = self._split_obs_groups(obs, obs_groups["policy"])
            self.critic_pose_groups, self.critic_rgb_group = self._split_obs_groups(obs, obs_groups["critic"])

            self.actor_cnn, actor_enc_dim = self._build_cnn(self.actor_rgb_group, obs, actor_cnn_cfg)
            self.critic_cnn, critic_enc_dim = self._build_cnn(self.critic_rgb_group, obs, critic_cnn_cfg)

            actor_pose_dim = self._pose_dim(obs, self.actor_pose_groups)
            critic_pose_dim = self._pose_dim(obs, self.critic_pose_groups)

            actor_input_dim = actor_pose_dim + actor_enc_dim
            critic_input_dim = critic_pose_dim + critic_enc_dim

            self.memory_a = Memory(actor_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
            self.memory_c = Memory(critic_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(rnn_hidden_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(rnn_hidden_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and actor_pose_dim > 0
            self.critic_obs_normalization = critic_obs_normalization and critic_pose_dim > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(actor_pose_dim)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(critic_pose_dim)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]) -> tuple[List[str], str]:
            pose_groups: List[str] = []
            rgb_group: str | None = None
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 5:
                    rgb_group = obs_group
                elif len(obs_shape) == 2:
                    pose_groups.append(obs_group)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            if rgb_group is None:
                raise ValueError("ActorCriticRGBCNNRecurrent requires a 5D RGB observation.")
            return pose_groups, rgb_group

        @staticmethod
        def _resolve_cnn_cfg(rgb_group: str, cnn_cfg: dict | None):
            if cnn_cfg is None:
                raise ValueError("CNN config must be provided for RGB inputs.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                return cnn_cfg
            if rgb_group not in cnn_cfg:
                raise ValueError(f"Missing CNN config for {rgb_group}.")
            return cnn_cfg[rgb_group]

        def _build_cnn(self, rgb_group: str, obs, cnn_cfg: dict | None):
            cfg = self._resolve_cnn_cfg(rgb_group, cnn_cfg)
            in_dims = obs[rgb_group].shape[-3:-1] if obs[rgb_group].shape[-1] == 3 else obs[rgb_group].shape[-2:]
            cnn = CNN(input_dim=in_dims, input_channels=3, **cfg)
            if cnn.output_channels is not None:
                raise ValueError("CNN outputs must be flattened for the MLP.")
            return cnn, int(cnn.output_dim)

        @staticmethod
        def _pose_dim(obs, pose_groups: List[str]) -> int:
            if not pose_groups:
                return 0
            return int(sum(obs[group].shape[-1] for group in pose_groups))

        def _encode_pose(self, obs, pose_groups: List[str], normalizer: nn.Module) -> torch.Tensor | None:
            if not pose_groups:
                return None
            obs_list = [obs[group] for group in pose_groups]
            pose = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
            return normalizer(pose)

        def _encode_rgb(self, rgb: torch.Tensor, cnn: nn.Module) -> torch.Tensor:
            squeeze_time = False
            if rgb.dim() == 5:
                rgb = rgb.unsqueeze(0)
                squeeze_time = True
            if rgb.dim() != 6:
                raise ValueError(
                    f"RGB tensor must be 5D (B,F,H,W,3)/(B,F,3,H,W) or 6D (T,B,F,H,W,3)/(T,B,F,3,H,W), got {rgb.shape}."
                )
            if rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 1, 2, 5, 3, 4)
            elif rgb.shape[3] != 3:
                raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}.")
            time_steps, batch, frames, channels, height, width = rgb.shape
            flat = rgb.reshape(time_steps * batch * frames, channels, height, width)
            enc = cnn(flat)
            enc = enc.reshape(time_steps, batch, frames, -1)
            enc = torch.amax(enc, dim=2)
            if squeeze_time:
                enc = enc.squeeze(0)
            return enc

        def _encode(self, obs, pose_groups: List[str], rgb_group: str, cnn: nn.Module, normalizer: nn.Module):
            pose = self._encode_pose(obs, pose_groups, normalizer)
            rgb = self._encode_rgb(obs[rgb_group], cnn)
            if pose is None:
                return rgb
            return torch.cat([pose, rgb], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        def _update_distribution(self, rnn_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(rnn_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(rnn_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
            obs_encoded = self._encode(
                obs, self.actor_pose_groups, self.actor_rgb_group, self.actor_cnn, self.actor_obs_normalizer
            )
            out_mem = self.memory_a(obs_encoded, masks, hidden_state).squeeze(0)
            self._update_distribution(out_mem)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            obs_encoded = self._encode(
                obs, self.actor_pose_groups, self.actor_rgb_group, self.actor_cnn, self.actor_obs_normalizer
            )
            out_mem = self.memory_a(obs_encoded).squeeze(0)
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(out_mem))
                return mean
            return self.actor(out_mem)

        def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
            obs_encoded = self._encode(
                obs, self.critic_pose_groups, self.critic_rgb_group, self.critic_cnn, self.critic_obs_normalizer
            )
            out_mem = self.memory_c(obs_encoded, masks, hidden_state).squeeze(0)
            return self.critic(out_mem)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def get_hidden_states(self):
            return self.memory_a.hidden_state, self.memory_c.hidden_state

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_pose_groups:
                actor_obs = self._encode_pose(obs, self.actor_pose_groups, self.actor_obs_normalizer)
                if actor_obs is not None:
                    self.actor_obs_normalizer.update(actor_obs)
            if self.critic_obs_normalization and self.critic_pose_groups:
                critic_obs = self._encode_pose(obs, self.critic_pose_groups, self.critic_obs_normalizer)
                if critic_obs is not None:
                    self.critic_obs_normalizer.update(critic_obs)

        def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
            self.memory_a.reset(dones)
            self.memory_c.reset(dones)

    class ActorCriticRGBCNNPoseDelta(nn.Module):
        """Feed-forward RGB policy with per-frame camera pose embedding for delta xyz+quat actions."""

        is_recurrent: bool = False

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            frame_pose_hidden_dim: int = 64,
            frame_joint_hidden_dim: int = 128,
            robot_pose_hidden_dim: int = 128,
            **kwargs,
        ):
            if kwargs:
                print(
                    "ActorCriticRGBCNNPoseDelta.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if CNN is None or EmpiricalNormalization is None:
                raise ImportError("rsl_rl is required to use ActorCriticRGBCNNPoseDelta")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            (
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_num_frames,
            ) = self._split_obs_groups(obs, obs_groups["policy"])
            (
                self.critic_pose_groups,
                self.critic_rgb_group,
                self.critic_cam_pose_group,
                self.critic_num_frames,
            ) = self._split_obs_groups(obs, obs_groups["critic"])

            self.actor_cnn, actor_visual_dim = self._build_cnn(self.actor_rgb_group, obs, actor_cnn_cfg)
            self.critic_cnn, critic_visual_dim = self._build_cnn(self.critic_rgb_group, obs, critic_cnn_cfg)

            self.actor_frame_pose_encoder = _build_mlp(
                7, frame_pose_hidden_dim, [frame_pose_hidden_dim], activation
            )
            self.critic_frame_pose_encoder = _build_mlp(
                7, frame_pose_hidden_dim, [frame_pose_hidden_dim], activation
            )
            self.actor_frame_fusion = _build_mlp(
                actor_visual_dim + frame_pose_hidden_dim,
                frame_joint_hidden_dim,
                [frame_joint_hidden_dim],
                activation,
            )
            self.critic_frame_fusion = _build_mlp(
                critic_visual_dim + frame_pose_hidden_dim,
                frame_joint_hidden_dim,
                [frame_joint_hidden_dim],
                activation,
            )

            actor_pose_dim = self._pose_dim(obs, self.actor_pose_groups)
            critic_pose_dim = self._pose_dim(obs, self.critic_pose_groups)

            self.actor_robot_pose_encoder = (
                _build_mlp(actor_pose_dim, robot_pose_hidden_dim, [robot_pose_hidden_dim], activation)
                if actor_pose_dim > 0
                else None
            )
            self.critic_robot_pose_encoder = (
                _build_mlp(critic_pose_dim, robot_pose_hidden_dim, [robot_pose_hidden_dim], activation)
                if critic_pose_dim > 0
                else None
            )

            actor_input_dim = self.actor_num_frames * frame_joint_hidden_dim + (
                robot_pose_hidden_dim if actor_pose_dim > 0 else 0
            )
            critic_input_dim = self.critic_num_frames * frame_joint_hidden_dim + (
                robot_pose_hidden_dim if critic_pose_dim > 0 else 0
            )

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(critic_input_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and actor_pose_dim > 0
            self.critic_obs_normalization = critic_obs_normalization and critic_pose_dim > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(actor_pose_dim)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(critic_pose_dim)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]) -> tuple[List[str], str, str, int]:
            pose_groups: List[str] = []
            rgb_group: str | None = None
            cam_pose_group: str | None = None
            num_frames = 0
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 5:
                    rgb_group = obs_group
                    num_frames = int(obs_shape[1])
                elif len(obs_shape) == 3 and obs_shape[-1] == 7:
                    cam_pose_group = obs_group
                    if num_frames == 0:
                        num_frames = int(obs_shape[1])
                elif len(obs_shape) == 2:
                    pose_groups.append(obs_group)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            if rgb_group is None:
                raise ValueError("ActorCriticRGBCNNPoseDelta requires a 5D RGB observation.")
            if cam_pose_group is None:
                raise ValueError("ActorCriticRGBCNNPoseDelta requires a 3D camera-pose observation.")
            if num_frames <= 0:
                raise ValueError("Invalid number of frames inferred from observation shapes.")
            return pose_groups, rgb_group, cam_pose_group, num_frames

        @staticmethod
        def _resolve_cnn_cfg(rgb_group: str, cnn_cfg: dict | None):
            if cnn_cfg is None:
                raise ValueError("CNN config must be provided for RGB inputs.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                return cnn_cfg
            if rgb_group not in cnn_cfg:
                raise ValueError(f"Missing CNN config for {rgb_group}.")
            return cnn_cfg[rgb_group]

        def _build_cnn(self, rgb_group: str, obs, cnn_cfg: dict | None):
            cfg = self._resolve_cnn_cfg(rgb_group, cnn_cfg)
            in_dims = obs[rgb_group].shape[-3:-1] if obs[rgb_group].shape[-1] == 3 else obs[rgb_group].shape[-2:]
            cnn = CNN(input_dim=in_dims, input_channels=3, **cfg)
            if cnn.output_channels is not None:
                raise ValueError("CNN outputs must be flattened for the MLP.")
            return cnn, int(cnn.output_dim)

        @staticmethod
        def _pose_dim(obs, pose_groups: List[str]) -> int:
            if not pose_groups:
                return 0
            return int(sum(obs[group].shape[-1] for group in pose_groups))

        def _encode_pose(self, obs, pose_groups: List[str], normalizer: nn.Module) -> torch.Tensor | None:
            if not pose_groups:
                return None
            obs_list = [obs[group] for group in pose_groups]
            pose = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
            return normalizer(pose)

        def _encode_rgb_frames(self, rgb: torch.Tensor, cnn: nn.Module) -> torch.Tensor:
            if rgb.dim() != 5:
                raise ValueError(
                    f"RGB tensor must be 5D (B,F,H,W,3)/(B,F,3,H,W), got {rgb.shape}."
                )
            if rgb.shape[-1] == 3:
                batch, frames, height, width, channels = rgb.shape
                flat = rgb.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
            elif rgb.shape[2] == 3:
                batch, frames, channels, height, width = rgb.shape
                flat = rgb.reshape(batch * frames, channels, height, width)
            else:
                raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}.")
            enc = cnn(flat)
            return enc.reshape(batch, frames, -1)

        def _encode_camera_pose_frames(self, cam_pose: torch.Tensor, pose_encoder: nn.Module) -> torch.Tensor:
            if cam_pose.dim() != 3:
                raise ValueError(f"Camera pose tensor must be 3D (B,F,7), got {cam_pose.shape}.")
            batch, frames, dim = cam_pose.shape
            if dim < 7:
                raise ValueError(f"Camera pose tensor must have at least 7 dims, got {cam_pose.shape}.")
            flat = cam_pose[..., :7].reshape(batch * frames, 7)
            enc = pose_encoder(flat)
            return enc.reshape(batch, frames, -1)

        def _encode_frames(
            self,
            rgb: torch.Tensor,
            cam_pose: torch.Tensor,
            cnn: nn.Module,
            pose_encoder: nn.Module,
            frame_fusion: nn.Module,
            expected_frames: int,
        ) -> torch.Tensor:
            rgb_enc = self._encode_rgb_frames(rgb, cnn)
            pose_enc = self._encode_camera_pose_frames(cam_pose, pose_encoder)
            batch = rgb_enc.shape[0]
            frames = min(rgb_enc.shape[1], pose_enc.shape[1], expected_frames)
            fused = torch.cat([rgb_enc[:, :frames], pose_enc[:, :frames]], dim=-1)
            fused = frame_fusion(fused.reshape(batch * frames, -1)).reshape(batch, frames, -1)
            if frames < expected_frames:
                pad = torch.zeros(
                    batch,
                    expected_frames - frames,
                    fused.shape[-1],
                    dtype=fused.dtype,
                    device=fused.device,
                )
                fused = torch.cat([fused, pad], dim=1)
            return fused.reshape(batch, -1)

        def _encode(
            self,
            obs,
            pose_groups: List[str],
            rgb_group: str,
            cam_pose_group: str,
            cnn: nn.Module,
            frame_pose_encoder: nn.Module,
            frame_fusion: nn.Module,
            pose_encoder: nn.Module | None,
            normalizer: nn.Module,
            expected_frames: int,
        ) -> torch.Tensor:
            frame_feat = self._encode_frames(
                obs[rgb_group],
                obs[cam_pose_group],
                cnn=cnn,
                pose_encoder=frame_pose_encoder,
                frame_fusion=frame_fusion,
                expected_frames=expected_frames,
            )
            pose = self._encode_pose(obs, pose_groups, normalizer)
            if pose is None or pose_encoder is None:
                return frame_feat
            pose_feat = pose_encoder(pose)
            return torch.cat([frame_feat, pose_feat], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        def _update_distribution(self, mlp_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(mlp_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(mlp_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_cnn,
                self.actor_frame_pose_encoder,
                self.actor_frame_fusion,
                self.actor_robot_pose_encoder,
                self.actor_obs_normalizer,
                self.actor_num_frames,
            )
            self._update_distribution(mlp_obs)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_cnn,
                self.actor_frame_pose_encoder,
                self.actor_frame_fusion,
                self.actor_robot_pose_encoder,
                self.actor_obs_normalizer,
                self.actor_num_frames,
            )
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(mlp_obs))
                return mean
            return self.actor(mlp_obs)

        def evaluate(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.critic_pose_groups,
                self.critic_rgb_group,
                self.critic_cam_pose_group,
                self.critic_cnn,
                self.critic_frame_pose_encoder,
                self.critic_frame_fusion,
                self.critic_robot_pose_encoder,
                self.critic_obs_normalizer,
                self.critic_num_frames,
            )
            return self.critic(mlp_obs)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_pose_groups:
                actor_pose = self._encode_pose(obs, self.actor_pose_groups, nn.Identity())
                if actor_pose is not None:
                    self.actor_obs_normalizer.update(actor_pose)
            if self.critic_obs_normalization and self.critic_pose_groups:
                critic_pose = self._encode_pose(obs, self.critic_pose_groups, nn.Identity())
                if critic_pose is not None:
                    self.critic_obs_normalizer.update(critic_pose)

        def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
            return None

        def get_hidden_states(self):
            return None

    class ActorCriticRGBVGGTFrozenPoseDelta(nn.Module):
        """Feed-forward RGB policy with frozen VGGT feature extractor (layers 4/11/17/23 by default)."""

        is_recurrent: bool = False

        def __init__(
            self,
            obs,
            obs_groups: dict[str, List[str]],
            num_actions: int,
            actor_hidden_dims: List[int],
            critic_hidden_dims: List[int],
            activation: str,
            init_noise_std: float,
            state_dependent_std: bool = False,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_cnn_cfg: dict[str, dict] | dict | None = None,
            critic_cnn_cfg: dict[str, dict] | dict | None = None,
            noise_std_type: str = "scalar",
            frame_pose_hidden_dim: int = 64,
            frame_joint_hidden_dim: int = 128,
            robot_pose_hidden_dim: int = 128,
            vggt_model_id: str = "facebook/VGGT-1B",
            vggt_model_cache_dir: str = "",
            vggt_model_local_files_only: bool = True,
            vggt_input_size: int = 518,
            vggt_embed_dim: int = 1024,
            vggt_feature_layers: List[int] | tuple[int, ...] = (4, 11, 17, 23),
            vggt_frame_hidden_dim: int = 256,
            **kwargs,
        ):
            if kwargs:
                print(
                    "ActorCriticRGBVGGTFrozenPoseDelta.__init__ got unexpected arguments, which will be ignored: "
                    + str(kwargs.keys())
                )
            if EmpiricalNormalization is None:
                raise ImportError("rsl_rl is required to use ActorCriticRGBVGGTFrozenPoseDelta")
            super().__init__()

            self.obs_groups = obs_groups
            self.state_dependent_std = state_dependent_std
            self.noise_std_type = noise_std_type

            (
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_num_frames,
            ) = self._split_obs_groups(obs, obs_groups["policy"])
            (
                self.critic_pose_groups,
                self.critic_rgb_group,
                self.critic_cam_pose_group,
                self.critic_num_frames,
            ) = self._split_obs_groups(obs, obs_groups["critic"])

            self.vggt_input_size = int(vggt_input_size)
            self.vggt_feature_layers = tuple(int(i) for i in vggt_feature_layers)
            if len(self.vggt_feature_layers) != 4:
                raise ValueError(
                    "ActorCriticRGBVGGTFrozenPoseDelta expects exactly 4 feature layers (e.g., 4,11,17,23)."
                )
            self._resolved_vggt_layers: tuple[int, ...] | None = None
            self._vggt_cache_key = None
            self._vggt_cache_value = None

            self.vggt = self._build_vggt(
                model_id=vggt_model_id,
                cache_dir=vggt_model_cache_dir,
                local_files_only=bool(vggt_model_local_files_only),
                input_size=int(vggt_input_size),
                embed_dim=int(vggt_embed_dim),
            )
            patch_size = int(self.vggt.aggregator.patch_size)
            if self.vggt_input_size % patch_size != 0:
                raise ValueError(
                    f"vggt_input_size={self.vggt_input_size} must be divisible by VGGT patch_size={patch_size}."
                )
            self.vggt.eval()
            for param in self.vggt.parameters():
                param.requires_grad_(False)

            token_dim = 2 * int(self.vggt.aggregator.camera_token.shape[-1])
            self.actor_dpt = self._build_light_dpt_modules(
                token_dim=token_dim,
                hidden_dim=int(vggt_frame_hidden_dim),
                activation=activation,
            )
            self.critic_dpt = self._build_light_dpt_modules(
                token_dim=token_dim,
                hidden_dim=int(vggt_frame_hidden_dim),
                activation=activation,
            )
            self.actor_frame_pose_encoder = _build_mlp(
                7, frame_pose_hidden_dim, [frame_pose_hidden_dim], activation
            )
            self.critic_frame_pose_encoder = _build_mlp(
                7, frame_pose_hidden_dim, [frame_pose_hidden_dim], activation
            )
            self.actor_frame_fusion = _build_mlp(
                vggt_frame_hidden_dim + frame_pose_hidden_dim,
                frame_joint_hidden_dim,
                [frame_joint_hidden_dim],
                activation,
            )
            self.critic_frame_fusion = _build_mlp(
                vggt_frame_hidden_dim + frame_pose_hidden_dim,
                frame_joint_hidden_dim,
                [frame_joint_hidden_dim],
                activation,
            )

            actor_pose_dim = self._pose_dim(obs, self.actor_pose_groups)
            critic_pose_dim = self._pose_dim(obs, self.critic_pose_groups)

            self.actor_robot_pose_encoder = (
                _build_mlp(actor_pose_dim, robot_pose_hidden_dim, [robot_pose_hidden_dim], activation)
                if actor_pose_dim > 0
                else None
            )
            self.critic_robot_pose_encoder = (
                _build_mlp(critic_pose_dim, robot_pose_hidden_dim, [robot_pose_hidden_dim], activation)
                if critic_pose_dim > 0
                else None
            )

            actor_input_dim = self.actor_num_frames * frame_joint_hidden_dim + (
                robot_pose_hidden_dim if actor_pose_dim > 0 else 0
            )
            critic_input_dim = self.critic_num_frames * frame_joint_hidden_dim + (
                robot_pose_hidden_dim if critic_pose_dim > 0 else 0
            )

            actor_out_dim = num_actions * 2 if state_dependent_std else num_actions
            self.actor = _build_mlp(actor_input_dim, actor_out_dim, actor_hidden_dims, activation)
            self.critic = _build_mlp(critic_input_dim, 1, critic_hidden_dims, activation)

            self.actor_obs_normalization = actor_obs_normalization and actor_pose_dim > 0
            self.critic_obs_normalization = critic_obs_normalization and critic_pose_dim > 0
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(actor_pose_dim)
            else:
                self.actor_obs_normalizer = nn.Identity()
            if self.critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(critic_pose_dim)
            else:
                self.critic_obs_normalizer = nn.Identity()

            if self.state_dependent_std:
                if isinstance(self.actor[-1], nn.Linear):
                    if self.noise_std_type == "scalar":
                        nn.init.constant_(self.actor[-1].bias[num_actions:], init_noise_std)
                    elif self.noise_std_type == "log":
                        nn.init.constant_(
                            self.actor[-1].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                        )
                    else:
                        raise ValueError(
                            f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                        )
            else:
                if self.noise_std_type == "scalar":
                    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                elif self.noise_std_type == "log":
                    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                    )

            self.distribution = None
            Normal.set_default_validate_args(False)

        @staticmethod
        def _build_vggt(
            model_id: str,
            cache_dir: str,
            local_files_only: bool,
            input_size: int,
            embed_dim: int,
        ):
            try:
                from vggt.models.vggt import VGGT
            except ImportError as exc:
                raise ImportError("Failed to import VGGT. Ensure local vggt package is available.") from exc

            if model_id:
                return VGGT.from_pretrained(
                    model_id,
                    local_files_only=bool(local_files_only),
                    cache_dir=cache_dir or None,
                )
            if int(embed_dim) != 1024:
                raise ValueError(
                    "When vggt_model_id is empty, local VGGT constructor currently expects embed_dim=1024."
                )
            return VGGT(
                img_size=int(input_size),
                embed_dim=1024,
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=False,
            )

        @staticmethod
        def _build_light_dpt_modules(token_dim: int, hidden_dim: int, activation: str) -> nn.ModuleDict:
            dpt_channels = [128, 192, 256, 256]
            dpt_features = 128
            modules = nn.ModuleDict()
            modules["projects"] = nn.ModuleList(
                [nn.Conv2d(token_dim, out_ch, kernel_size=1, stride=1, padding=0) for out_ch in dpt_channels]
            )
            modules["resize_layers"] = nn.ModuleList(
                [
                    nn.ConvTranspose2d(dpt_channels[0], dpt_channels[0], kernel_size=4, stride=4, padding=0),
                    nn.ConvTranspose2d(dpt_channels[1], dpt_channels[1], kernel_size=2, stride=2, padding=0),
                    nn.Identity(),
                    nn.Conv2d(dpt_channels[3], dpt_channels[3], kernel_size=3, stride=2, padding=1),
                ]
            )
            modules["scratch_layers"] = nn.ModuleList(
                [
                    nn.Conv2d(dpt_channels[0], dpt_features, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Conv2d(dpt_channels[1], dpt_features, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Conv2d(dpt_channels[2], dpt_features, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Conv2d(dpt_channels[3], dpt_features, kernel_size=3, stride=1, padding=1, bias=False),
                ]
            )
            modules["refinenet4"] = _LightFeatureFusionBlock(dpt_features, has_residual=False)
            modules["refinenet3"] = _LightFeatureFusionBlock(dpt_features, has_residual=True)
            modules["refinenet2"] = _LightFeatureFusionBlock(dpt_features, has_residual=True)
            modules["refinenet1"] = _LightFeatureFusionBlock(dpt_features, has_residual=True)
            modules["output_conv1"] = nn.Conv2d(dpt_features, dpt_features, kernel_size=3, stride=1, padding=1)
            modules["output_proj"] = _build_mlp(dpt_features, hidden_dim, [hidden_dim], activation)
            return modules

        @staticmethod
        def _split_obs_groups(obs, obs_groups: List[str]) -> tuple[List[str], str, str, int]:
            pose_groups: List[str] = []
            rgb_group: str | None = None
            cam_pose_group: str | None = None
            num_frames = 0
            for obs_group in obs_groups:
                obs_shape = obs[obs_group].shape
                if len(obs_shape) == 5:
                    rgb_group = obs_group
                    num_frames = int(obs_shape[1])
                elif len(obs_shape) == 3 and obs_shape[-1] == 7:
                    cam_pose_group = obs_group
                    if num_frames == 0:
                        num_frames = int(obs_shape[1])
                elif len(obs_shape) == 2:
                    pose_groups.append(obs_group)
                else:
                    raise ValueError(f"Invalid observation shape for {obs_group}: {obs_shape}")
            if rgb_group is None:
                raise ValueError("ActorCriticRGBVGGTFrozenPoseDelta requires a 5D RGB observation.")
            if cam_pose_group is None:
                raise ValueError("ActorCriticRGBVGGTFrozenPoseDelta requires a 3D camera-pose observation.")
            if num_frames <= 0:
                raise ValueError("Invalid number of frames inferred from observation shapes.")
            return pose_groups, rgb_group, cam_pose_group, num_frames

        @staticmethod
        def _pose_dim(obs, pose_groups: List[str]) -> int:
            if not pose_groups:
                return 0
            return int(sum(obs[group].shape[-1] for group in pose_groups))

        def _encode_pose(self, obs, pose_groups: List[str], normalizer: nn.Module) -> torch.Tensor | None:
            if not pose_groups:
                return None
            obs_list = [obs[group] for group in pose_groups]
            pose = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
            return normalizer(pose)

        def _prepare_rgb_for_vggt(self, rgb: torch.Tensor) -> torch.Tensor:
            if rgb.dim() != 5:
                raise ValueError(f"RGB tensor must be 5D (B,F,H,W,3)/(B,F,3,H,W), got {rgb.shape}.")
            if rgb.shape[-1] == 3:
                batch, frames, height, width, channels = rgb.shape
                images = rgb.permute(0, 1, 4, 2, 3).contiguous()
            elif rgb.shape[2] == 3:
                batch, frames, channels, height, width = rgb.shape
                images = rgb.contiguous()
            else:
                raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}.")
            images = images.to(dtype=torch.float32).clamp(0.0, 1.0)
            if height != self.vggt_input_size or width != self.vggt_input_size:
                flat = images.reshape(batch * frames, channels, height, width)
                flat = torch_f.interpolate(
                    flat,
                    size=(self.vggt_input_size, self.vggt_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
                images = flat.reshape(batch, frames, channels, self.vggt_input_size, self.vggt_input_size)
            return images

        def _resolve_vggt_layers(self, num_layers: int) -> tuple[int, ...]:
            if self._resolved_vggt_layers is not None:
                return self._resolved_vggt_layers
            resolved: List[int] = []
            for layer in self.vggt_feature_layers:
                idx = int(layer)
                if idx < 0:
                    idx = num_layers + idx
                if idx < 0 or idx >= num_layers:
                    raise IndexError(f"VGGT feature layer index {layer} resolved to {idx}, valid [0, {num_layers - 1}]")
                resolved.append(idx)
            self._resolved_vggt_layers = tuple(resolved)
            return self._resolved_vggt_layers

        def _extract_vggt_layer_maps(self, rgb: torch.Tensor) -> tuple[List[torch.Tensor], int, int]:
            cache_key = (
                int(rgb.data_ptr()),
                tuple(rgb.shape),
                str(rgb.device),
                str(rgb.dtype),
            )
            if self._vggt_cache_key == cache_key and self._vggt_cache_value is not None:
                return self._vggt_cache_value

            images = self._prepare_rgb_for_vggt(rgb)
            batch, frames = int(images.shape[0]), int(images.shape[1])
            patch_size = int(self.vggt.aggregator.patch_size)
            patch_h = int(images.shape[-2] // patch_size)
            patch_w = int(images.shape[-1] // patch_size)
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)

            resolved_layers = self._resolve_vggt_layers(len(aggregated_tokens_list))
            layer_maps: List[torch.Tensor] = []
            for layer_idx in resolved_layers:
                tokens = aggregated_tokens_list[layer_idx]
                patch_tokens = tokens[:, :, patch_start_idx:]
                if patch_tokens.shape[2] == 0:
                    raise ValueError("VGGT patch token set is empty. Invalid patch_start_idx.")
                if int(patch_tokens.shape[2]) != patch_h * patch_w:
                    raise ValueError(
                        f"Patch token count mismatch: got {int(patch_tokens.shape[2])}, expected {patch_h * patch_w}."
                    )
                fmap = patch_tokens.reshape(batch * frames, patch_h, patch_w, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
                layer_maps.append(fmap.contiguous())
            out = (layer_maps, batch, frames)
            self._vggt_cache_key = cache_key
            self._vggt_cache_value = out
            return out

        def _fuse_dpt_features(self, layer_maps: List[torch.Tensor], dpt_modules: nn.ModuleDict) -> torch.Tensor:
            if len(layer_maps) != 4:
                raise ValueError(f"DPT fusion expects 4 layer maps, got {len(layer_maps)}.")
            projected: List[torch.Tensor] = []
            for idx in range(4):
                x = dpt_modules["projects"][idx](layer_maps[idx])
                x = dpt_modules["resize_layers"][idx](x)
                x = dpt_modules["scratch_layers"][idx](x)
                projected.append(x)
            layer1, layer2, layer3, layer4 = projected
            out = dpt_modules["refinenet4"](layer4, size=layer3.shape[2:])
            out = dpt_modules["refinenet3"](out, layer3, size=layer2.shape[2:])
            out = dpt_modules["refinenet2"](out, layer2, size=layer1.shape[2:])
            out = dpt_modules["refinenet1"](out, layer1)
            out = dpt_modules["output_conv1"](out)
            pooled = out.mean(dim=(2, 3))
            return dpt_modules["output_proj"](pooled)

        def _encode_camera_pose_frames(self, cam_pose: torch.Tensor, pose_encoder: nn.Module) -> torch.Tensor:
            if cam_pose.dim() != 3:
                raise ValueError(f"Camera pose tensor must be 3D (B,F,7), got {cam_pose.shape}.")
            batch, frames, dim = cam_pose.shape
            if dim < 7:
                raise ValueError(f"Camera pose tensor must have at least 7 dims, got {cam_pose.shape}.")
            flat = cam_pose[..., :7].reshape(batch * frames, 7)
            enc = pose_encoder(flat)
            return enc.reshape(batch, frames, -1)

        def _encode_frames(
            self,
            rgb: torch.Tensor,
            cam_pose: torch.Tensor,
            dpt_modules: nn.ModuleDict,
            pose_encoder: nn.Module,
            frame_fusion: nn.Module,
            expected_frames: int,
        ) -> torch.Tensor:
            layer_maps, batch, frames = self._extract_vggt_layer_maps(rgb)
            visual = self._fuse_dpt_features(layer_maps, dpt_modules).reshape(batch, frames, -1)
            pose_enc = self._encode_camera_pose_frames(cam_pose, pose_encoder)
            frames = min(visual.shape[1], pose_enc.shape[1], expected_frames)
            fused = torch.cat([visual[:, :frames], pose_enc[:, :frames]], dim=-1)
            fused = frame_fusion(fused.reshape(batch * frames, -1)).reshape(batch, frames, -1)
            if frames < expected_frames:
                pad = torch.zeros(
                    batch,
                    expected_frames - frames,
                    fused.shape[-1],
                    dtype=fused.dtype,
                    device=fused.device,
                )
                fused = torch.cat([fused, pad], dim=1)
            return fused.reshape(batch, -1)

        def _encode(
            self,
            obs,
            pose_groups: List[str],
            rgb_group: str,
            cam_pose_group: str,
            dpt_modules: nn.ModuleDict,
            frame_pose_encoder: nn.Module,
            frame_fusion: nn.Module,
            pose_encoder: nn.Module | None,
            normalizer: nn.Module,
            expected_frames: int,
        ) -> torch.Tensor:
            frame_feat = self._encode_frames(
                obs[rgb_group],
                obs[cam_pose_group],
                dpt_modules=dpt_modules,
                pose_encoder=frame_pose_encoder,
                frame_fusion=frame_fusion,
                expected_frames=expected_frames,
            )
            pose = self._encode_pose(obs, pose_groups, normalizer)
            if pose is None or pose_encoder is None:
                return frame_feat
            pose_feat = pose_encoder(pose)
            return torch.cat([frame_feat, pose_feat], dim=-1)

        def _split_mean_std(self, mean_and_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std_or_log = mean_and_std.chunk(2, dim=-1)
            if self.noise_std_type == "scalar":
                return mean, std_or_log
            if self.noise_std_type == "log":
                return mean, torch.exp(std_or_log)
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        def _update_distribution(self, mlp_obs: torch.Tensor) -> None:
            if self.state_dependent_std:
                mean_and_std = self.actor(mlp_obs)
                mean, std = self._split_mean_std(mean_and_std)
            else:
                mean = self.actor(mlp_obs)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            self.distribution = Normal(mean, std)

        def act(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_dpt,
                self.actor_frame_pose_encoder,
                self.actor_frame_fusion,
                self.actor_robot_pose_encoder,
                self.actor_obs_normalizer,
                self.actor_num_frames,
            )
            self._update_distribution(mlp_obs)
            return self.distribution.sample()

        def act_inference(self, obs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.actor_pose_groups,
                self.actor_rgb_group,
                self.actor_cam_pose_group,
                self.actor_dpt,
                self.actor_frame_pose_encoder,
                self.actor_frame_fusion,
                self.actor_robot_pose_encoder,
                self.actor_obs_normalizer,
                self.actor_num_frames,
            )
            if self.state_dependent_std:
                mean, _ = self._split_mean_std(self.actor(mlp_obs))
                return mean
            return self.actor(mlp_obs)

        def evaluate(self, obs, **kwargs) -> torch.Tensor:
            mlp_obs = self._encode(
                obs,
                self.critic_pose_groups,
                self.critic_rgb_group,
                self.critic_cam_pose_group,
                self.critic_dpt,
                self.critic_frame_pose_encoder,
                self.critic_frame_fusion,
                self.critic_robot_pose_encoder,
                self.critic_obs_normalizer,
                self.critic_num_frames,
            )
            return self.critic(mlp_obs)

        def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return self.distribution.log_prob(actions).sum(dim=-1)

        @property
        def action_mean(self) -> torch.Tensor:
            return self.distribution.mean

        @property
        def action_std(self) -> torch.Tensor:
            return self.distribution.stddev

        @property
        def entropy(self) -> torch.Tensor:
            return self.distribution.entropy().sum(dim=-1)

        def update_normalization(self, obs) -> None:
            if self.actor_obs_normalization and self.actor_pose_groups:
                actor_pose = self._encode_pose(obs, self.actor_pose_groups, nn.Identity())
                if actor_pose is not None:
                    self.actor_obs_normalizer.update(actor_pose)
            if self.critic_obs_normalization and self.critic_pose_groups:
                critic_pose = self._encode_pose(obs, self.critic_pose_groups, nn.Identity())
                if critic_pose is not None:
                    self.critic_obs_normalizer.update(critic_pose)

        def train(self, mode: bool = True):
            super().train(mode)
            self.vggt.eval()
            return self

        def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
            self._vggt_cache_key = None
            self._vggt_cache_value = None
            return None

        def get_hidden_states(self):
            return None


else:

    class DepthActorCriticCNN(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use DepthActorCriticCNN")

    class DepthActorCriticCNNRecurrent(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use DepthActorCriticCNNRecurrent")

    class ActorCriticCNNMaxpooling(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticCNNMaxpooling")

    class ActorCriticRGBCNNMaxpooling_2D(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticRGBCNNMaxpooling_2D")

    class ActorCriticRGBCNNMaxpooling_3D(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticRGBCNNMaxpooling_3D")

    class ActorCriticRGBCNNRecurrent(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticRGBCNNRecurrent")

    class ActorCriticRGBCNNPoseDelta(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticRGBCNNPoseDelta")

    class ActorCriticRGBVGGTFrozenPoseDelta(nn.Module):
        """Placeholder if rsl_rl is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rsl_rl is required to use ActorCriticRGBVGGTFrozenPoseDelta")

    ActorCriticRGBCNNMaxpooling = ActorCriticRGBCNNMaxpooling_2D


def register_custom_models() -> None:
    if ActorCriticCNN is None:
        raise ImportError("rsl_rl is required to register custom models")

    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner

    rsl_modules.DepthActorCriticCNN = DepthActorCriticCNN
    rsl_runner.DepthActorCriticCNN = DepthActorCriticCNN
    rsl_modules.DepthActorCriticCNNRecurrent = DepthActorCriticCNNRecurrent
    rsl_runner.DepthActorCriticCNNRecurrent = DepthActorCriticCNNRecurrent
    rsl_modules.ActorCriticCNNMaxpooling = ActorCriticCNNMaxpooling
    rsl_runner.ActorCriticCNNMaxpooling = ActorCriticCNNMaxpooling
    rsl_modules.ActorCriticRGBCNNMaxpooling_2D = ActorCriticRGBCNNMaxpooling_2D
    rsl_runner.ActorCriticRGBCNNMaxpooling_2D = ActorCriticRGBCNNMaxpooling_2D
    rsl_modules.ActorCriticRGBCNNMaxpooling_3D = ActorCriticRGBCNNMaxpooling_3D
    rsl_runner.ActorCriticRGBCNNMaxpooling_3D = ActorCriticRGBCNNMaxpooling_3D
    rsl_modules.ActorCriticRGBCNNRecurrent = ActorCriticRGBCNNRecurrent
    rsl_runner.ActorCriticRGBCNNRecurrent = ActorCriticRGBCNNRecurrent
    rsl_modules.ActorCriticRGBCNNPoseDelta = ActorCriticRGBCNNPoseDelta
    rsl_runner.ActorCriticRGBCNNPoseDelta = ActorCriticRGBCNNPoseDelta
    rsl_modules.ActorCriticRGBVGGTFrozenPoseDelta = ActorCriticRGBVGGTFrozenPoseDelta
    rsl_runner.ActorCriticRGBVGGTFrozenPoseDelta = ActorCriticRGBVGGTFrozenPoseDelta
    # rsl_modules.ActorCriticRGBCNNMaxpooling = ActorCriticRGBCNNMaxpooling
    # rsl_runner.ActorCriticRGBCNNMaxpooling = ActorCriticRGBCNNMaxpooling
