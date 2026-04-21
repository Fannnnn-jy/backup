from __future__ import annotations

from typing import Any, Callable, List

import torch
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
    # rsl_modules.ActorCriticRGBCNNMaxpooling = ActorCriticRGBCNNMaxpooling
    # rsl_runner.ActorCriticRGBCNNMaxpooling = ActorCriticRGBCNNMaxpooling
