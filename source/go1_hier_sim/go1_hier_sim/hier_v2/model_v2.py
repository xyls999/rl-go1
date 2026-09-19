"""fwwb-style encoder policies for the hierarchical v2 system.

The design follows the fwwb ``ActorCriticEncoder``: geometry (height scan or
range scan) is compressed by a dedicated MLP encoder with an L2-normalized
latent, concatenated with the proprioceptive prefix, and consumed by a
recurrent core.  The critic owns a *separate* encoder so privileged value
gradients never reshape the actor's scan representation.

Differences from the fwwb reference that are required by this project:
  * the recurrent core is kept (``is_recurrent = True``) so RSL-RL rollout
    storage, hidden-state reset per environment and padded-sequence training
    continue to work unchanged;
  * observation normalisation stays fixed in ``contracts.py`` instead of being
    estimated at runtime.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from ..contracts import LOCO_OBS_DIM, LOCO_PREFIX_DIM, NAV_OBS_DIM, NAV_PREFIX_DIM
from ..models import ActorCore, CriticCore


ENCODER_HIDDEN = (512, 256)
LATENT_DIM = 32
GRU_HIDDEN = 128


class L2Norm(nn.Module):
    """Project the scan latent onto the unit sphere, as in the fwwb reference."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, p=2.0, dim=-1)


def _encoder(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, ENCODER_HIDDEN[0]), nn.ELU(),
        nn.Linear(ENCODER_HIDDEN[0], ENCODER_HIDDEN[1]), nn.ELU(),
        nn.Linear(ENCODER_HIDDEN[1], LATENT_DIM), L2Norm(),
    )


class EncoderActorCore(nn.Module):
    """Actor with a scan encoder, a recurrent core and a 12-D / 3-D head."""

    variant = "v2"

    def __init__(self, stage: str = "locomotion"):
        super().__init__()
        if stage not in ("locomotion", "navigation"):
            raise ValueError(f"unknown actor stage: {stage}")
        self.stage = stage
        self.prefix_dim = LOCO_PREFIX_DIM if stage == "locomotion" else NAV_PREFIX_DIM
        self.obs_dim = LOCO_OBS_DIM if stage == "locomotion" else NAV_OBS_DIM
        self.num_actions = 12 if stage == "locomotion" else 3
        self.num_geometry = (self.obs_dim - self.prefix_dim) // 2
        self.num_scan = 2 * self.num_geometry
        self.scan_encoder = _encoder(self.num_scan)
        self.gru = nn.GRU(self.prefix_dim + LATENT_DIM, GRU_HIDDEN)
        widths = (256, 128) if stage == "locomotion" else (128, 64)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, widths[0]), nn.ELU(),
            nn.Linear(widths[0], widths[1]), nn.ELU(),
            nn.Linear(widths[1], self.num_actions),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        values = obs[..., self.prefix_dim:self.prefix_dim + self.num_geometry]
        confidence = obs[..., self.prefix_dim + self.num_geometry:self.obs_dim]
        scan = torch.cat((values * confidence, confidence), dim=-1)
        return torch.cat((obs[..., :self.prefix_dim], self.scan_encoder(scan)), dim=-1)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        if obs.dim() not in (2, 3) or obs.shape[-1] != self.obs_dim:
            raise ValueError("actor observation shape is incompatible")
        single = obs.dim() == 2
        seq = obs.unsqueeze(0) if single else obs
        if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq.shape[1] \
                or hidden.shape[2] != GRU_HIDDEN:
            raise ValueError(f"actor hidden state must have shape [1,N,{GRU_HIDDEN}]")
        output, next_hidden = self.gru(self.encode(seq), hidden)
        mean = self.head(output)
        return (mean.squeeze(0) if single else mean), next_hidden


class EncoderCriticCore(nn.Module):
    """Critic with an independent scan encoder and seven privileged fields."""

    variant = "v2"

    def __init__(self, stage: str):
        super().__init__()
        if stage not in ("locomotion", "navigation"):
            raise ValueError(f"unknown critic stage: {stage}")
        self.stage = stage
        self.prefix_dim = LOCO_PREFIX_DIM if stage == "locomotion" else NAV_PREFIX_DIM
        self.actor_dim = LOCO_OBS_DIM if stage == "locomotion" else NAV_OBS_DIM
        self.num_geometry = (self.actor_dim - self.prefix_dim) // 2
        self.num_scan = 2 * self.num_geometry
        self.scan_encoder = _encoder(self.num_scan)
        self.gru = nn.GRU(self.prefix_dim + LATENT_DIM + 7, GRU_HIDDEN)
        widths = (256, 128) if stage == "locomotion" else (128, 64)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, widths[0]), nn.ELU(),
            nn.Linear(widths[0], widths[1]), nn.ELU(),
            nn.Linear(widths[1], 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        values = obs[..., self.prefix_dim:self.prefix_dim + self.num_geometry]
        confidence = obs[..., self.prefix_dim + self.num_geometry:self.actor_dim]
        scan = torch.cat((values * confidence, confidence), dim=-1)
        return torch.cat(
            (obs[..., :self.prefix_dim], self.scan_encoder(scan), obs[..., self.actor_dim:]),
            dim=-1,
        )

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        if obs.dim() not in (2, 3) or obs.shape[-1] != self.actor_dim + 7:
            raise ValueError("critic observation shape is incompatible")
        single = obs.dim() == 2
        seq = obs.unsqueeze(0) if single else obs
        if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq.shape[1] \
                or hidden.shape[2] != GRU_HIDDEN:
            raise ValueError(f"critic hidden state must have shape [1,N,{GRU_HIDDEN}]")
        output, next_hidden = self.gru(self.encode(seq), hidden)
        value = self.head(output)
        return (value.squeeze(0) if single else value), next_hidden


class HierActorCriticV2(nn.Module):
    """RSL-RL 3.1.2 adapter for the encoder policies."""

    is_recurrent = True

    def __init__(self, obs, obs_groups, num_actions, stage, init_noise_std=1.0,
                 min_noise_std=0.05, max_noise_std=1.5):
        super().__init__()
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")
        if (not math.isfinite(min_noise_std) or not math.isfinite(max_noise_std)
                or not 0 < min_noise_std < max_noise_std):
            raise ValueError("noise limits must be finite and satisfy 0 < min < max")
        self.obs_groups = {key: list(obs_groups[key]) for key in ("policy", "critic")}
        self.actor = EncoderActorCore(stage)
        self.critic = EncoderCriticCore(stage)
        if num_actions != self.actor.num_actions:
            raise ValueError("action dimension does not match stage")
        actor_obs, critic_obs = self.get_actor_obs(obs), self.get_critic_obs(obs)
        if actor_obs.ndim != 2 or actor_obs.shape[-1] != self.actor.obs_dim:
            raise ValueError("policy observation groups do not match the stage contract")
        if critic_obs.shape != (actor_obs.shape[0], self.actor.obs_dim + 7):
            raise ValueError("critic observation groups do not match the stage contract")
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))
        self._min_log_std = math.log(min_noise_std)
        self._max_log_std = math.log(max_noise_std)
        self.register_buffer("hidden_actor", actor_obs.new_zeros(1, actor_obs.shape[0], GRU_HIDDEN),
                             persistent=False)
        self.register_buffer("hidden_critic", actor_obs.new_zeros(1, actor_obs.shape[0], GRU_HIDDEN),
                             persistent=False)
        self.distribution = None

    def get_actor_obs(self, obs):
        return torch.cat([obs[key] for key in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[key] for key in self.obs_groups["critic"]], dim=-1)

    def _run(self, core, observations, masks, hidden_state, actor):
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            if hidden_state is None or observations.ndim != 3 or masks.shape != observations.shape[:2]:
                raise ValueError("padded recurrent training requires masks and initial hidden state")
            output, _ = core(observations, hidden_state)
            return unpad_trajectories(output, masks)
        if hidden_state is not None:
            raise ValueError("explicit training hidden state requires trajectory masks")
        if observations.ndim != 2:
            raise ValueError("rollout observations must be [N,D]")
        current = self.hidden_actor if actor else self.hidden_critic
        output, new_hidden = core(observations, current)
        if actor:
            self.hidden_actor = new_hidden.detach()
        else:
            self.hidden_critic = new_hidden.detach()
        return output

    def act(self, obs, masks=None, hidden_state=None):
        mean = self._run(self.actor, self.get_actor_obs(obs), masks, hidden_state, True)
        bounded_std = self.log_std.clamp(self._min_log_std, self._max_log_std).exp()
        self.distribution = Normal(mean, bounded_std.expand_as(mean), validate_args=True)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._run(self.actor, self.get_actor_obs(obs), None, None, True)

    def evaluate(self, obs, masks=None, hidden_state=None):
        return self._run(self.critic, self.get_critic_obs(obs), masks, hidden_state, False)

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError("act must initialize the action distribution")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            raise RuntimeError("act must initialize the action distribution")
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError("act must initialize the action distribution")
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            raise RuntimeError("act must initialize the action distribution")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        if dones is None:
            self.hidden_actor.zero_()
            self.hidden_critic.zero_()
            return
        done = dones.reshape(-1).bool()
        if done.numel() != self.hidden_actor.shape[1]:
            raise ValueError("reset mask does not match environment count")
        self.hidden_actor[:, done] = 0
        self.hidden_critic[:, done] = 0

    def get_hidden_states(self):
        return self.hidden_actor.clone(), self.hidden_critic.clone()

    def update_normalization(self, obs):
        for key in set(self.obs_groups["policy"] + self.obs_groups["critic"]):
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f"nonfinite policy observation: {key}")

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


def detect_actor_variant(state: dict) -> str:
    """Return ``v2`` for encoder checkpoints, ``v1`` for the original cores."""
    if any(key.startswith("scan_encoder.") for key in state):
        return "v2"
    return "v1"


def build_actor(kind: str, state: dict):
    """Instantiate the actor core matching a saved state dict."""
    if detect_actor_variant(state) == "v2":
        return EncoderActorCore(kind)
    return ActorCore(kind)


def build_critic(kind: str, state: dict):
    if detect_actor_variant(state) == "v2":
        return EncoderCriticCore(kind)
    return CriticCore(kind)
