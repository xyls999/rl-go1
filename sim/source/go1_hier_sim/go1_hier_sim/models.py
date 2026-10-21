"""Recurrent policies with complete, explicitly stateful deployment graphs."""

import math

import torch
from torch import nn
from torch.distributions import Normal

from .contracts import LOCO_OBS_DIM, LOCO_PREFIX_DIM, NAV_OBS_DIM, TERRAIN_HEIGHT_DIM


LEGACY_LOCO_PREFIX_DIM = 47


def migrate_locomotion_heading_state(state):
    """Insert two zero-weight heading columns into a schema-v3 Stage-1 policy.

    Geometry and privileged inputs followed the old 47-D prefix.  Shifting
    those existing columns by two and zero-initializing only the new heading
    columns preserves the complete V17 policy function at migration time.
    """
    migrated = dict(state)
    # Evaluation/deployment checkpoints store the standalone locomotion actor
    # with unprefixed keys, while on-policy warm starts store actor/critic keys.
    if "gru.weight_ih_l0" in migrated and "actor.gru.weight_ih_l0" not in migrated:
        old = migrated["gru.weight_ih_l0"]
        old_width = LEGACY_LOCO_PREFIX_DIM + 32
        new_width = LOCO_PREFIX_DIM + 32
        if old.ndim != 2 or old.shape[1] != old_width:
            raise ValueError(
                f"legacy locomotion tensor gru.weight_ih_l0 has input width "
                f"{old.shape[1] if old.ndim == 2 else 'invalid'}, expected {old_width}"
            )
        new = old.new_zeros(old.shape[0], new_width)
        new[:, :LEGACY_LOCO_PREFIX_DIM] = old[:, :LEGACY_LOCO_PREFIX_DIM]
        new[:, LOCO_PREFIX_DIM:] = old[:, LEGACY_LOCO_PREFIX_DIM:]
        migrated["gru.weight_ih_l0"] = new
        return migrated
    specifications = {
        "actor.gru.weight_ih_l0": (LEGACY_LOCO_PREFIX_DIM + 32, LOCO_PREFIX_DIM + 32),
        "critic.gru.weight_ih_l0": (LEGACY_LOCO_PREFIX_DIM + 32 + 7, LOCO_PREFIX_DIM + 32 + 7),
    }
    for name, (old_width, new_width) in specifications.items():
        if name not in migrated:
            raise ValueError(f"legacy locomotion checkpoint is missing {name}")
        old = migrated[name]
        if old.ndim != 2:
            raise ValueError(f"legacy locomotion tensor {name} must be a matrix")
        if old.shape[1] == new_width:
            continue
        if old.shape[1] != old_width:
            raise ValueError(
                f"legacy locomotion tensor {name} has input width {old.shape[1]}, expected {old_width}"
            )
        new = old.new_zeros(old.shape[0], new_width)
        new[:, :LEGACY_LOCO_PREFIX_DIM] = old[:, :LEGACY_LOCO_PREFIX_DIM]
        new[:, LOCO_PREFIX_DIM:] = old[:, LEGACY_LOCO_PREFIX_DIM:]
        migrated[name] = new
    return migrated


class ActorCore(nn.Module):
    """Fixed observation layout; geometry occupies the final actor fields."""

    def __init__(self, stage: str = "locomotion"):
        super().__init__()
        if stage not in ("locomotion", "navigation"):
            raise ValueError(f"unknown actor stage: {stage}")
        self.stage = stage
        self.prefix_dim = LOCO_PREFIX_DIM if stage == "locomotion" else 49
        self.obs_dim = LOCO_OBS_DIM if stage == "locomotion" else NAV_OBS_DIM
        self.num_actions = 12 if stage == "locomotion" else 3
        self.num_geometry = (self.obs_dim - self.prefix_dim) // 2
        self.encoder = nn.Sequential(nn.Linear(self.obs_dim - self.prefix_dim, 128), nn.ELU(),
                                     nn.Linear(128, 32), nn.ELU())
        self.gru = nn.GRU(self.prefix_dim + 32, 128)
        widths = (256, 128) if stage == "locomotion" else (128, 64)
        self.head = nn.Sequential(nn.Linear(128, widths[0]), nn.ELU(),
                                  nn.Linear(widths[0], widths[1]), nn.ELU(),
                                  nn.Linear(widths[1], self.num_actions))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() not in (2, 3) or obs.shape[-1] != self.obs_dim:
            raise ValueError("actor observation shape is incompatible")
        single = obs.dim() == 2
        seq = obs.unsqueeze(0) if single else obs
        if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq.shape[1] or hidden.shape[2] != 128:
            raise ValueError("actor hidden state must have shape [1,N,128]")
        values = seq[..., self.prefix_dim:self.prefix_dim + self.num_geometry]
        confidence = seq[..., self.prefix_dim + self.num_geometry:]
        geometry = self.encoder(torch.cat((values * confidence, confidence), dim=-1))
        encoded = torch.cat((seq[..., :self.prefix_dim], geometry), dim=-1)
        output, next_hidden = self.gru(encoded, hidden)
        mean = self.head(output)
        return (mean.squeeze(0) if single else mean), next_hidden


class CriticCore(nn.Module):
    """Asymmetric critic encodes actor geometry independently and appends seven privileged fields."""

    def __init__(self, stage: str):
        super().__init__()
        if stage not in ("locomotion", "navigation"):
            raise ValueError(f"unknown critic stage: {stage}")
        self.prefix_dim = LOCO_PREFIX_DIM if stage == "locomotion" else 49
        self.actor_dim = LOCO_OBS_DIM if stage == "locomotion" else NAV_OBS_DIM
        self.num_geometry = (self.actor_dim - self.prefix_dim) // 2
        self.encoder = nn.Sequential(nn.Linear(self.actor_dim - self.prefix_dim, 128), nn.ELU(),
                                     nn.Linear(128, 32), nn.ELU())
        self.gru = nn.GRU(self.prefix_dim + 32 + 7, 128)
        widths = (256, 128) if stage == "locomotion" else (128, 64)
        self.head = nn.Sequential(nn.Linear(128, widths[0]), nn.ELU(),
                                  nn.Linear(widths[0], widths[1]), nn.ELU(), nn.Linear(widths[1], 1))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() not in (2, 3) or obs.shape[-1] != self.actor_dim + 7:
            raise ValueError("critic observation shape is incompatible")
        single = obs.dim() == 2
        seq = obs.unsqueeze(0) if single else obs
        if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq.shape[1] or hidden.shape[2] != 128:
            raise ValueError("critic hidden state must have shape [1,N,128]")
        values = seq[..., self.prefix_dim:self.prefix_dim + self.num_geometry]
        confidence = seq[..., self.prefix_dim + self.num_geometry:self.actor_dim]
        geometry = self.encoder(torch.cat((values * confidence, confidence), dim=-1))
        encoded = torch.cat((seq[..., :self.prefix_dim], geometry, seq[..., self.actor_dim:]), dim=-1)
        output, next_hidden = self.gru(encoded, hidden)
        value = self.head(output)
        return (value.squeeze(0) if single else value), next_hidden


class HierActorCritic(nn.Module):
    """RSL-RL 3.1.2 policy adapter; padded training batches never overwrite rollout memory."""

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
        self.actor = ActorCore(stage)
        self.critic = CriticCore(stage)
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
        self.register_buffer("hidden_actor", actor_obs.new_zeros(1, actor_obs.shape[0], 128), persistent=False)
        self.register_buffer("hidden_critic", actor_obs.new_zeros(1, actor_obs.shape[0], 128), persistent=False)
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
        # PPO stores pre-action states. Clones prevent later resets from mutating them.
        return self.hidden_actor.clone(), self.hidden_critic.clone()

    def update_normalization(self, obs):
        """Normalization is fixed in contracts.py; validate observations instead of updating statistics."""
        for key in set(self.obs_groups["policy"] + self.obs_groups["critic"]):
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f"nonfinite policy observation: {key}")

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class VisionBranch(nn.Module):
    """One camera stream with separate terrain or obstacle temporal memory."""

    def __init__(self, kind: str):
        super().__init__()
        if kind not in ("terrain", "obstacle"):
            raise ValueError(f"unknown visual branch: {kind}")
        self.kind = kind
        self.num_values = TERRAIN_HEIGHT_DIM if kind == "terrain" else 72
        self.cnn = nn.Sequential(nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ELU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
                                 nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(),
                                 nn.Linear(384, 128), nn.ELU())
        self.gru = nn.GRU(173, 256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2 * self.num_values))

    def forward(self, image: torch.Tensor, state: torch.Tensor,
                hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.dim() not in (4, 5) or image.shape[-3:] != (2, 60, 80):
            raise ValueError("visual image must be [N,2,60,80] or [T,N,2,60,80]")
        single = image.dim() == 4
        seq_image = image.unsqueeze(0) if single else image
        seq_state = state.unsqueeze(0) if single else state
        if seq_state.dim() != 3 or seq_state.shape[0] != seq_image.shape[0] or seq_state.shape[1] != seq_image.shape[1] or seq_state.shape[2] != 45:
            raise ValueError("visual state must match image batches and have 45 fields")
        if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq_image.shape[1] or hidden.shape[2] != 256:
            raise ValueError("visual hidden state must have shape [1,N,256]")
        features = self.cnn(seq_image.flatten(0, 1)).reshape(seq_image.shape[0], seq_image.shape[1], 128)
        output, next_hidden = self.gru(torch.cat((features, seq_state), dim=-1), hidden)
        raw = self.head(output)
        values = torch.tanh(raw[..., :self.num_values]) if self.kind == "terrain" else torch.sigmoid(raw[..., :self.num_values])
        prediction = torch.cat((values, torch.sigmoid(raw[..., self.num_values:])), dim=-1)
        return (prediction.squeeze(0) if single else prediction), next_hidden


class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.terrain = VisionBranch("terrain")
        self.obstacle = VisionBranch("obstacle")

    def forward(self, image: torch.Tensor, state: torch.Tensor,
                hidden_terrain: torch.Tensor, hidden_obstacle: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terrain, next_terrain = self.terrain(image, state, hidden_terrain)
        obstacle, next_obstacle = self.obstacle(image, state, hidden_obstacle)
        return terrain, obstacle, next_terrain, next_obstacle
