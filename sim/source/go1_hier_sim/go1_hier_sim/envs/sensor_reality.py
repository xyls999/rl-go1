"""Training-only sensor realism: control latency, observation noise, scan dropout.

These knobs model the reality-gap sources identified for legged sim-to-real
transfer (Tan et al. 2018, "Sim-to-Real: Learning Agile Locomotion For
Quadruped Robots", sections V-A.b and V-B): feedback latency, IMU bias and
noise, and an observation whose real distribution differs from simulation.

Design rules
------------
* Every knob defaults to zero; then ``degrade`` is a bit-exact no-op, so
  evaluation, the regression gates and distillation keep previous behaviour.
* Only the *actor* observation is degraded. The privileged critic keeps seeing
  the clean teacher observation, which is the point of asymmetric training.
* Latency is sampled per episode, applied in whole control steps, and served
  from a history of the clean actor observation (the paper interpolates between
  the two bracketing samples; at 50 Hz a whole-step lookup is the same
  operation for integer delays).
* Scan dropout masks a fixed per-episode subset of height cells: the value and
  its confidence are both zeroed, i.e. "unknown space", never "free space".

Environment variables (read once at construction unless overridden):
    GO1_OBS_DELAY_STEPS    int    fixed latency in control steps      (0)
    GO1_OBS_DELAY_RANDOM   int    extra uniform 0..k steps per episode (0)
    GO1_OBS_NOISE_STD      float  per-step white noise on rate/joint  (0)
    GO1_OBS_BIAS_RAD       float  per-episode constant bias in rad    (0)
    GO1_OBS_SCAN_DROPOUT   float  fraction of height cells masked      (0)
"""

from __future__ import annotations

import os

import torch

# Actor observation layout shared with contracts.py:
#   [0:42]  proprio (omega 3, gravity 3, qrel 12, dq 12, previous action 12)
#   [42:45] command, [45:47] gait phase, [47:49] fixed heading
#   [49:49+H] height value * confidence, [49+H:49+2H] height confidence
PROPRIO_DIM = 42
PREFIX_DIM = 49
# Channel scale factors applied when the observation was packed. A physical
# bias must be divided by the same factor before it is added back.
OMEGA_SCALE = 0.25
DQ_SCALE = 0.05


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return max(0, int(float(raw)))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return max(0.0, float(raw))


class SensorReality:
    """Per-environment observation degradation with all-zero defaults."""

    def __init__(self, num_envs: int, device, obs_dim: int, scan_dim: int = 512,
                 delay_steps: int | None = None, delay_random: int | None = None,
                 noise_std: float | None = None, bias_rad: float | None = None,
                 scan_dropout: float | None = None, generator: torch.Generator | None = None):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.scan_dim = int(scan_dim)
        self.delay_steps = _env_int("GO1_OBS_DELAY_STEPS", 0) if delay_steps is None else max(0, int(delay_steps))
        self.delay_random = _env_int("GO1_OBS_DELAY_RANDOM", 0) if delay_random is None else max(0, int(delay_random))
        self.noise_std = _env_float("GO1_OBS_NOISE_STD", 0.0) if noise_std is None else max(0.0, float(noise_std))
        self.bias_rad = _env_float("GO1_OBS_BIAS_RAD", 0.0) if bias_rad is None else max(0.0, float(bias_rad))
        self.scan_dropout = _env_float("GO1_OBS_SCAN_DROPOUT", 0.0) if scan_dropout is None else max(0.0, float(scan_dropout))
        self.max_delay = self.delay_steps + self.delay_random
        self.enabled = bool(self.max_delay or self.noise_std or self.bias_rad or self.scan_dropout)
        self.scan_available = self.obs_dim >= PREFIX_DIM + 2 * self.scan_dim
        self._generator = generator
        self._history: list[torch.Tensor] = []
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bias = torch.zeros(self.num_envs, PROPRIO_DIM, device=self.device)
        self._drop = torch.zeros(self.num_envs, self.scan_dim, dtype=torch.bool, device=self.device)
        self._episode = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    # -- per-episode resampling -------------------------------------------------
    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        if high <= low:
            return torch.full(shape, low, device=self.device)
        return torch.rand(shape, generator=self._generator, device=self.device) * (high - low) + low

    def _resample(self, ids: torch.Tensor) -> None:
        count = int(ids.numel())
        if self.max_delay:
            extra = torch.randint(0, self.delay_random + 1, (count,),
                                  generator=self._generator, device=self.device)
            self._delay[ids] = self.delay_steps + extra
        if self.noise_std or self.bias_rad:
            bias = torch.zeros(count, PROPRIO_DIM, device=self.device)
            if self.bias_rad:
                bias[:, 3:6] = self._rand((count, 3), -self.bias_rad, self.bias_rad)      # gravity
                bias[:, 6:18] = self._rand((count, 12), -self.bias_rad, self.bias_rad)    # qrel
                bias[:, 0:3] = self._rand((count, 3), -self.bias_rad, self.bias_rad) * OMEGA_SCALE
                bias[:, 18:30] = self._rand((count, 12), -self.bias_rad, self.bias_rad) * DQ_SCALE
            self._bias[ids] = bias
        if self.scan_dropout and self.scan_available:
            self._drop[ids] = torch.rand(count, self.scan_dim, generator=self._generator,
                                         device=self.device) < self.scan_dropout

    def sync(self, episode_ids: torch.Tensor) -> None:
        """Resample episode-scoped parameters for environments that just reset."""
        if not self.enabled:
            return
        changed = torch.nonzero((episode_ids != self._episode), as_tuple=False).flatten()
        self._episode[:] = episode_ids
        if changed.numel():
            self._resample(changed)

    def reset(self, env_ids=None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._episode[ids] = -1
        self._history = []

    # -- degradation ------------------------------------------------------------
    def degrade(self, obs: torch.Tensor, episode_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled:
            return obs
        if episode_ids is not None:
            self.sync(episode_ids)
        self._history.append(obs.detach().clone())
        keep = self.max_delay + 1
        if len(self._history) > keep:
            del self._history[:-keep]
        out = obs
        if self.max_delay:
            out = obs.clone()
            for steps in range(self.max_delay + 1):
                ids = torch.nonzero(self._delay == steps, as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue
                source = self._history[-min(steps + 1, len(self._history))]
                out[ids] = source[ids]
        if self.bias_rad or self.noise_std:
            out = out.clone()
            out[:, :PROPRIO_DIM] = out[:, :PROPRIO_DIM] + self._bias
            if self.noise_std:
                shape = out[:, :PROPRIO_DIM].shape
                out[:, :PROPRIO_DIM] = out[:, :PROPRIO_DIM] + self.noise_std * torch.randn(
                    shape, generator=self._generator, device=self.device
                )
        if self.scan_dropout and self.scan_available:
            out = out.clone()
            value = out[:, PREFIX_DIM:PREFIX_DIM + self.scan_dim]
            confidence = out[:, PREFIX_DIM + self.scan_dim:PREFIX_DIM + 2 * self.scan_dim]
            value[self._drop] = 0.0
            confidence[self._drop] = 0.0
        return out
