"""Sim-to-real robustness curriculum for the hierarchical v2 system.

Mirrors the fwwb mixing schedule: the actor population is split into clean,
mild, medium and a small hard fraction whose sizes grow with the update index,
while the critic always consumes clean state.  The module only produces the
per-environment disturbance parameters; applying them (observation noise, UWB
packet loss/staleness, actuator gain error, one-step action delay) stays in the
runtime so the deployment contract remains explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RobustnessLevel:
    name: str
    uwb_noise_std: float
    uwb_dropout: float
    action_delay_steps: int
    action_gain_error: float
    obs_noise_std: float


LEVELS: dict[str, RobustnessLevel] = {
    "clean": RobustnessLevel("clean", 0.0, 0.0, 0, 0.0, 0.0),
    "mild": RobustnessLevel("mild", 0.02, 0.01, 0, 0.02, 0.005),
    "medium": RobustnessLevel("medium", 0.05, 0.03, 1, 0.05, 0.010),
    "hard": RobustnessLevel("hard", 0.10, 0.08, 1, 0.10, 0.020),
}

MIXES: tuple[tuple[int, dict[str, float]], ...] = (
    (0, {"clean": 1.0}),
    (25, {"clean": 0.50, "mild": 0.50}),
    (100, {"clean": 0.35, "mild": 0.35, "medium": 0.25, "hard": 0.05}),
    (250, {"clean": 0.30, "mild": 0.35, "medium": 0.25, "hard": 0.10}),
)


def mix_for(iteration: int) -> dict[str, float]:
    """Return the population mix for a learning iteration."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    selected = MIXES[0][1]
    for threshold, mix in MIXES:
        if iteration >= threshold:
            selected = mix
    total = sum(selected.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"robustness mix must sum to one, got {total}")
    return dict(selected)


@dataclass
class RobustnessBatch:
    uwb_noise_std: torch.Tensor          # [N]
    uwb_dropout: torch.Tensor            # [N]
    action_delay_steps: torch.Tensor     # [N] long
    command_gain: torch.Tensor           # [N,3] velocity-command gain error
    obs_noise_std: torch.Tensor          # [N]
    level_index: torch.Tensor            # [N] long
    levels: tuple[str, ...] = field(default_factory=tuple)


class RobustnessCurriculum:
    """Deterministic per-iteration sampling of the disturbance population."""

    def __init__(self, num_envs: int, device, seed: int = 0):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._levels = tuple(LEVELS)

    def sample(self, iteration: int) -> RobustnessBatch:
        mix = mix_for(iteration)
        weights = torch.tensor([mix.get(name, 0.0) for name in self._levels],
                               device=self.device, dtype=torch.float32)
        if float(weights.sum()) <= 0.0:
            raise ValueError("robustness mix has no mass")
        weights = weights / weights.sum()
        index = torch.multinomial(
            weights, self.num_envs, replacement=True, generator=self._generator
        )

        def gather(attr: str, dtype: torch.dtype) -> torch.Tensor:
            values = torch.tensor([getattr(LEVELS[name], attr) for name in self._levels],
                                  device=self.device, dtype=dtype)
            return values[index]

        gain_error = gather("action_gain_error", torch.float32)
        gain = 1.0 + (torch.rand(self.num_envs, 3, device=self.device,
                                 generator=self._generator) * 2.0 - 1.0) * gain_error[:, None]
        return RobustnessBatch(
            uwb_noise_std=gather("uwb_noise_std", torch.float32),
            uwb_dropout=gather("uwb_dropout", torch.float32),
            action_delay_steps=gather("action_delay_steps", torch.long),
            command_gain=gain,
            obs_noise_std=gather("obs_noise_std", torch.float32),
            level_index=index,
            levels=self._levels,
        )

    def describe(self, iteration: int) -> dict[str, float]:
        """Numeric summary for TensorBoard logging."""
        mix = mix_for(iteration)
        return {f"robustness/{name}": value for name, value in mix.items()}
