"""Training-only bounded upward-preparation credit before an observable long lip.

The gap bounds and foot net force are simulator-only reward candidates, not
actor inputs or validated contact points. Credit never certifies a jump.
"""
from __future__ import annotations

import math
import os

import torch


class PrelaunchGradientState:
    def __init__(self, bounds: torch.Tensor, long_mask: torch.Tensor, scale: float = 1.):
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0.:
            raise ValueError("prelaunch scale must be a positive finite number")
        if bounds.ndim != 3 or bounds.shape[1:] != (24, 3):
            raise ValueError("prelaunch bounds must be [N,24,3]")
        if long_mask.shape != bounds.shape[:1] or long_mask.dtype != torch.bool:
            raise ValueError("long family mask must be boolean [N]")
        start, far, end = bounds.unbind(-1)
        finite = torch.isfinite(bounds).all(dim=-1)
        if (finite & ((start >= far) | (far > end))).any():
            raise ValueError("malformed prelaunch bounds")
        self.scale = float(scale)
        self.start = start
        # v38: the width floor was a fixed .26 m, which silently excluded the
        # narrow opening voids of the mixed route from any take-off credit.
        # The override is training-only and defaults to the published value.
        min_width = float(os.environ.get("GO1_V19_MIN_WIDTH", ".26"))
        self.valid = finite & ((far - start) > min_width) & long_mask[:, None]
        self.best_score = torch.zeros_like(start)

    def reset(self, ids: torch.Tensor) -> None:
        self.best_score[ids] = 0

    def step(self, *, local_x: torch.Tensor, forward_velocity: torch.Tensor,
             up_velocity: torch.Tensor, root_z: torch.Tensor, upright: torch.Tensor,
             command_speed: torch.Tensor, contacts: torch.Tensor,
             window_open: torch.Tensor, fallen: torch.Tensor,
             collision: torch.Tensor) -> torch.Tensor:
        # Improvements in this score telescope: at most `scale` units per gap,
        # even if the animal bobs or revisits the approach band many times.
        score = self.scale * ((up_velocity + .20) / .60).clamp(0., 1.)
        count = contacts.sum(-1)
        supported = (count >= 2) & (count < 4)
        eligible = (self.valid & window_open[:, None]
                    & (local_x[:, None] >= self.start - .55)
                    & (local_x[:, None] <= self.start - .15)
                    & (forward_velocity[:, None] >= .30)
                    & (command_speed[:, None] >= .35)
                    & (root_z[:, None] >= .29)
                    & (upright[:, None] >= .90)
                    & supported[:, None] & ~fallen[:, None] & ~collision[:, None])
        improvement = (score[:, None] - self.best_score).clamp(min=0.) * eligible
        self.best_score = torch.maximum(self.best_score, torch.where(
            eligible, score[:, None], self.best_score,
        ))
        return improvement.sum(dim=-1).clamp(max=self.scale)
