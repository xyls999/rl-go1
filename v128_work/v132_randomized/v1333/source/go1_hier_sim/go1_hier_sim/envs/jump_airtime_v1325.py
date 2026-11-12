"""Bounded airtime shaping for useful, four-foot trench flights."""
from __future__ import annotations

import math

import torch


def gap_window_airtime_reward(
    current_air_time: torch.Tensor,
    forward_progress: torch.Tensor,
    window_open: torch.Tensor,
    gap_mask: torch.Tensor,
    intrusion: torch.Tensor,
    max_air_time: float,
) -> torch.Tensor:
    """Reward all-feet flight only in a valid gap window with forward progress.

    The minimum per-foot airtime makes the term zero whenever any foot is in
    contact. `forward_progress` scales credit in [0, 1], while the window,
    gap-family and no-intrusion gates prevent ordinary gait or falling into a
    trench from farming this term.
    """
    if current_air_time.ndim != 2 or current_air_time.shape[1] != 4:
        raise ValueError("current_air_time must have shape [N, 4]")
    count = current_air_time.shape[0]
    vectors = (forward_progress, window_open, gap_mask, intrusion)
    if any(value.shape != (count,) for value in vectors):
        raise ValueError("reward gates must have shape [N]")
    if not math.isfinite(float(max_air_time)) or max_air_time <= 0:
        raise ValueError("max_air_time must be finite and positive")

    all_feet_air_time = current_air_time.amin(dim=-1).clamp(min=0.0, max=max_air_time)
    progress_gate = forward_progress.clamp(min=0.0, max=1.0)
    eligible = (window_open > 0) & (gap_mask > 0) & (intrusion <= 0.0) & (progress_gate > 0)
    return all_feet_air_time * progress_gate * eligible.to(dtype=current_air_time.dtype)
