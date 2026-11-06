"""Gap-window masks and bounded gait regularizers for the V1330 candidate."""
from __future__ import annotations

import math

import torch


def gap_windows(
    bounds: torch.Tensor,
    local_x: torch.Tensor,
    approach_distance: float,
    landing_distance: float,
    short_width_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return near-any-gap and near-short-gap masks from simulator-only bounds.

    Bounds contain each trench's near lip, far lip, and far-pad end in local
    course coordinates. The short/long split matches ``gap_clearance_step``.
    """
    if bounds.ndim != 3 or bounds.shape[-1] != 3:
        raise ValueError("bounds must have shape [N, K, 3]")
    count = bounds.shape[0]
    if local_x.shape != (count,):
        raise ValueError("local_x must have shape [N]")
    if (not math.isfinite(float(approach_distance)) or approach_distance < 0
            or not math.isfinite(float(landing_distance)) or landing_distance < 0
            or not math.isfinite(float(short_width_max)) or short_width_max <= 0):
        raise ValueError("gap window distances and width threshold must be finite and valid")

    near_lip, far_lip, end = bounds.unbind(dim=-1)
    width = far_lip - near_lip
    valid = (
        torch.isfinite(bounds).all(dim=-1)
        & (width >= 0.08 - 1.0e-5)
        & (width <= 0.42 + 1.0e-5)
        & (end > far_lip)
    )
    x = local_x[:, None]
    in_window = valid & (x >= near_lip - approach_distance) & (x <= end + landing_distance)
    near_gap = in_window.any(dim=-1)
    near_short_gap = (in_window & (width < short_width_max)).any(dim=-1)
    return near_gap, near_short_gap


def fore_aft_swing_cost(
    foot_forward: torch.Tensor,
    contacts: torch.Tensor,
    near_gap: torch.Tensor,
    max_excursion: float,
) -> torch.Tensor:
    """Softly penalize excessive swing-foot fore/aft excursion away from gaps."""
    if foot_forward.ndim != 2 or foot_forward.shape[1] != 4:
        raise ValueError("foot_forward must have shape [N, 4]")
    count = foot_forward.shape[0]
    if contacts.shape != (count, 4) or near_gap.shape != (count,):
        raise ValueError("contacts and near_gap must have shapes [N, 4] and [N]")
    if not math.isfinite(float(max_excursion)) or max_excursion <= 0:
        raise ValueError("max_excursion must be finite and positive")

    excess = (foot_forward.abs() - max_excursion).clamp(min=0.0)
    swing = (~contacts.bool()).to(dtype=foot_forward.dtype)
    outside_gap = (~near_gap.bool()).to(dtype=foot_forward.dtype)
    return torch.tanh((excess.square() * swing).sum(dim=-1)) * outside_gap


def short_gap_clearance_target(
    ordinary_target: torch.Tensor,
    near_short_gap: torch.Tensor,
    forward_progress: torch.Tensor,
    target_height: float,
) -> torch.Tensor:
    """Raise the touchdown target modestly on moving short-gap approaches only."""
    if ordinary_target.ndim != 1:
        raise ValueError("ordinary_target must have shape [N]")
    count = ordinary_target.shape[0]
    if near_short_gap.shape != (count,) or forward_progress.shape != (count,):
        raise ValueError("gap/progress masks must have shape [N]")
    if not math.isfinite(float(target_height)) or target_height <= 0:
        raise ValueError("target_height must be finite and positive")

    active = near_short_gap.bool() & (forward_progress > 0.0)
    short_target = torch.full_like(ordinary_target, float(target_height))
    return torch.where(active, torch.maximum(ordinary_target, short_target), ordinary_target)


def short_gap_lift_reward(
    peak_clearance: torch.Tensor,
    touchdown_events: torch.Tensor,
    clearance_target: torch.Tensor,
    near_short_gap: torch.Tensor,
    forward_progress: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Reward leading-foot clearance on moving short-gap approaches only."""
    if peak_clearance.ndim != 2 or peak_clearance.shape[1] != 4:
        raise ValueError("peak_clearance must have shape [N, 4]")
    count = peak_clearance.shape[0]
    if (touchdown_events.shape != (count, 4)
            or clearance_target.shape != (count,)
            or near_short_gap.shape != (count,)
            or forward_progress.shape != (count,)):
        raise ValueError("short-gap lift inputs have inconsistent batch shapes")
    if not math.isfinite(float(std)) or std <= 0:
        raise ValueError("std must be finite and positive")

    leading_clearance = peak_clearance[:, :2]
    target = clearance_target[:, None]
    score = torch.exp(-((leading_clearance - target) / std).square())
    event = score * touchdown_events[:, :2].to(dtype=peak_clearance.dtype)
    reward = event.sum(dim=-1) / 2.0
    gate = near_short_gap.bool() & (forward_progress > 0.0)
    return reward * gate.to(dtype=peak_clearance.dtype)


def non_gap_airborne_cost(
    current_air_time: torch.Tensor,
    near_gap: torch.Tensor,
    grace_seconds: float,
    ramp_seconds: float,
) -> torch.Tensor:
    """Penalize sustained four-foot flight only outside gap approach/landing windows."""
    if current_air_time.ndim != 2 or current_air_time.shape[1] != 4:
        raise ValueError("current_air_time must have shape [N, 4]")
    count = current_air_time.shape[0]
    if near_gap.shape != (count,):
        raise ValueError("near_gap must have shape [N]")
    if (not math.isfinite(float(grace_seconds)) or grace_seconds < 0
            or not math.isfinite(float(ramp_seconds)) or ramp_seconds <= 0):
        raise ValueError("grace_seconds must be non-negative and ramp_seconds positive")

    all_feet_air = current_air_time.clamp(min=0.0).amin(dim=-1)
    sustained = ((all_feet_air - grace_seconds) / ramp_seconds).clamp(0.0, 1.0)
    outside_gap = (~near_gap.bool()).to(dtype=current_air_time.dtype)
    return sustained * outside_gap
