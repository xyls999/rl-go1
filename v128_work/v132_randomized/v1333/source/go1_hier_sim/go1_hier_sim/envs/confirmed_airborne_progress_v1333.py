"""Clearance-confirmed per-gap credit for airborne forward distance."""
from __future__ import annotations

import math

import torch


def confirmed_airborne_forward_distance_reward(
    *,
    current_air_time: torch.Tensor,
    forward_delta: torch.Tensor,
    local_x: torch.Tensor,
    gap_bounds: torch.Tensor,
    window_open: torch.Tensor,
    gap_mask: torch.Tensor,
    intrusion: torch.Tensor,
    hard_failure: torch.Tensor,
    cleared_gaps: torch.Tensor,
    pending_distance: torch.Tensor,
    paid_distance: torch.Tensor,
    approach_distance: float,
    landing_distance: float,
    short_width_max: float,
    max_gap_width: float,
    max_credit_m: float,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reward rate and updated pending/paid distance per gap.

    Airborne displacement is escrowed, not rewarded immediately. It is paid
    once when the runtime's stable far-pad clearance flag becomes true. A
    grounded touchdown before the far lip cancels pending credit, as do an
    intrusion or hard failure. Already-paid clearance credit is never refunded.
    ``paid_distance`` and ``pending_distance`` are independent for each gap.
    """
    if current_air_time.ndim != 2 or current_air_time.shape[1] != 4:
        raise ValueError("current_air_time must have shape [N, 4]")
    count = current_air_time.shape[0]
    vectors = (
        forward_delta,
        local_x,
        window_open,
        gap_mask,
        intrusion,
        hard_failure,
    )
    if any(value.shape != (count,) for value in vectors):
        raise ValueError("reward gates must have shape [N]")
    if gap_bounds.ndim != 3 or gap_bounds.shape[0] != count or gap_bounds.shape[2] != 3:
        raise ValueError("gap_bounds must have shape [N, K, 3]")
    expected_gap_shape = gap_bounds.shape[:2]
    if cleared_gaps.shape != expected_gap_shape:
        raise ValueError("cleared_gaps must have shape [N, K]")
    if pending_distance.shape != expected_gap_shape:
        raise ValueError("pending_distance must have shape [N, K]")
    if paid_distance.shape != expected_gap_shape:
        raise ValueError("paid_distance must have shape [N, K]")

    parameters = (
        approach_distance, landing_distance, short_width_max,
        max_gap_width, max_credit_m, control_dt,
    )
    if not all(math.isfinite(float(value)) for value in parameters):
        raise ValueError("confirmed distance reward parameters must be finite")
    if (approach_distance < 0 or landing_distance < 0 or short_width_max <= 0
            or max_gap_width < short_width_max or max_credit_m <= 0 or control_dt <= 0):
        raise ValueError("confirmed distance reward limits are invalid")
    if not all(torch.isfinite(value).all() for value in (
        current_air_time, forward_delta, local_x, intrusion,
        pending_distance, paid_distance,
    )):
        raise ValueError("confirmed distance reward state must be finite")
    if (pending_distance < 0).any() or (paid_distance < 0).any():
        raise ValueError("confirmed distance credit must be nonnegative")
    if gap_bounds.shape[1] == 0:
        zero = torch.zeros(count, dtype=current_air_time.dtype, device=current_air_time.device)
        return zero, pending_distance, paid_distance

    near_lip, far_lip, pad_end = gap_bounds.unbind(dim=-1)
    width = far_lip - near_lip
    valid_gap = (
        torch.isfinite(gap_bounds).all(dim=-1)
        & (width >= short_width_max)
        & (width <= max_gap_width + 1.0e-5)
        & (far_lip > near_lip)
        & (pad_end > far_lip)
    )
    x = local_x[:, None]
    active_window = window_open.bool() & gap_mask.bool()
    in_reward_window = (
        valid_gap
        & (x >= near_lip - approach_distance)
        & (x <= far_lip + 1.0e-5)
        & active_window[:, None]
        & ~cleared_gaps.bool()
    )
    in_cancel_window = (
        valid_gap
        & (x >= near_lip - approach_distance)
        & (x <= far_lip + landing_distance + 1.0e-5)
        & active_window[:, None]
    )
    reward_gap_index = in_reward_window.to(torch.int64).argmax(dim=-1)
    cancel_gap_index = in_cancel_window.to(torch.int64).argmax(dim=-1)
    has_reward_gap = in_reward_window.any(dim=-1)
    has_cancel_gap = in_cancel_window.any(dim=-1)
    rows = torch.arange(count, device=current_air_time.device)

    selected_width = torch.where(
        has_reward_gap, width[rows, reward_gap_index], torch.zeros(count, dtype=width.dtype, device=width.device)
    )
    selected_cap = torch.minimum(
        torch.full_like(selected_width, float(max_credit_m)), selected_width
    )
    remaining = (
        selected_cap
        - paid_distance[rows, reward_gap_index]
        - pending_distance[rows, reward_gap_index]
    ).clamp(min=0.0)
    all_feet_airborne = (current_air_time > 0.0).all(dim=-1)
    can_earn = (
        has_reward_gap
        & all_feet_airborne
        & (forward_delta > 0.0)
        & (intrusion <= 0.0)
        & ~hard_failure.bool()
    )
    earned = torch.minimum(forward_delta.to(current_air_time.dtype).clamp(min=0.0), remaining)
    earned = earned * can_earn.to(current_air_time.dtype)
    gap_indices = torch.arange(gap_bounds.shape[1], device=current_air_time.device)[None, :]
    reward_selector = (gap_indices == reward_gap_index[:, None]).to(current_air_time.dtype)
    pending = pending_distance + reward_selector * earned[:, None]

    cancel_selector = (gap_indices == cancel_gap_index[:, None]) & has_cancel_gap[:, None]
    near_side_touchdown = (
        ~all_feet_airborne[:, None]
        & (x < far_lip)
        & valid_gap
        & ~cleared_gaps.bool()
    )
    cancel_pending = (
        near_side_touchdown
        | (hard_failure.bool()[:, None] & ~cleared_gaps.bool())
        | ((intrusion > 0.0)[:, None] & cancel_selector & ~cleared_gaps.bool())
    )
    pending = torch.where(cancel_pending, torch.zeros_like(pending), pending)

    # Clearance is latched per gap by the existing stable far-pad scorer. It
    # may be reported after the robot has moved beyond the local reward window.
    safe_clearance = cleared_gaps.bool() & valid_gap
    committed = torch.where(safe_clearance, pending, torch.zeros_like(pending))
    pending = pending - committed
    paid = paid_distance + committed
    raw_reward = committed.sum(dim=-1) / float(control_dt)
    return raw_reward, pending, paid
