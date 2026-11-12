"""Bounded per-gap reward for forward distance during a real four-foot flight."""
from __future__ import annotations

import math

import torch


def airborne_forward_distance_reward(
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
    credited_distance: torch.Tensor,
    approach_distance: float,
    landing_distance: float,
    short_width_max: float,
    max_gap_width: float,
    max_credit_m: float,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw m/s reward and updated per-gap credited distance.

    The environment RewardManager multiplies raw terms by ``control_dt``;
    returning positive forward speed therefore integrates to metres. Credit is
    allowed only while all feet are airborne and the root is between the
    approach zone and a long gap's far lip. The cumulative credited metres are
    capped independently for each gap by both ``max_credit_m`` and that gap's
    measured width. An unconfirmed attempt that intrudes, falls, collides, or
    leaves bounds refunds its pending distance credit once; a stable per-gap
    clearance record preserves it.
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
    if credited_distance.shape != gap_bounds.shape[:2]:
        raise ValueError("credited_distance must have shape [N, K]")
    if cleared_gaps.shape != gap_bounds.shape[:2]:
        raise ValueError("cleared_gaps must have shape [N, K]")
    if gap_bounds.shape[1] == 0:
        return torch.zeros(count, dtype=current_air_time.dtype, device=current_air_time.device), credited_distance

    parameters = (approach_distance, landing_distance, short_width_max, max_gap_width, max_credit_m, control_dt)
    if not all(math.isfinite(float(value)) for value in parameters):
        raise ValueError("distance reward parameters must be finite")
    if (approach_distance < 0 or landing_distance < 0 or short_width_max <= 0
            or max_gap_width < short_width_max or max_credit_m <= 0 or control_dt <= 0):
        raise ValueError("distance reward limits are invalid")
    if not all(torch.isfinite(value).all() for value in (
        current_air_time, forward_delta, local_x, intrusion, credited_distance
    )):
        raise ValueError("distance reward state must be finite")
    if (credited_distance < 0).any():
        raise ValueError("credited_distance must be nonnegative")

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
    window_gate = window_open.bool() & gap_mask.bool()
    in_reward_window = (
        valid_gap
        & (x >= near_lip - approach_distance)
        & (x <= far_lip + 1.0e-5)
        & window_gate[:, None]
    )
    in_failure_window = (
        valid_gap
        & (x >= near_lip - approach_distance)
        & (x <= far_lip + landing_distance + 1.0e-5)
        & window_gate[:, None]
    )

    reward_gap_index = in_reward_window.to(torch.int64).argmax(dim=-1)
    failure_gap_index = in_failure_window.to(torch.int64).argmax(dim=-1)
    has_reward_gap = in_reward_window.any(dim=-1)
    has_failure_gap = in_failure_window.any(dim=-1)
    rows = torch.arange(count, device=current_air_time.device)

    selected_width = torch.where(
        has_reward_gap, width[rows, reward_gap_index], torch.zeros(count, dtype=width.dtype, device=width.device)
    )
    selected_cap = torch.minimum(
        torch.full_like(selected_width, float(max_credit_m)), selected_width
    )
    selected_credit = credited_distance[rows, reward_gap_index]
    remaining_credit = (selected_cap - selected_credit).clamp(min=0.0)
    forward_metres = forward_delta.to(dtype=current_air_time.dtype).clamp(min=0.0)
    all_feet_airborne = (current_air_time > 0.0).all(dim=-1)
    can_earn = (
        has_reward_gap
        & all_feet_airborne
        & (forward_delta > 0.0)
        & (intrusion <= 0.0)
        & ~hard_failure.bool()
    )
    earned_metres = torch.minimum(forward_metres, remaining_credit)
    earned_metres = earned_metres * can_earn.to(dtype=current_air_time.dtype)

    failure_credit = credited_distance[rows, failure_gap_index]
    failed_attempt = hard_failure.bool() | (intrusion > 0.0)
    was_cleared = cleared_gaps[rows, failure_gap_index].bool()
    refund = has_failure_gap & failed_attempt & ~was_cleared
    refunded_metres = failure_credit * refund.to(dtype=current_air_time.dtype)

    gap_indices = torch.arange(gap_bounds.shape[1], device=current_air_time.device)[None, :]
    reward_selector = (gap_indices == reward_gap_index[:, None]).to(current_air_time.dtype)
    failure_selector = (gap_indices == failure_gap_index[:, None]).to(current_air_time.dtype)
    updated_credit = (
        credited_distance
        + reward_selector * earned_metres[:, None]
        - failure_selector * refunded_metres[:, None]
    ).clamp(min=0.0)
    raw_reward = (earned_metres - refunded_metres) / float(control_dt)
    return raw_reward, updated_credit
