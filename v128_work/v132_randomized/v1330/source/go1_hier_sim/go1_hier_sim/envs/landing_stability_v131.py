"""Phase-aware reward for settling after a gap touchdown."""

import torch


def landing_stability_score(
    upright: torch.Tensor,
    ang_vel_xy: torch.Tensor,
    recent_landing: torch.Tensor,
    window_open: torch.Tensor,
    *,
    upright_std: float,
    angular_velocity_scale: float,
) -> torch.Tensor:
    """Reward upright, low-roll/pitch-rate posture after touchdown in a gap window."""
    if upright_std <= 0.0 or angular_velocity_scale <= 0.0:
        raise ValueError("landing reward scales must be positive")
    if ang_vel_xy.shape[:-1] != upright.shape or ang_vel_xy.shape[-1] != 2:
        raise ValueError("ang_vel_xy must have shape upright.shape + (2,)")

    upright_score = torch.exp(-(1.0 - upright) / upright_std)
    rate_score = torch.exp(-((ang_vel_xy / angular_velocity_scale).square()).sum(dim=-1))
    active = recent_landing.to(dtype=upright.dtype) * window_open.to(dtype=upright.dtype)
    return upright_score * rate_score * active
