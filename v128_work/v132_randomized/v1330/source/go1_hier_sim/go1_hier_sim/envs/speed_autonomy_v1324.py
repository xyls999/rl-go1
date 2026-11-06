"""Direction-only command cue for the V132.4 speed-autonomy diagnostic."""
from __future__ import annotations

import math

import torch


def autonomous_progress_reference(command_speed: torch.Tensor, max_progress_speed: float) -> torch.Tensor:
    """Use the configured saturation speed, not a commanded speed target."""
    if not math.isfinite(float(max_progress_speed)) or max_progress_speed <= 0:
        raise ValueError("maximum progress speed must be finite and positive")
    return torch.full_like(command_speed, float(max_progress_speed))


def direction_only_command(command: torch.Tensor) -> torch.Tensor:
    """Keep forward/backward direction while removing commanded speed and yaw."""
    if command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("direction-only command must have shape (num_envs, 3)")
    result = torch.zeros_like(command)
    result[:, 0] = torch.sign(command[:, 0])
    return result
