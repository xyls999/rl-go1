"""Training-only brake shaping for tire-field slots (default off, stateless, bounded).

Measured failure signature (2026-10-17): every tire-field fall had base speed
0.5-0.77 m/s under commands 0.58-0.77, while the pure-tire v15 teacher crossed
the same field at a ~0.565 m/s mean. This additive signal pays for holding the
capped target and charges overspeed beyond cap+margin; it is shaping only and
certifies no stability.
"""
from __future__ import annotations

import torch


def parse_brake_families(raw: str | None) -> frozenset[str]:
    """Parse GO1_V23_TIRE_BRAKE_FAMILIES; default keeps the tire-field-only semantics."""
    if raw is None or raw == '':
        return frozenset({'tire_field'})
    parts = raw.split(',')
    if any(p.strip() == '' for p in parts):
        raise ValueError('brake family list has an empty entry')
    names = frozenset(p.strip() for p in parts)
    if len(names) != len(parts):
        raise ValueError('brake family list has duplicates')
    return names


def tire_brake_adjustment(*, forward_velocity: torch.Tensor, command_speed: torch.Tensor,
                          tire_mask: torch.Tensor, std: float, cap: float = .60,
                          margin: float = .08, span: float = .30,
                          penalty_scale: float = .5) -> torch.Tensor:
    if std <= 0. or span <= 0.:
        raise ValueError("brake shaping needs positive std and span")
    target = torch.clamp(command_speed, max=cap)
    brake_bonus = (torch.exp(-(forward_velocity - target).square() / std ** 2)
                   - torch.exp(-(forward_velocity - command_speed).square() / std ** 2))
    overspeed = penalty_scale * ((forward_velocity - cap - margin) / span).clamp(0., 1.)
    return torch.where(tire_mask, brake_bonus - overspeed,
                       torch.zeros_like(forward_velocity)).clamp(-1.5, 1.)
