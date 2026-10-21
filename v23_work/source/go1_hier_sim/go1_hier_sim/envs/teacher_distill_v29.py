"""Multi-teacher DAgger gate for the eleven-family student (Stage-B/C design).

Teachers act on the exact deployment actor observation; the training-side
scheduler may use terrain-family and segment geometry as privileged truth, but
that truth never enters any actor input or the student's own policy.
"""
from __future__ import annotations

import torch
from torch import nn


FAMILY_TO_TEACHER = {
    'gaps_long': 'long',
    'tire_field': 'tire',
}
BASE_TEACHER = 'general'
COMBINED = 'combined'


class TeacherPool:
    def __init__(self, actors: dict[str, nn.Module], num_envs: int, device):
        if not actors:
            raise ValueError('teacher pool must not be empty')
        self.names = sorted(actors)
        self.actors = {name: actors[name].to(device).eval() for name in self.names}
        for actor in self.actors.values():
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
        self.hidden = {name: torch.zeros(1, num_envs, 128, device=device)
                       for name in self.names}
        self.num_envs = num_envs
        self.device = device

    def reset(self, ids: torch.Tensor) -> None:
        for state in self.hidden.values():
            state[:, ids] = 0

    def act(self, name: str, obs: torch.Tensor, ids: torch.Tensor):
        actor = self.actors[name]
        if not bool(ids.any()):
            return None
        state = self.hidden[name][:, ids]
        mean, next_state = actor(obs[ids], state)
        self.hidden[name][:, ids] = next_state
        return mean


def combined_segment_teacher(course_x: torch.Tensor, segments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher code per combined env from its four course segments.

    segments[:,0:3] are the opening gaps (start, void_end, end) and segments[:,3]
    is the tire band (start, end, end). Code 0 = general teacher (lead-in, gap
    band and everything after the tire band), 1 = tire teacher inside the band,
    -1 = undefined for NaN or malformed rows.
    """
    n = course_x.shape[0]
    valid = torch.isfinite(segments).all(dim=(1, 2))
    band = segments[:, 3]
    keys = torch.zeros(n, dtype=torch.long)
    keys[(course_x >= band[:, 0] - 0.20) & (course_x <= band[:, 1] + 0.20)] = 1
    keys[~valid] = -1
    return keys, valid


GENERAL_FAMILIES = frozenset({'gaps_short', 'slope_down', 'slope_mixed', 'slope_up',
                              'stairs_down_long', 'stairs_mixed', 'stairs_narrow_8_mixed',
                              'stairs_up_long'})


def family_teacher_keys(families: list[str], course_x: torch.Tensor,
                        segments: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    """Per-env teacher name plus valid mask; unknown or malformed -> undefined."""
    codes, seg_valid = combined_segment_teacher(course_x, segments)
    names = []
    out_valid = torch.zeros(len(families), dtype=torch.bool)
    for index, family in enumerate(families):
        name = ''
        if family == COMBINED:
            if bool(seg_valid[index]) and int(codes[index]) >= 0:
                name = 'tire' if int(codes[index]) == 1 else 'general'
        elif family in FAMILY_TO_TEACHER:
            name = FAMILY_TO_TEACHER[family]
        elif family in GENERAL_FAMILIES:
            name = 'general'
        names.append(name)
        out_valid[index] = bool(name)
    return names, out_valid
