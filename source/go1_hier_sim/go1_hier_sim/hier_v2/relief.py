"""Terrain relief map for the schema-5 speed-layer observation.

Kept free of Isaac imports so the masking rules stay unit-testable without the
simulator. The ray caster reports a probe that missed the mesh as a non-finite
hit; subtracting the reference row from an unmasked non-finite height produced
``inf - inf = NaN`` and turned the whole map into NaN, which killed a training
run at random once a robot left the mesh.
"""

from __future__ import annotations

import torch

from ..contracts import RELIEF_COLS, RELIEF_FORWARD_M, RELIEF_ROWS


def relief_map(hit_z: torch.Tensor, face_hit: torch.Tensor):
    """Return ``(relief, confidence)`` for a ``[N, rows, cols]`` ground probe.

    ``hit_z`` holds the world height of each downward probe and ``face_hit``
    marks which probes actually hit the mesh. A probe that missed is unknown
    space: relief 0 with confidence 0, never free space. The gradient is measured
    against the probe directly ahead of the body (falling back to the mean hit
    height) and clamped to +/-1, so a 0.16 m step reads 0.27 from 0.6 m away and
    0.13 from 1.2 m away.
    """
    if hit_z.ndim != 3 or hit_z.shape[1:] != (RELIEF_ROWS, RELIEF_COLS):
        raise ValueError(
            f"relief probe must be [N, {RELIEF_ROWS}, {RELIEF_COLS}], got {tuple(hit_z.shape)}"
        )
    if face_hit.shape != hit_z.shape:
        raise ValueError("relief hits and face ids must have the same shape")

    distances = torch.tensor(RELIEF_FORWARD_M, device=hit_z.device, dtype=hit_z.dtype)
    finite = torch.isfinite(hit_z)
    valid = face_hit.bool() & finite
    heights = torch.where(finite, hit_z, torch.zeros_like(hit_z))
    valid_float = valid.to(hit_z.dtype)

    centre = heights[:, 0, RELIEF_COLS // 2]
    mean_hit = (heights * valid_float).sum(dim=(1, 2)) / valid_float.sum(dim=(1, 2)).clamp_min(1.0)
    reference = torch.where(valid[:, 0, RELIEF_COLS // 2], centre, mean_hit)
    gradient = (
        (heights - reference[:, None, None]) / distances.clamp_min(1.0e-3)[None, :, None]
    ).clamp(-1.0, 1.0)
    gradient = torch.where(valid, gradient, torch.zeros_like(gradient))
    return gradient.reshape(hit_z.shape[0], -1), valid_float.reshape(hit_z.shape[0], -1)
