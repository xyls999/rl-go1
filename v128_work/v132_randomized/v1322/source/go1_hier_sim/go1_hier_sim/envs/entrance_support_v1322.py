"""Reward-only supported approach to real entrance gaps; never actor input.

A per-gap forward watermark prevents re-paying previously traversed near-pad
positions. This is not clearance state: only gap_clearance_step owns paid credit.
"""
import torch


def entrance_support_step(bounds, local_x, foot_x, foot_z, contacts, root_z,
                          upright, forward_velocity, command_speed, failed, paid,
                          entrance_mask, watermark):
    """Return bounded new supported progress and the updated approach watermark.

    The caller resets only the selected environment rows of watermark to -inf on
    episode reset. A watermark advances even without support, preventing a
    backward detour from later claiming unsupported progress as safe approach.
    """
    if bounds.ndim != 3 or bounds.shape[-1] != 3:
        raise ValueError('entrance gap bounds must have shape [N,K,3]')
    n, k, _ = bounds.shape
    if (paid.shape != (n, k) or watermark.shape != (n, k) or
            foot_x.shape != (n, 4) or foot_z.shape != (n, 4) or
            contacts.shape != (n, 4) or
            any(value.shape != (n,) for value in
                (local_x, root_z, upright, forward_velocity, command_speed,
                 failed, entrance_mask))):
        raise ValueError('entrance support input shapes do not match [N,K] and [N,4]')

    start, far_lip, end = bounds.unbind(-1)
    width = far_lip - start
    valid = (torch.isfinite(bounds).all(-1) & torch.isfinite(local_x[:, None]) &
             (width >= .08 - 1e-5) & (width <= .24 + 1e-5) & (end > far_lip) &
             ~paid & entrance_mask[:, None])
    x = local_x[:, None]
    approach_start = start - .45
    window = (x >= approach_start) & (x <= end + .20)
    # Positions on the void lip or below the pad surface are not support.
    pad_contacts = contacts[:, None, :] & (foot_z[:, None, :] >= -.02) & (
        (foot_x[:, None, :] <= start[:, :, None] - .02) |
        (foot_x[:, None, :] >= far_lip[:, :, None] + .02))
    safe = (valid & window & (pad_contacts.sum(-1) >= 2) & ~failed[:, None] &
            (root_z[:, None] >= .23) & (upright[:, None] >= .85) &
            (forward_velocity[:, None] > .1) & (command_speed[:, None] > .1))
    previous = torch.maximum(watermark, approach_start)
    new_progress = (x - previous).clamp(min=0.)
    height_quality = ((root_z - .23) / .09).clamp(0., 1.)
    upright_quality = ((upright - .85) / .10).clamp(0., 1.)
    motion_quality = (((forward_velocity - .1) / .2).clamp(0., 1.) *
                      ((command_speed - .1) / .25).clamp(0., 1.))
    quality = height_quality * upright_quality * motion_quality
    per_gap = torch.where(safe, (new_progress / .45).clamp(0., 1.) * quality[:, None],
                          torch.zeros_like(new_progress))
    next_watermark = torch.where(valid, torch.maximum(watermark, x), watermark)
    return per_gap.amax(-1), next_watermark
