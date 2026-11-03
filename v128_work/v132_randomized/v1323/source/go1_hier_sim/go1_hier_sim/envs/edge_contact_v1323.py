"""Reward-only contacts close to real entrance-gap pad edges."""
import torch


def edge_contact_step(bounds, foot_x, foot_z, contacts, failed, entrance_mask):
    """Count contacting feet within 5 cm on solid near/far pads, at most four."""
    if bounds.ndim != 3 or bounds.shape[-1] != 3:
        raise ValueError('entrance gap bounds must have shape [N,K,3]')
    n, k, _ = bounds.shape
    if (foot_x.shape != (n, 4) or foot_z.shape != (n, 4) or
            contacts.shape != (n, 4) or failed.shape != (n,) or
            entrance_mask.shape != (n,)):
        raise ValueError('edge contact input shapes must match [N,K,3] and [N,4]')

    start, far_lip, end = bounds.unbind(-1)
    # Geometry includes a fixed 0.05 m lip beyond the sampled short opening.
    physical_width = far_lip - start - .05
    valid = (torch.isfinite(bounds).all(dim=-1) &
             (physical_width >= .08 - 1e-5) & (physical_width <= .24 + 1e-5) &
             (end > far_lip))
    foot = foot_x[:, None, :]
    tolerance = 1e-6
    near_pad = ((foot >= start[:, :, None] - .05 - tolerance) &
                (foot <= start[:, :, None]))
    far_pad = ((foot >= far_lip[:, :, None]) &
               (foot <= far_lip[:, :, None] + .05 + tolerance))
    on_edge = ((near_pad | far_pad) & valid[:, :, None]).any(dim=1)
    supported = contacts & (foot_z >= -.02) & on_edge
    return (supported & entrance_mask[:, None] & ~failed[:, None]).sum(dim=-1).to(foot_x.dtype)
