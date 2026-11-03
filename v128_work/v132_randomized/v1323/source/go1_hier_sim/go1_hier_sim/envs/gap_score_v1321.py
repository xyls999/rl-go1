"""Simulator-only, per-gap credit for a safe short stride or long airborne crossing.

The terrain bounds never enter actor observations. Hard failures remain the
runtime's existing contact/fall/void terminations; this is only reward credit.
"""
import torch


def gap_route_complete(bounds, paid, failed, specialist_mask):
    """Only real all-gap clearance qualifies a specialist x-line crossing.

    Other terrain families retain their existing finish criterion unchanged.
    """
    if bounds.ndim != 3 or bounds.shape[-1] != 3 or paid.shape != bounds.shape[:2]:
        raise ValueError('gap gate shapes must be [N,K,3] and [N,K]')
    n = bounds.shape[0]
    if failed.shape != (n,) or specialist_mask.shape != (n,):
        raise ValueError('gap gate masks must have shape [N]')
    valid = torch.isfinite(bounds).all(dim=-1)
    cleared = valid.any(dim=-1) & (~valid | paid).all(dim=-1) & ~failed
    return ~specialist_mask | cleared


def tire_coverage_step(bounds, local_x, local_y, contact_count, failed, visited):
    """One progress credit per eighth of the *generated* band, inside its footprint.

    This prevents a tire teacher from collecting a flat side-route finish.
    Feet on ground near the tire array count; ring contact is not mandatory.
    """
    n = bounds.shape[0]
    if bounds.shape != (n, 4) or visited.shape != (n, 8):
        raise ValueError('tire bounds/coverage must be [N,4]/[N,8]')
    if any(v.shape != (n,) for v in (local_x, local_y, contact_count, failed)):
        raise ValueError('tire scoring inputs must have shape [N]')
    start, end, centre, half_width = bounds.unbind(dim=-1)
    valid = torch.isfinite(bounds).all(dim=-1) & (end > start + 25.0) & (half_width > .4)
    inside = (valid & (local_x >= start) & (local_x < end) &
              ((local_y - centre).abs() <= half_width + .35) &
              (contact_count >= 2) & ~failed)
    span = torch.where(valid, end-start, torch.ones_like(start))
    fraction = torch.where(valid, (local_x-start)/span, torch.zeros_like(local_x))
    index = (fraction * 8).floor().to(torch.long).clamp(0, 7)
    hits = torch.nn.functional.one_hot(index, 8).bool() & inside[:, None]
    newly = hits & ~visited
    return newly.any(dim=-1).to(local_x.dtype), visited | hits


def tire_route_complete(visited, tire_mask, failed):
    if visited.ndim != 2 or visited.shape[1] != 8:
        raise ValueError('tire coverage must have shape [N,8]')
    if tire_mask.shape != visited.shape[:1] or failed.shape != visited.shape[:1]:
        raise ValueError('tire gate masks must have shape [N]')
    return ~tire_mask | (visited.all(dim=-1) & ~failed)


def gap_clearance_step(bounds, local_x, foot_x, foot_z, contacts, airborne,
                       upright, root_z, failed, active, paid):
    """Credit once on the far pad, never for jumping in place or entering a void.

    Short gaps can be stepped over while grounded; long gaps still require
    uninterrupted four-foot flight. All four feet must pass the far lip before
    credit and at least two must support a stable, upright touchdown.
    """
    if bounds.ndim != 3 or bounds.shape[-1] != 3:
        raise ValueError('gap bounds must have shape [N,K,3]')
    n, k, _ = bounds.shape
    if active.shape != (n, k) or paid.shape != (n, k):
        raise ValueError('gap reward state must have shape [N,K]')
    if foot_x.shape != (n, 4) or foot_z.shape != (n, 4) or contacts.shape != (n, 4):
        raise ValueError('foot positions/contacts must have shape [N,4]')
    if any(v.shape != (n,) for v in (local_x, airborne, upright, root_z, failed)):
        raise ValueError('gap reward inputs must have shape [N]')
    start, far_lip, end = bounds.unbind(dim=-1)
    width = far_lip - start
    # Late-course lip subtraction rounds an exactly .08 m gap below .08 in
    # float32. Tolerance is numerical only; missing trenches still stay invalid.
    valid = (torch.isfinite(bounds).all(dim=-1) &
             (width >= .08 - 1e-5) & (width <= .42 + 1e-5))
    # The sampled short/long bands have a gap (.24, .29); use its midpoint
    # so float32 rounding cannot misclassify a nominal .29 m long gap.
    short = valid & (width < .265)
    long = valid & ~short
    x = local_x[:, None]
    support_bad = ((foot_z < -.02) & contacts).any(dim=-1)
    ok = ~(failed | support_bad)
    short_arrival = short & (x >= start - .20) & (x <= start + .10)
    long_takeoff = long & airborne[:, None] & (x >= start - .15) & (x <= start + .10)
    pending = (active | short_arrival | long_takeoff) & valid & ~paid & ok[:, None]
    # A long attempt is invalidated by contact before the far-side landing.
    pending = pending & (~long | airborne[:, None] | (x >= far_lip + .10))
    stable = ((contacts.sum(dim=-1) >= 2) & (upright >= .85) &
              (root_z >= .23) & ok)[:, None]
    beyond = (x >= far_lip + .10) & (x <= end)
    short_clear = short & beyond & (foot_x.amin(dim=-1)[:, None] > far_lip + .02)
    long_clear = long & beyond & ~airborne[:, None]
    landed = pending & stable & (short_clear | long_clear)
    next_paid = paid | landed
    next_active = pending & ~landed & (x <= end)
    return landed.any(dim=-1).to(root_z.dtype), next_active, next_paid
