"""New hierarchical locomotion + obstacle-avoidance training system (fwwb-inspired).

The package is additive: it does not modify the validated Stage-1 contract. It
provides two independent reward contracts, the fwwb-style encoder policies and
the sim-to-real robustness curriculum used by the ``nav_v2`` stage.
"""
