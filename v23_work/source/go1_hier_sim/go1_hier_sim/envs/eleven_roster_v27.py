"""Roster with combined included: the final eleven-family mixture stage."""
from __future__ import annotations


def select_training_roster(config: dict, families: tuple[str, ...]) -> tuple[str, ...]:
    roster = config.get('training_roster')
    if roster != 'eleven_with_combined':
        raise ValueError('v27 roster must be eleven_with_combined')
    if 'combined' not in families or len(families) != 11:
        raise ValueError('v27 requires the full eleven-family registry')
    return tuple(families)
