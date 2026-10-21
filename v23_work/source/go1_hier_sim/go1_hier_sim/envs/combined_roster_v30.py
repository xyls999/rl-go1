"""Single-family roster for the combined specialization attempt (Stage-C prerequisite)."""
from __future__ import annotations


def select_training_roster(config: dict, families: tuple[str, ...]) -> tuple[str, ...]:
    roster = config.get('training_roster')
    if roster != 'single:combined':
        raise ValueError('v30 roster must be single:combined')
    if 'combined' not in families:
        raise ValueError('combined family missing from registry')
    return ('combined',)
