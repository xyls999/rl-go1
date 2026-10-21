"""Single-family roster for the v32/v33 soft-curriculum combined ladder."""
from __future__ import annotations


def select_training_roster(config: dict, families: tuple[str, ...]) -> tuple[str, ...]:
    roster = str(config.get('training_roster') or '')
    if roster == 'ladder:combined':
        # Composition ladder: the mixed route with 0/1/2/3 voids plus the real
        # combined tile. Training sees positive experience at the easy rungs
        # while the acceptance family stays in the mixture.
        ladder = ('course', 'combined_soft13', 'combined_soft14',
                  'combined_soft15', 'combined')
        missing = [f for f in ladder if f not in families]
        if missing:
            raise ValueError(f'ladder families missing from registry: {missing}')
        return ladder
    if not roster.startswith('single:'):
        raise ValueError('roster must be single:<family> or ladder:combined')
    family = roster.split(':', 1)[1]
    if family not in families:
        raise ValueError(f'{family} missing from registry')
    return (family,)
