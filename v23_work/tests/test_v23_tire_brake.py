"""Detect: no brake bonus when command is already low, bounded overspeed charge, family gating."""
import pytest
import torch
try:
    from go1_hier_sim.envs.tire_brake_v23 import tire_brake_adjustment
except ImportError:  # RED until the module is deployed
    tire_brake_adjustment = None

TIRE = torch.tensor([True, False, True])


def adj(v, cmd):
    assert tire_brake_adjustment is not None, "tire brake not implemented"
    return tire_brake_adjustment(
        forward_velocity=torch.tensor(v), command_speed=torch.tensor(cmd),
        tire_mask=TIRE, std=.45)


def test_low_command_tracking_is_untouched_and_high_command_braking_is_paid():
    assert adj([.5, .9, .2], [.5, .5, .1]).tolist() == pytest.approx([0., 0., 0.], abs=1e-6)
    assert adj([.58, .9, .9], [.8, .5, .1])[0].item() == pytest.approx(0.2106, abs=1e-3)


def test_overspeed_above_cap_plus_margin_is_charged_and_clamped():
    got = adj([.8, .9, .9], [.8, .8, .4])
    assert got[0].item() == pytest.approx(-0.3792, abs=1e-3)
    assert got[1].item() == 0.  # non-tire slot: shaping must never touch it
    assert got[2].item() == pytest.approx(-0.3667, abs=1e-3)
    assert adj([3.], [3.])[0].item() == pytest.approx(-1.5, abs=1e-3)


def test_brake_signal_prefers_slower_than_dash_on_the_field():
    assert adj([.55], [.8])[0] > adj([.70], [.8])[0] > adj([.85], [.8])[0]


def test_non_tire_slots_never_receive_the_adjustment():
    assert adj([.9, .9, .9], [.9, .9, .9])[1].item() == 0.
    assert adj([.58, .58, .58], [.8, .8, .8])[1].item() == 0.


@pytest.mark.parametrize("kwargs", [
    {"std": 0.}, {"std": -1.}, {"span": 0.},
])
def test_malformed_shape_constants_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        tire_brake_adjustment(forward_velocity=torch.tensor([.5]),
                              command_speed=torch.tensor([.5]),
                              tire_mask=torch.tensor([True]), **{"std": .45, **kwargs})


def test_parameters_shift_charge_and_target_without_breaking_gating():
    # cap=.65 moves the charge onset to .73; hand-checked literals below.
    a = tire_brake_adjustment(forward_velocity=torch.tensor([.68, .9, .9]),
                              command_speed=torch.tensor([.77, .77, .4]),
                              tire_mask=torch.tensor([True, True, False]), std=.45,
                              cap=.65, penalty_scale=.35)
    assert a[0].item() == pytest.approx(0.0348, abs=2e-3)   # e^-.00444 - e^-.04, no charge
    assert a[1].item() == pytest.approx(-0.3838, abs=2e-3)  # e^-.30864 - e^-.08346 - .19833
    assert a[2].item() == 0.
    b = tire_brake_adjustment(forward_velocity=torch.tensor([.72]),
                              command_speed=torch.tensor([.77]),
                              tire_mask=torch.tensor([True]), std=.45,
                              cap=.65, penalty_scale=.35)
    assert b[0].item() == pytest.approx(-0.0116, abs=2e-3)  # e^-.024198 - e^-.012346, no charge


def test_parse_brake_families_splits_validates_and_keeps_default():
    from go1_hier_sim.envs.tire_brake_v23 import parse_brake_families
    assert parse_brake_families(None) == frozenset({'tire_field'})
    assert parse_brake_families('') == frozenset({'tire_field'})
    assert parse_brake_families('tire_field') == frozenset({'tire_field'})
    assert parse_brake_families('tire_field, combined') == frozenset({'tire_field', 'combined'})
    for bad in (',', 'tire_field,', ',combined', '  '):
        try:
            parse_brake_families(bad)
            raise AssertionError('accepted malformed list: ' + repr(bad))
        except ValueError:
            pass
