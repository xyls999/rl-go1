"""Multi-teacher scheduler: family routing, combined segment ownership, hidden reset."""
import pytest
import torch
try:
    from go1_hier_sim.envs.teacher_distill_v29 import (
        TeacherPool, combined_segment_teacher, family_teacher_keys)
    from go1_hier_sim.models import ActorCore
except ImportError:  # RED until the module ships
    TeacherPool = None


def test_family_routing_covers_ten_families_and_rejects_unknown():
    x = torch.zeros(11)
    segs = torch.full((11, 4, 3), float('nan'))
    fams = ['gaps_long', 'gaps_short', 'tire_field', 'slope_down', 'slope_mixed',
            'slope_up', 'stairs_down_long', 'stairs_mixed', 'stairs_narrow_8_mixed',
            'stairs_up_long', 'not_a_family']
    names, valid = family_teacher_keys(fams, x, segs)
    assert names[:10] == ['long', 'general', 'tire', 'general', 'general', 'general',
                          'general', 'general', 'general', 'general']
    assert valid[:10].all() and not valid[10] and names[10] == ''


def test_combined_segments_route_tire_band_only_to_the_tire_teacher():
    segs = torch.tensor([[[1.0, 1.2, 1.4], [2.0, 2.2, 2.4], [3.0, 3.2, 3.4],
                          [4.0, 5.0, 5.0]]])
    codes, valid = combined_segment_teacher(torch.tensor([0.5, 1.5, 3.99, 4.5, 5.21, 9.0]), segs.repeat(6, 1, 1))
    assert codes.tolist() == [0, 0, 1, 1, 0, 0] and valid.all()
    bad = torch.full((1, 4, 3), float('nan'))
    codes, valid = combined_segment_teacher(torch.tensor([2.0]), bad)
    assert not bool(valid[0]) and int(codes[0]) == -1


def test_combined_family_uses_segment_schedule_and_malformed_rows_are_undefined():
    good = torch.tensor([[1.0, 1.2, 1.4], [2.0, 2.2, 2.4], [3.0, 3.2, 3.4], [4.0, 5.0, 5.0]])
    nan = torch.full((4, 3), float('nan'))
    names, valid = family_teacher_keys(['combined', 'combined'], torch.tensor([4.5, 2.0]),
                                       torch.stack([good, nan]))
    assert names == ['tire', ''] and valid.tolist() == [True, False]


@pytest.mark.parametrize('bad', [{}, {'general': None}])
def test_empty_pool_rejects(bad):
    if bad == {}:
        with pytest.raises(ValueError, match='must not be empty'):
            TeacherPool({}, 1, 'cpu')
    else:
        with pytest.raises((ValueError, AttributeError, TypeError)):
            TeacherPool({'general': None}, 1, 'cpu')


def test_pool_act_is_deterministic_after_reset_and_freeze():
    assert TeacherPool is not None
    torch.manual_seed(3)
    actors = {'general': ActorCore('locomotion'), 'tire': ActorCore('locomotion')}
    pool = TeacherPool(actors, num_envs=3, device='cpu')
    obs = torch.randn(3, 1073)
    ids = torch.tensor([True, False, True])
    mean_a = pool.act('general', obs, ids).clone()
    assert mean_a.shape == (2, 12) and not mean_a.requires_grad
    mean_b = pool.act('general', obs, ids)
    assert not torch.allclose(mean_a, mean_b)  # hidden advanced
    pool.reset(torch.tensor([0]))
    mean_c = pool.act('general', obs, ids)
    assert torch.allclose(mean_c[0], mean_a[0])            # env0 restarted from zero hidden
    assert not torch.allclose(mean_c[1], mean_b[1])        # env2 kept advancing
    pool.reset(torch.tensor([0, 1, 2]))
    state = pool.hidden['general']
    assert state.abs().sum().item() == 0.0 and state.shape == (1, 3, 128)
    with torch.no_grad():
        mean_d = pool.act('general', obs, ids)
    assert torch.allclose(mean_d, mean_a)


def test_recurrent_distill_epoch_matches_hand_computed_masked_mse():
    from go1_hier_sim.training.on_policy import recurrent_distill_epoch
    from go1_hier_sim.models import ActorCore
    torch.manual_seed(1)
    actor = ActorCore('locomotion')
    T, N = 3, 2
    obs = torch.randn(T, N, 1073)
    teacher = torch.randn(T, N, 12)
    valid = torch.tensor([[True, False], [True, True], [False, False]])
    dones = torch.zeros(T, N, dtype=torch.bool)
    loss, count = recurrent_distill_epoch(actor, obs, dones, teacher, valid, lam=2.0)
    assert count == 3
    hidden = torch.zeros(1, N, 128)
    total = 0.
    for t in range(T):
        mean, hidden = actor(obs[t], hidden)
        rows = [i for i in range(N) if bool(valid[t, i])]
        for i in rows:
            total = total + (mean[i] - teacher[t, i]).square().sum().item()
    assert loss.item() == pytest.approx(2.0 * total / 12 / 3, rel=1e-5)
    assert loss.requires_grad


def test_distill_epoch_zeroes_hidden_after_done():
    from go1_hier_sim.training.on_policy import recurrent_distill_epoch
    from go1_hier_sim.models import ActorCore
    actor = ActorCore('locomotion')
    torch.manual_seed(2)
    obs = torch.randn(2, 1, 1073)
    teacher = torch.zeros(2, 1, 12)
    valid = torch.ones(2, 1, dtype=torch.bool)
    done_mid = torch.tensor([[True], [False]])
    a, _ = recurrent_distill_epoch(actor, obs, done_mid, teacher, valid, lam=1.)
    no_done, _ = recurrent_distill_epoch(actor, obs, torch.zeros(2, 1, dtype=torch.bool),
                                         teacher, valid, lam=1.)
    assert a.abs().item() != no_done.abs().item()
