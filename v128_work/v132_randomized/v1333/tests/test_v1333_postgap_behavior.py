import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

from go1_hier_sim.envs import gap_score_v1321, gait_gap_rewards_v1330


def test_postgap_success_uses_four_metres_and_stable_walk_not_paid_credits():
    step = getattr(gap_score_v1321, "gap_post_finish_step", None)
    assert callable(step), "post-gap success helper is not implemented"
    bounds = torch.tensor([[[1.0, 1.2, 2.0], [8.0, 8.3, 9.1],
                            [float("nan"), float("nan"), float("nan")]]])
    completed, next_steps = step(
        bounds=bounds,
        local_x=torch.tensor([12.3]),
        lateral_y=torch.tensor([0.0]),
        stable=torch.tensor([True]),
        fallen=torch.tensor([False]),
        fell_into_gap=torch.tensor([False]),
        stable_steps=torch.tensor([24]),
        update_mask=torch.tensor([True]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
        stable_steps_required=25,
    )
    assert completed.tolist() == [True]
    assert next_steps.tolist() == [25]


def test_gap_goal_distance_tracks_last_far_lip_plus_four_metres():
    distance_fn = getattr(gap_score_v1321, "gap_post_finish_distance", None)
    assert callable(distance_fn), "post-gap goal distance helper is not implemented"
    bounds = torch.tensor([[[1.0, 1.2, 2.0], [8.0, 8.3, 9.1],
                            [float("nan"), float("nan"), float("nan")]]])
    distance, has_gap = distance_fn(
        bounds,
        torch.tensor([11.3]),
        torch.tensor([0.0]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
    )
    assert has_gap.tolist() == [True]
    assert torch.allclose(distance, torch.tensor([1.0]))
    distance, _ = distance_fn(
        bounds,
        torch.tensor([12.3]),
        torch.tensor([5.0]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
    )
    assert torch.allclose(distance, torch.tensor([1.0]))
    distance, _ = distance_fn(
        bounds,
        torch.tensor([13.0]),
        torch.tensor([0.0]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
    )
    assert torch.allclose(distance, torch.tensor([0.0]))


def test_postgap_success_requires_zone_and_does_not_advance_twice_per_tick():
    step = getattr(gap_score_v1321, "gap_post_finish_step", None)
    assert callable(step), "post-gap success helper is not implemented"
    bounds = torch.tensor([[[8.0, 8.3, 9.1]]])
    common = dict(
        bounds=bounds,
        local_x=torch.tensor([12.3]),
        lateral_y=torch.tensor([0.0]),
        stable=torch.tensor([True]),
        fallen=torch.tensor([False]),
        fell_into_gap=torch.tensor([False]),
        stable_steps=torch.tensor([24]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
        stable_steps_required=25,
    )
    completed, next_steps = step(**common, update_mask=torch.tensor([False]))
    assert completed.tolist() == [False]
    assert next_steps.tolist() == [24]
    common["local_x"] = torch.tensor([12.29])
    completed, next_steps = step(**common, update_mask=torch.tensor([True]))
    assert completed.tolist() == [False]
    assert next_steps.tolist() == [25]


def test_postgap_success_rejects_gap_falls_and_unstable_or_nonwalking_states():
    step = getattr(gap_score_v1321, "gap_post_finish_step", None)
    assert callable(step), "post-gap success helper is not implemented"
    bounds = torch.tensor([[[8.0, 8.3, 9.1]]])
    base = dict(
        bounds=bounds,
        local_x=torch.tensor([12.3]),
        lateral_y=torch.tensor([0.0]),
        fallen=torch.tensor([False]),
        fell_into_gap=torch.tensor([False]),
        stable_steps=torch.tensor([25]),
        update_mask=torch.tensor([True]),
        distance_after_gap=4.0,
        lateral_half_width=4.0,
        stable_steps_required=25,
    )
    for overrides in (
        {"stable": torch.tensor([False])},
        {"stable": torch.tensor([True]), "fallen": torch.tensor([True])},
        {"stable": torch.tensor([True]), "fell_into_gap": torch.tensor([True])},
        {"stable": torch.tensor([True]), "lateral_y": torch.tensor([4.01])},
    ):
        args = {**base, "stable": torch.tensor([True]), **overrides}
        completed, _ = step(**args)
        assert completed.tolist() == [False]


def test_flight_window_only_exempts_long_gaps_not_short_or_oversized_gaps():
    window = getattr(gait_gap_rewards_v1330, "long_gap_flight_window", None)
    assert callable(window), "long-gap-only flight window is not implemented"
    bounds = torch.tensor([
        [[1.0, 1.2, 2.0]],
        [[5.0, 5.35, 6.1]],
        [[9.0, 9.60, 10.2]],
        [[12.0, 12.80, 13.3]],
    ])
    result = window(
        bounds,
        torch.tensor([1.1, 5.1, 9.2, 12.2]),
        approach_distance=0.35,
        landing_distance=0.15,
        short_width_max=0.265,
        max_gap_width=0.70,
    )
    assert result.tolist() == [False, True, True, False]


def test_long_gap_flight_window_is_spatially_limited_to_the_far_lip():
    window = getattr(gait_gap_rewards_v1330, "long_gap_flight_window", None)
    assert callable(window), "long-gap-only flight window is not implemented"
    bounds = torch.tensor([[[2.0, 2.35, 3.1]]])
    result = window(
        bounds,
        torch.tensor([1.65]),
        approach_distance=0.35,
        landing_distance=0.15,
        short_width_max=0.265,
        max_gap_width=0.70,
    )
    assert result.tolist() == [True]
    result = window(
        bounds,
        torch.tensor([2.501]),
        approach_distance=0.35,
        landing_distance=0.15,
        short_width_max=0.265,
        max_gap_width=0.70,
    )
    assert result.tolist() == [False]


def test_runtime_uses_postgap_completion_and_long_gap_only_airtime_gates():
    runtime = (ROOT / "source" / "go1_hier_sim" / "go1_hier_sim" / "envs" / "runtime.py").read_text()
    assert "gap_post_finish_step(" in runtime
    assert "long_gap_flight_window(" in runtime
    assert '"non_gap_airborne": non_gap_airborne_cost(' in runtime
    non_gap = runtime.index('"non_gap_airborne": non_gap_airborne_cost(')
    non_gap_end = runtime.index("),", non_gap)
    assert "long_gap_window" in runtime[non_gap:non_gap_end]
    flight_window = runtime.index("window_open =", runtime.index("if self._jump_active:"))
    assert "long_gap_window" in runtime[flight_window:flight_window + 180]
