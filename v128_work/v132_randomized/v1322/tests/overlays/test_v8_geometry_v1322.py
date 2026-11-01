"""V8 geometry checks with explicit legacy and production threshold environments."""
import numpy as np
import pytest

from go1_hier_sim.envs.geometry import FEATURE_STOP, build_tile, build_scene


def profile(seed, difficulty):
    mesh = build_tile("combined", difficulty, np.random.default_rng(seed), False)
    return mesh.metadata["course_profile"]


@pytest.mark.parametrize("seed", [2026, 2027, 2030])
@pytest.mark.parametrize("difficulty", [0.0, 0.1, 0.6, 1.0])
def test_mid_course_tyre_bands_have_more_than_two_forward_rows(seed, difficulty):
    blocks = profile(seed, difficulty)["blocks"]
    tyre = [block for block in blocks if block["kind"] == "tire_field"]
    assert len(tyre) == 3
    first_minimum = 16 if difficulty <= 0.3 else 12
    assert [block["tyres"] for block in tyre[:2]] >= [first_minimum] * 2
    assert tyre[-1]["tyres"] >= 12
    assert tyre[-1]["end"] <= FEATURE_STOP + 1e-8
    assert all(left["end"] <= right["start"] + 1e-7
               for left, right in zip(blocks, blocks[1:]))
    gaps = [block for block in blocks if block["kind"] == "gap"]
    assert len(gaps) == 6
    assert all(gap["width"] <= 0.24 for gap in gaps[:3])
    assert all(0.29 <= gap["width"] <= 0.42 for gap in gaps[3:])
    for gap in gaps[:3]:
        assert gap["end"] - gap["void_end"] >= 0.65 - 1e-7
    for gap in gaps[3:]:
        assert gap["end"] - gap["void_end"] >= 1.10 - 1e-7


@pytest.mark.parametrize("difficulty", [0.0, 0.1, 0.6, 1.0])
def test_final_stair_has_one_half_height_riser_with_safe_tread(difficulty):
    blocks = profile(2026, difficulty)["blocks"]
    final = next(block for block in blocks if block["kind"] == "final_stairs")
    rises = final["riser_heights"]
    treads = final["tread_depths"]
    assert len(rises) == final["steps"] == 8
    high = [i for i, rise in enumerate(rises) if rise >= 0.16]
    assert high, rises
    assert all(treads[index] >= 0.45 for index in high)
    assert all(treads[2 * final["steps"] - 1 - index] >= 0.45 for index in high)
    assert final["end"] <= FEATURE_STOP


def test_reward_only_long_gap_boundaries_are_not_actor_observations(monkeypatch):
    monkeypatch.setenv("GO1_V8_LONG_GAP_THRESHOLD", "0.26")
    mesh, origins = build_scene(
        4, "lp_acrl", False, 2026, 0.1,
        tasks=[("combined", 0.1), ("gaps_long", 0.1),
               ("gaps_short", 0.1), ("tire_field", 0.1)],
    )
    bounds = mesh.metadata["long_gap_segments"]
    assert bounds.shape == (4, 24, 3)
    assert np.isfinite(bounds[0]).all(axis=1).sum() == 3
    assert np.isfinite(bounds[1]).all(axis=1).sum() > 3
    assert np.isnan(bounds[2:]).all()
    assert origins.shape == (4, 3)


def test_production_threshold_keeps_all_six_combined_gap_boundaries_reward_only(monkeypatch):
    monkeypatch.setenv("GO1_V8_LONG_GAP_THRESHOLD", "0.05")
    mesh, origins = build_scene(
        4, "lp_acrl", False, 2026, 0.1,
        tasks=[("combined", 0.1), ("gaps_long", 0.1),
               ("gaps_short", 0.1), ("tire_field", 0.1)],
    )
    bounds = mesh.metadata["long_gap_segments"]
    assert bounds.shape == (4, 24, 3)
    assert np.isfinite(bounds[0]).all(axis=1).sum() == 6
    assert np.isfinite(bounds[1]).all(axis=1).sum() > 3
    short_gaps = mesh.metadata["terrain_randomization"][2]["gaps"]
    assert short_gaps and all(gap["width"] > 0.05 for gap in short_gaps)
    assert np.isfinite(bounds[2, :len(short_gaps)]).all()
    assert np.isnan(bounds[2, len(short_gaps):]).all()
    assert np.isnan(bounds[3]).all()
    assert origins.shape == (4, 3)


def test_high_stair_is_not_added_to_short_walking_trenches():
    blocks = profile(2026, 0.1)["blocks"]
    assert all(block["kind"] == "gap" for block in blocks[:3])
    assert all(block["kind"] != "final_stairs" for block in blocks[:3])
