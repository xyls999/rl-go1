"""V132-aware regression overlay for per-tile randomized gap counts."""
import numpy as np

from go1_hier_sim.envs.geometry import START_IN_TILE, build_scene


def test_all_gap_bounds_are_per_tile_and_never_cover_tires():
    mesh, _ = build_scene(
        4, "lp_acrl", False, 2026, .1,
        tasks=[("combined", .1), ("gaps_short", .1), ("gaps_long", .1), ("tire_field", .1)],
    )
    all_gaps = mesh.metadata["all_gap_segments"]
    profiles = mesh.metadata["terrain_randomization"]
    assert all_gaps.shape == (4, 24, 3)
    assert len(profiles) == 4
    for index in (0, 1, 2):
        expected = profiles[index]["gaps"]
        actual = all_gaps[index]
        assert np.isfinite(actual).all(axis=1).sum() == len(expected)
        valid = actual[np.isfinite(actual).all(axis=1)]
        expected_bounds = np.asarray([
            [gap[key] - START_IN_TILE[0] for key in ("start", "void_end", "end")]
            for gap in expected
        ], dtype=np.float32)
        np.testing.assert_allclose(valid, expected_bounds, atol=1e-6)
        assert np.all(valid[:, 0] < valid[:, 1])
        assert np.all(valid[:, 1] <= valid[:, 2])
    assert np.isnan(all_gaps[3]).all()
    first_tire = mesh.metadata["mixed_reward_segments"][0, 3, 0]
    assert all_gaps[0, 2, 2] <= first_tire + 1e-6
