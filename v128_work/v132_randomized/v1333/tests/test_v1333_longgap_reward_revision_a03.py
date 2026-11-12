from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))
from go1_hier_sim.reward_config import load_stage1_rewards

A02_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a02.toml"
A03_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a03.toml"


def test_a03_increases_only_approved_forward_and_airtime_weights():
    assert A03_CONFIG.is_file(), "missing isolated long-gap a03 reward profile"
    previous = load_stage1_rewards(A02_CONFIG)
    updated = load_stage1_rewards(A03_CONFIG)

    changes = {}
    assert set(updated) == set(previous)
    for term in previous:
        assert set(updated[term]) == set(previous[term])
        for field, old_value in previous[term].items():
            new_value = updated[term][field]
            if old_value != new_value:
                changes[f"{term}.{field}"] = [old_value, new_value]

    assert changes == {
        "forward_progress.weight": [6.0, 8.0],
        "four_feet_jump_air_time.weight": [2.0, 4.0],
    }
    assert updated["forward_progress"]["max_progress_speed"] == previous["forward_progress"]["max_progress_speed"] == 0.9
    assert updated["four_feet_jump_air_time"]["max_air_time"] == previous["four_feet_jump_air_time"]["max_air_time"] == 0.55


def test_a03_preserves_gap_and_fall_safety_penalties():
    previous = load_stage1_rewards(A02_CONFIG)
    updated = load_stage1_rewards(A03_CONFIG)

    for term in ("fallen", "fell_into_gap", "collision", "out_of_bounds", "gap_intrusion"):
        assert updated[term] == previous[term]
