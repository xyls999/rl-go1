from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))
from go1_hier_sim.envs.jump_airtime_v1325 import gap_window_airtime_reward
from go1_hier_sim.reward_config import load_stage1_rewards


BASE_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_model1298_base_a02.toml"
CURRENT_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333.toml"
NEW_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a02.toml"


def test_longgap_profile_enables_useful_flight_and_increases_forward_credit_only():
    assert NEW_CONFIG.is_file(), "missing isolated long-gap a02 reward profile"
    baseline = load_stage1_rewards(BASE_CONFIG)
    updated = load_stage1_rewards(NEW_CONFIG)
    changes = {}
    for name in baseline:
        assert set(updated[name]) == set(baseline[name])
        for field, old in baseline[name].items():
            new = updated[name][field]
            if old != new:
                changes[f"{name}.{field}"] = [old, new]
    assert changes == {
        "forward_progress.weight": [4.0, 6.0],
        "four_feet_jump_air_time.weight": [0.0, 2.0],
    }
    assert updated["four_feet_jump_air_time"]["max_air_time"] == 0.55
    for name in ("fallen", "fell_into_gap", "collision", "out_of_bounds", "gap_intrusion"):
        assert updated[name] == baseline[name]
    assert baseline["forward_progress"]["weight"] == 4.0
    assert baseline["four_feet_jump_air_time"]["weight"] == 0.0
    assert baseline["flight_clearance"]["weight"] == 0.0
    current = load_stage1_rewards(CURRENT_CONFIG)
    assert current["flight_clearance"]["weight"] == 0.15


def test_longgap_airtime_reward_needs_flight_forward_motion_window_and_safe_clearance():
    air = torch.tensor([
        [0.20, 0.21, 0.22, 0.23],
        [0.20, 0.21, 0.22, 0.00],
        [0.20, 0.21, 0.22, 0.23],
        [0.20, 0.21, 0.22, 0.23],
        [0.20, 0.21, 0.22, 0.23],
    ])
    reward = gap_window_airtime_reward(
        current_air_time=air,
        forward_progress=torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]),
        window_open=torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0]),
        gap_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        intrusion=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        max_air_time=0.55,
    )
    assert reward[0].item() == pytest.approx(0.20)
    assert reward[1:].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
