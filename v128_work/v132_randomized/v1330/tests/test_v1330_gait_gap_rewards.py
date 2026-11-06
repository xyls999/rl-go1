from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "source" / "go1_hier_sim"
sys.path.insert(0, str(SOURCE))

from go1_hier_sim.envs.gait_gap_rewards_v1330 import (  # noqa: E402
    fore_aft_swing_cost,
    gap_windows,
    non_gap_airborne_cost,
    short_gap_clearance_target,
    short_gap_lift_reward,
)


def test_v1330_reward_contract_loads_with_bounded_weights_and_no_execution_authority():
    from go1_hier_sim.reward_config import load_stage1_rewards

    config_path = ROOT / "configs" / "rewards_stage1_gaps_v1330_gait_gap.toml"
    rewards = load_stage1_rewards(config_path)
    manifest = json.loads((ROOT / "configs" / "manifest.json").read_text())

    assert rewards["fallen"]["weight"] == -16.0
    assert rewards["gap_intrusion"]["weight"] == -8.0
    assert rewards["gap_intrusion"]["entry_multiplier"] == 1.5
    assert rewards["stall_grow"]["weight"] == -1.0
    assert rewards["backward_motion"]["weight"] == -1.0
    assert rewards["fore_aft_swing_l2"]["weight"] == -0.25
    assert rewards["fore_aft_swing_l2"]["max_excursion"] == 0.32
    assert rewards["non_gap_airborne"]["weight"] == -0.5
    assert rewards["non_gap_airborne"]["grace_seconds"] == 0.06
    assert rewards["short_gap_lift"]["weight"] == 0.5
    assert rewards["short_gap_lift"]["target_height"] == 0.22
    assert manifest["execution"]["train_environment"]["GO1_V121_STALL_GROW"] == "1"
    assert manifest["execution"]["train_environment"]["GO1_V121_BACKWARD"] == "1"
    assert manifest["status"] in {"authorized-one-shot", "trained-unverified", "training-failed"}
    if manifest["status"] == "authorized-one-shot":
        assert manifest["run_authorized"] is True
        assert manifest["training_authorized"] is True
    else:
        assert manifest["run_authorized"] is False
        assert manifest["training_authorized"] is False
    assert manifest["diagnostic_authorized"] is False
    assert manifest["automatic_gate_authorized"] is False
    assert manifest["automatic_retry_authorized"] is False
    assert manifest["automatic_follow_on_authorized"] is False
    assert manifest["execution"]["diagnostic_authorized"] is False
    assert manifest["execution"]["automatic_gate_authorized"] is False
    assert manifest["execution"]["automatic_retry_authorized"] is False
    assert manifest["execution"]["automatic_follow_on_authorized"] is False
    assert manifest["candidate"] == "v1330-gait-gap-aware-walk"
    assert manifest["candidate_reward_config"] == "configs/rewards_stage1_gaps_v1330_gait_gap.toml"
    assert manifest["observation_profile"]["sha256"]
    source_manifest = json.loads((ROOT.parent / "v1329" / "configs" / "manifest.json").read_text())
    assert (ROOT / "configs" / "observation_profile.json").read_bytes() == (
        ROOT.parent / "v1329" / "configs" / "observation_profile.json"
    ).read_bytes()
    assert (ROOT / "configs" / "task_plan.json").read_bytes() == (
        ROOT.parent / "v1329" / "configs" / "task_plan.json"
    ).read_bytes()
    assert source_manifest["candidate"] == "v1329-speed-feedback-restoration"
    execution = json.loads((ROOT / "configs" / "train_execution.json").read_text())
    assert execution["training_authorized"] is (manifest["status"] == "authorized-one-shot")
    assert execution["diagnostic_authorized"] is False
    assert execution["gate_authorized"] is False
    assert execution["holdout_authorized"] is False
    assert execution["retry_authorized"] is False
    assert execution["automatic_follow_on_authorized"] is False
    assert execution["run_id"] == "go1-v1330-gait-gap-aware-s01-p1-d1"
    assert execution["checkpoint_path"].endswith("/model_649.pt")
    assert execution["reward_config_sha256"] == manifest["candidate_reward_config_sha256"]
    assert not (ROOT / "configs" / "rewards_stage1_gaps_v1329_speed_feedback.toml").exists()
    assert manifest["candidate_reward_config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    for relative, expected in manifest["source_files_sha256"].items():
        path = ROOT / "source" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_v1330_runtime_wires_gap_gated_gait_terms():
    runtime = (SOURCE / "go1_hier_sim" / "envs" / "runtime.py").read_text()

    assert "from .gait_gap_rewards_v1330 import" in runtime
    assert '"fore_aft_swing_l2": fore_aft_swing_cost(' in runtime
    assert '"non_gap_airborne": non_gap_airborne_cost(' in runtime
    assert '"short_gap_lift": short_gap_lift' in runtime
    assert "short_gap_clearance_target(" in runtime


def test_v1330_rewards_are_registered_and_weighted_by_stage_config():
    config = (SOURCE / "go1_hier_sim" / "envs" / "config.py").read_text()

    for name in ("fore_aft_swing_l2", "non_gap_airborne", "short_gap_lift"):
        assert f'{name} = RewardTermCfg(func=locomotion_reward' in config
        assert f'"{name}"' in config


def test_gap_windows_distinguish_short_and_long_gaps_and_ignore_missing_bounds():
    bounds = torch.tensor([
        [[1.0, 1.20, 1.40]],
        [[1.0, 1.35, 1.55]],
        [[float("nan"), float("nan"), float("nan")]],
    ])
    local_x = torch.tensor([0.70, 1.10, 1.10])

    near, near_short = gap_windows(
        bounds, local_x, approach_distance=0.35, landing_distance=0.15,
        short_width_max=0.265,
    )

    assert near.tolist() == [True, True, False]
    assert near_short.tolist() == [True, False, False]


def test_fore_aft_swing_cost_only_charges_excess_swing_feet_away_from_gap():
    offsets = torch.tensor([[0.50, 0.10, -0.60, 0.40]])
    contacts = torch.tensor([[False, False, False, True]])

    cost = fore_aft_swing_cost(
        offsets, contacts, torch.tensor([False]), max_excursion=0.30,
    )
    near_gap_cost = fore_aft_swing_cost(
        offsets, contacts, torch.tensor([True]), max_excursion=0.30,
    )

    assert cost.shape == (1,)
    assert 0.0 < cost.item() <= 1.0
    assert near_gap_cost.item() == 0.0


def test_short_gap_clearance_target_is_bounded_and_requires_forward_progress():
    ordinary_target = torch.tensor([0.12, 0.19, 0.19])
    near_short_gap = torch.tensor([True, True, False])
    forward_progress = torch.tensor([0.2, 0.0, 0.2])

    target = short_gap_clearance_target(
        ordinary_target, near_short_gap, forward_progress, target_height=0.22,
    )

    assert torch.allclose(target, torch.tensor([0.22, 0.19, 0.19]))


def test_short_gap_lift_reward_is_paid_only_for_forward_front_foot_touchdowns():
    peak_clearance = torch.tensor([
        [0.22, 0.20, 0.22, 0.22],
        [0.22, 0.22, 0.22, 0.22],
        [0.22, 0.22, 0.22, 0.22],
    ])
    touchdowns = torch.tensor([
        [True, False, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ])
    target = torch.tensor([0.22, 0.22, 0.22])
    near_short_gap = torch.tensor([True, True, False])
    progress = torch.tensor([0.5, 0.0, 0.5])

    reward = short_gap_lift_reward(
        peak_clearance, touchdowns, target, near_short_gap, progress, std=0.04,
    )

    assert torch.allclose(reward, torch.tensor([0.5, 0.0, 0.0]), atol=1.0e-6)


def test_non_gap_airborne_cost_requires_four_feet_airborne_past_grace():
    air_time = torch.tensor([
        [0.20, 0.20, 0.20, 0.20],
        [0.20, 0.20, 0.00, 0.20],
        [0.20, 0.20, 0.20, 0.20],
    ])
    near_gap = torch.tensor([False, False, True])

    cost = non_gap_airborne_cost(
        air_time, near_gap, grace_seconds=0.06, ramp_seconds=0.12,
    )

    assert cost.shape == (3,)
    assert cost.tolist() == [1.0, 0.0, 0.0]
