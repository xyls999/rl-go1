from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

from go1_hier_sim.reward_config import load_stage1_rewards  # noqa: E402

A03_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a03.toml"
A04_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a04.toml"


def load_runner():
    path = ROOT / "scripts" / "train_long_gap_warmstart_v1333_a04.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("train_long_gap_warmstart_v1333_a04", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def api(name):
    assert runner is not None, "missing isolated V1333 long-gap a04 runner"
    function = getattr(runner, name, None)
    assert callable(function), f"missing long-gap a04 contract function: {name}"
    return function


def plan():
    value = dict(runner.FIXED)
    value.update({name: False for name in runner.FLAG_KEYS})
    value["input_fingerprints"] = {
        name: ("e" * 64 if name == "candidate_source" else "d" * 64 if name == "parent_candidate_source" else "a" * 64)
        for name in runner.INPUT_KEYS
    }
    value["output_paths"] = dict(runner.OUTPUT_PATHS)
    return value


def test_a04_plan_pins_900_iterations_a03_weights_only_parent_and_no_follow_on():
    assert runner is not None, "missing isolated V1333 long-gap a04 runner"
    candidate = plan()
    api("validate_plan")(candidate)
    assert candidate["iterations"] == 900
    assert candidate["final_checkpoint_iteration"] == 899
    assert candidate["run_id"] == "go1-v1333-longgap-air-distance-warmstart900-a04"
    assert candidate["initialize_from"].endswith(
        "go1-v1333-longgap-from-best-entrance-warmstart300-a03/model_299.pt"
    )
    assert candidate["initialization"] == "policy-value-weights-only-reset-optimizer"
    assert candidate["optimizer_state_restored"] is False

    candidate = plan()
    candidate["iterations"] = 300
    with pytest.raises(ValueError, match="iterations"):
        api("validate_plan")(candidate)
    candidate = plan()
    candidate["training_authorized"] = True
    with pytest.raises(ValueError, match="training_authorized"):
        api("validate_plan")(candidate)
    candidate = plan()
    candidate["initialize_from"] = "/tmp/unauthorized.pt"
    with pytest.raises(ValueError, match="initialize_from"):
        api("validate_plan")(candidate)


def test_a04_reward_delta_adds_only_distance_term_and_preserves_a03_safety():
    assert A04_CONFIG.is_file(), "missing isolated long-gap a04 reward profile"
    baseline = load_stage1_rewards(A03_CONFIG)
    updated = load_stage1_rewards(A04_CONFIG)
    assert set(updated) == set(baseline) | {"airborne_forward_progress"}
    for term, values in baseline.items():
        assert updated[term] == values
    assert updated["airborne_forward_progress"] == {"weight": 20.0, "max_credit_m": 0.45}
    api("validate_reward_delta")(baseline, updated, plan()["reward_changes"])

    unsafe = dict(updated)
    unsafe["fell_into_gap"] = dict(updated["fell_into_gap"], weight=-1000.0)
    with pytest.raises(ValueError, match="reward delta"):
        api("validate_reward_delta")(baseline, unsafe, plan()["reward_changes"])


def test_a04_command_warm_starts_from_a03_for_900_iterations():
    command = api("training_command")(
        plan(), 1_975_310_007, Path("/sim"), Path("/sim/v128_work/v132_randomized/v1333")
    )
    assert command[command.index("--initialize-from") + 1] == plan()["initialize_from"]
    assert command[command.index("--iterations") + 1] == "900"
    assert command[command.index("--num-envs") + 1] == "512"
    assert "--resume" not in command


def test_a04_environment_selects_only_the_new_reward_profile():
    env = api("training_environment")(
        {},
        {
            "GO1_POLICY_VARIANT": "map_attn_v3",
            "GO1_V8_TERRAIN_CONTROL": "1",
            "GO1_PRIVILEGED_ACTOR": "0",
            "GO1_V1333_GAP_SENSITIVITY": "1",
            "GO1_V1333_GAP_POSTURE_RELIEF": "0",
        },
        plan(),
        Path("/sim"),
        Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert env["GO1_TASK_PLAN"].endswith("configs/task_plan_gaps_long_a02.json")
    assert env["GO1_PPO_CONFIG_PATH"].endswith("configs/ppo_long_gap_warmstart_a01.json")
    assert env["GO1_REWARD_CONFIG_PATH"].endswith("configs/rewards_stage1_gaps_v1333_longgap_a04.toml")
    assert env["GO1_PRIVILEGED_ACTOR"] == "0"


def test_a04_report_labels_training_rollouts_not_independent_evaluation():
    report = {
        "run": plan()["run_id"],
        "iterations": 900,
        "task_plan": "/sim/v128_work/v132_randomized/v1333/configs/task_plan_gaps_long_a02.json",
        "episodes": 512,
        "completion_rate": 0.0,
        "task_finish_reached": 0,
        "task_completed": 0,
        "task_completion_contract": "task_finish_reached and not fallen and not fell_into_gap",
        "task_completion_rate": 0.0,
        "no_fall_no_gap_rate": 0.01,
        "completion_metric_scope": "legacy success; training rollouts, not independent evaluation",
    }
    api("validate_stage_report")(report, plan(), Path(report["task_plan"]))
    report["completion_metric_scope"] = "independent evaluation"
    with pytest.raises(ValueError, match="scope"):
        api("validate_stage_report")(report, plan(), Path(report["task_plan"]))
