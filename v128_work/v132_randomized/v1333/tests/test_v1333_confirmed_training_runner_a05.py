from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

from go1_hier_sim.reward_config import load_stage1_rewards  # noqa: E402

A03_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a03.toml"
A04_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a04.toml"
A05_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a05.toml"


def load_runner():
    path = ROOT / "scripts" / "train_long_gap_warmstart_v1333_a05.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("train_long_gap_warmstart_v1333_a05", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def api(name):
    assert runner is not None, "missing isolated V1333 objective-aligned A05 runner"
    function = getattr(runner, name, None)
    assert callable(function), f"missing A05 contract function: {name}"
    return function


def plan():
    value = dict(runner.FIXED)
    value.update({name: False for name in runner.FLAG_KEYS})
    value["input_fingerprints"] = {
        name: ("e" * 64 if name == "candidate_source" else
               "d" * 64 if name == "parent_candidate_source" else "a" * 64)
        for name in runner.INPUT_KEYS
    }
    value["output_paths"] = dict(runner.OUTPUT_PATHS)
    return value


def test_checked_in_a05_plan_matches_the_one_shot_runner_contract():
    assert runner is not None, "missing isolated V1333 objective-aligned A05 runner"
    plan_path = ROOT / "configs" / "long_gap_warmstart_plan_a05.json"
    actual = json.loads(plan_path.read_text(encoding="utf-8"))
    api("validate_plan")(actual)
    assert set(actual["input_fingerprints"]) == set(runner.INPUT_KEYS)
    assert actual["output_paths"] == runner.OUTPUT_PATHS


def test_a05_plan_pins_strict_objective_a03_parent_budget_and_no_follow_on():
    assert runner is not None, "missing isolated V1333 objective-aligned A05 runner"
    candidate = plan()
    api("validate_plan")(candidate)
    assert candidate["run_id"] == "go1-v1333-longgap-objective-aligned-warmstart900-a05"
    assert candidate["iterations"] == 900
    assert candidate["final_checkpoint_iteration"] == 899
    assert candidate["num_envs"] == 512
    assert candidate["initialize_from"].endswith(
        "go1-v1333-longgap-from-best-entrance-warmstart300-a03/model_299.pt"
    )
    assert candidate["initialize_from_sha256"] == (
        "18c495c7f0e2660bff16455a9d7c53b45f7bfe0858be62979436046e8fadf278"
    )
    assert candidate["initialization"] == "policy-value-weights-only-reset-optimizer"
    assert candidate["optimizer_state_restored"] is False
    assert candidate["task_completion_contract"] == (
        "task_finish_reached and not fallen and not fell_into_gap"
    )
    assert candidate["completion_termination_mode"] == "strict-task-completion"
    assert all(candidate[name] is False for name in runner.FLAG_KEYS)

    for name, value in (
        ("iterations", 300),
        ("training_authorized", True),
        ("evaluation_authorized", True),
        ("initialize_from", "/tmp/unauthorized.pt"),
        ("completion_termination_mode", "legacy-success"),
    ):
        candidate = plan()
        candidate[name] = value
        with pytest.raises(ValueError):
            api("validate_plan")(candidate)


def test_a05_adds_only_the_two_approved_terms_and_preserves_a03_and_a04():
    baseline = load_stage1_rewards(A03_CONFIG)
    a04 = load_stage1_rewards(A04_CONFIG)
    updated = load_stage1_rewards(A05_CONFIG)
    expected = {
        "airborne_forward_progress_confirmed": {"weight": 20.0, "max_credit_m": 0.45},
        "task_completed": {"weight": 200.0},
    }
    assert set(updated) == set(baseline) | set(expected)
    assert "airborne_forward_progress" in a04
    assert "airborne_forward_progress" not in updated
    for term, values in baseline.items():
        assert updated[term] == values
    assert {name: updated[name] for name in expected} == expected
    api("validate_reward_delta")(baseline, updated, plan()["reward_changes"])

    unsafe = dict(updated)
    unsafe["fell_into_gap"] = dict(updated["fell_into_gap"], weight=-1000.0)
    with pytest.raises(ValueError, match="reward delta"):
        api("validate_reward_delta")(baseline, unsafe, plan()["reward_changes"])


def test_a05_command_warm_starts_from_a03_for_exactly_512_by_900():
    command = api("training_command")(
        plan(), 1_975_310_007, Path("/sim"), Path("/sim/v128_work/v132_randomized/v1333")
    )
    assert command[command.index("--initialize-from") + 1] == plan()["initialize_from"]
    assert command[command.index("--iterations") + 1] == "900"
    assert command[command.index("--num-envs") + 1] == "512"
    assert "--resume" not in command


def test_a05_environment_selects_only_the_pinned_reward_and_task_profiles():
    environment = api("training_environment")(
        {"V1333_LONG_GAP_TRAIN_AUTHORIZED": "one-shot"},
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
    assert environment["GO1_TASK_PLAN"].endswith("configs/task_plan_gaps_long_a02.json")
    assert environment["GO1_PPO_CONFIG_PATH"].endswith("configs/ppo_long_gap_warmstart_a01.json")
    assert environment["GO1_REWARD_CONFIG_PATH"].endswith(
        "configs/rewards_stage1_gaps_v1333_longgap_a05.toml"
    )
    assert environment["GO1_PRIVILEGED_ACTOR"] == "0"
    assert "V1333_LONG_GAP_TRAIN_AUTHORIZED" not in environment


def test_a05_report_scope_and_one_shot_authorization_are_explicit():
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

    api("require_one_shot")("one-shot")
    for invalid in (None, "", "true", "one-shot-extra"):
        with pytest.raises(ValueError, match="exact one-shot"):
            api("require_one_shot")(invalid)


def test_a05_training_wrapper_is_checked_in_and_matches_its_fingerprint():
    plan = json.loads((ROOT / "configs" / "long_gap_warmstart_plan_a05.json").read_text())
    wrapper = ROOT / "scripts" / "sim_v1333.sh"
    content = wrapper.read_text(encoding="utf-8")
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == plan["input_fingerprints"]["sim_wrapper"]
    assert '"${1:-}" == "train" && "${V1333_TRAIN_CHILD:-}" == "1"' in content
    assert '"${expected_runner} train"' in content
    assert 'readlink -f "/proc/${PPID}/exe"' in content
    assert "V1333_LONG_GAP_TRAIN_AUTHORIZED=one-shot" in content
    assert 'readlink -f /proc/$$/fd/9' in content
    assert "flock -n -E 75 8" in content


def test_a05_requires_the_simulator_lock_to_be_actually_held(tmp_path, monkeypatch):
    lock_path = tmp_path / "sim.lock"
    lock_path.touch()
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "_assert_lock_held", lambda: None)

    with pytest.raises(ValueError, match="open but the flock is not held"):
        api("assert_simulator_lock_held")()

    fd = os.open(lock_path, os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api("assert_simulator_lock_held")()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_a05_report_rejects_inconsistent_strict_completion_counts_and_rates():
    report = {
        "run": plan()["run_id"],
        "iterations": 900,
        "task_plan": "/sim/configs/task_plan_gaps_long_a02.json",
        "episodes": 4,
        "completion_rate": 0.0,
        "task_finish_reached": 2,
        "task_completed": 1,
        "task_completion_contract": "task_finish_reached and not fallen and not fell_into_gap",
        "task_completion_rate": 0.25,
        "no_fall_no_gap_rate": 0.5,
        "completion_metric_scope": "legacy success; training rollouts, not independent evaluation",
    }
    api("validate_stage_report")(report, plan(), Path(report["task_plan"]))

    inconsistent = dict(report, task_completed=3)
    with pytest.raises(ValueError, match="exceeds finish-reached"):
        api("validate_stage_report")(inconsistent, plan(), Path(report["task_plan"]))

    inconsistent = dict(report, task_completion_rate=0.5)
    with pytest.raises(ValueError, match="disagrees with its count"):
        api("validate_stage_report")(inconsistent, plan(), Path(report["task_plan"]))

    inconsistent = dict(report, no_fall_no_gap_rate=0.2)
    with pytest.raises(ValueError, match="exceeds no-fall/no-gap rate"):
        api("validate_stage_report")(inconsistent, plan(), Path(report["task_plan"]))
