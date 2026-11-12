import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "/config/Desktop/sim/checkpoints/stage1/go1-v1333-gap-scan-only-s01-p1-d1-continue650/model_1298.pt"
CHECKPOINT_SHA256 = "15351c88f106f1dfc1d9226f9cdd8284050b63218f3b20369c3337ecb177d7ad"
RUN_ID = "go1-v1333-gap-scan-fc015-warmstart200-a02"


def load_runner():
    path = ROOT / "scripts" / "warmstart_train_v1333.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("warmstart_train_v1333", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def api(name):
    assert runner is not None, "missing isolated V1333 reward warm-start runner"
    function = getattr(runner, name, None)
    assert callable(function), f"missing V1333 warm-start contract function: {name}"
    return function


def pilot_plan():
    return {
        "schema_version": 1,
        "candidate": "v1333-gap-scan-only", "run_id": RUN_ID,
        "parent_run_id": "go1-v1333-gap-scan-only-s01-p1-d1-continue650",
        "stage": "stage1", "terrain": "lp_acrl",
        "training_roster": "single:combined_part_entrance", "num_envs": 512,
        "episode_seconds": 200, "difficulty_min": 0.0, "difficulty_max": 0.6,
        "speed_min": 0.35, "speed_max": 0.8,
        "initialize_from": CHECKPOINT, "initialize_from_sha256": CHECKPOINT_SHA256,
        "initialization": "policy-weights-only-reset-optimizer",
        "iterations": 200, "final_checkpoint_iteration": 199,
        "critic_warmup_iterations": 0, "optimizer_state_restored": False,
        "training_authorized": False,
        "task_plan_path": "configs/task_plan.json",
        "reward_config_path": "configs/rewards_stage1_gaps_v1333.toml",
        "source_policy": "pin-parent-and-current-fingerprints; mismatch-is-recorded",
        "seed_policy": "fresh-unused-audited",
        "input_fingerprints": {
            "parent_receipt": "a" * 64,
            "parent_candidate_source": "b" * 64,
            "active_source": "c" * 64,
            "candidate_source": "d" * 64,
            "reward_config": "e" * 64,
            "task_plan": "f" * 64,
            "candidate_execution": "1" * 64,
            "observation_profile": "2" * 64,
            "ppo_config": "3" * 64,
            "runner": "4" * 64,
            "train_runner": "5" * 64,
            "sim_wrapper": "6" * 64,
            "simulator": "7" * 64,
        },
        "diagnostic_authorized": False, "evaluation_authorized": False,
        "holdout_authorized": False, "gate_authorized": False,
        "retry_authorized": False, "automatic_follow_on_authorized": False,
        "automatic_gate_authorized": False, "automatic_retry_authorized": False,
        "output_paths": {
            "training_log": "v128_work/v132_randomized/v1333/reports/go1-v1333-gap-scan-fc015-warmstart200-a02_full_train.log",
            "pilot_receipt": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_result_fc015_a02.json",
            "authorization_record": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_authorization_fc015_a02.json",
            "seed_audit": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_seed_fc015_a02.txt",
            "stage_report": "reports/stage_go1-v1333-gap-scan-fc015-warmstart200-a02.json",
            "checkpoint": "checkpoints/stage1/go1-v1333-gap-scan-fc015-warmstart200-a02/model_199.pt",
            "run_dir": "runs/stage1/go1-v1333-gap-scan-fc015-warmstart200-a02",
        },
    }


def test_warmstart_plan_pins_model_1298_and_resets_optimizer_for_200_iterations():
    api("require_warmstart_plan")(pilot_plan())


def test_warmstart_plan_rejects_optimizer_resume_or_follow_on_evaluation():
    candidate = pilot_plan()
    candidate["optimizer_state_restored"] = True
    with pytest.raises(ValueError, match="optimizer"):
        api("require_warmstart_plan")(candidate)
    candidate = pilot_plan()
    candidate["evaluation_authorized"] = True
    with pytest.raises(ValueError, match="evaluation"):
        api("require_warmstart_plan")(candidate)


def test_consumed_warmstart_bundle_fails_closed_after_candidate_source_changes():
    with pytest.raises(ValueError, match="fingerprints differ"):
        api("validate_warmstart_bundle")()


def test_warmstart_plan_rejects_unpinned_output_path():
    candidate = pilot_plan()
    candidate["output_paths"]["checkpoint"] = "checkpoints/stage1/unsafe/model_199.pt"
    with pytest.raises(ValueError, match="output paths"):
        api("require_warmstart_plan")(candidate)


def test_warmstart_command_uses_initialize_from_not_resume():
    command = api("warmstart_command")(
        pilot_plan(), 1_975_310_007, Path("/sim"),
        Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert command[command.index("--initialize-from") + 1] == CHECKPOINT
    assert "--resume" not in command
    assert command[command.index("--iterations") + 1] == "200"
    assert command[command.index("--run-name") + 1] == RUN_ID
    assert "evaluate" not in command and "--certify" not in command


def test_warmstart_requires_one_shot_authorization_before_writing_outputs(monkeypatch):
    protected = [ROOT / "reports" / name for name in (
        "reward_pilot_warmstart_authorization_fc015_a02.json",
        "reward_pilot_warmstart_result_fc015_a02.json")]
    before = {path: path.read_bytes() if path.exists() else None for path in protected}
    monkeypatch.delenv("V1333_FC015_WARMSTART_AUTHORIZED", raising=False)
    with pytest.raises(ValueError, match="one-shot"):
        api("run_warmstart_pilot")()
    after = {path: path.read_bytes() if path.exists() else None for path in protected}
    assert after == before
