from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_runner():
    path = ROOT / "scripts" / "train_long_gap_warmstart_v1333.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("train_long_gap_warmstart_v1333", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def api(name):
    assert runner is not None, "missing isolated V1333 long-gap training runner"
    function = getattr(runner, name, None)
    assert callable(function), f"missing long-gap training contract function: {name}"
    return function


def plan():
    fingerprints = {
        "parent_receipt": "a" * 64,
        "parent_checkpoint": "b" * 64,
        "parent_candidate_source": "c" * 64,
        "active_source": "d" * 64,
        "candidate_source": "e" * 64,
        "candidate_execution": "f" * 64,
        "observation_profile": "1" * 64,
        "reward_config": "2" * 64,
        "task_plan": "3" * 64,
        "ppo_config": "4" * 64,
        "runner": "5" * 64,
        "train_runner": "6" * 64,
        "sim_wrapper": "7" * 64,
        "simulator": "8" * 64,
        "completion_module": "9" * 64,
        "completion_scorer": "0" * 64,
    }
    outputs = {
        "training_log": "v128_work/v132_randomized/v1333/reports/go1-v1333-longgap-from-entrance-warmstart300-a01_full_train.log",
        "training_receipt": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_result_a01.json",
        "authorization_record": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_authorization_a01.json",
        "seed_audit": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_seed_a01.txt",
        "stage_report": "reports/stage_go1-v1333-longgap-from-entrance-warmstart300-a01.json",
        "checkpoint": "checkpoints/stage1/go1-v1333-longgap-from-entrance-warmstart300-a01/model_299.pt",
        "run_dir": "runs/stage1/go1-v1333-longgap-from-entrance-warmstart300-a01",
    }
    return {
        "schema_version": 1,
        "candidate": "v1333-gap-scan-only",
        "run_id": "go1-v1333-longgap-from-entrance-warmstart300-a01",
        "parent_run_id": "go1-v1333-postgap-walk-warmstart200-a01",
        "stage": "stage1",
        "terrain": "lp_acrl",
        "training_roster": "single:gaps_long",
        "num_envs": 512,
        "iterations": 300,
        "final_checkpoint_iteration": 299,
        "episode_seconds": 200,
        "difficulty_min": 0.0,
        "difficulty_max": 0.6,
        "speed_min": 0.35,
        "speed_max": 0.8,
        "initialize_from": "/config/Desktop/sim/checkpoints/stage1/go1-v1333-postgap-walk-warmstart200-a01/model_199.pt",
        "initialize_from_sha256": "02da23fa9d51233c656f6ec95766de260b57cde19cff36a0c6f426e74d7d4f41",
        "initialization": "policy-value-weights-only-reset-optimizer",
        "optimizer_state_restored": False,
        "critic_warmup_iterations": 0,
        "actor_observation_dim": 1073,
        "action_dim": 12,
        "policy_variant": "map_attn_v3",
        "task_plan_path": "configs/task_plan_gaps_long_a01.json",
        "reward_config_path": "configs/rewards_stage1_gaps_v1333.toml",
        "ppo_config_path": "configs/ppo_long_gap_warmstart_a01.json",
        "finish_region": {"start_offset": 33.1, "end_offset": 34.1, "half_width": 4.0},
        "ppo_contract": {
            "warm_start_actor_learning_rate": 5e-5,
            "warm_start_learning_rate": 3e-4,
            "warm_start_clip_param": 0.1,
            "warm_start_entropy_coef": 0.002,
            "warm_start_noise_std": 0.15,
            "num_steps_per_env": 48,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
        "seed_policy": "fresh-unused-audited",
        "training_authorized": False,
        "diagnostic_authorized": False,
        "evaluation_authorized": False,
        "holdout_authorized": False,
        "gate_authorized": False,
        "retry_authorized": False,
        "automatic_follow_on_authorized": False,
        "automatic_gate_authorized": False,
        "automatic_retry_authorized": False,
        "input_fingerprints": fingerprints,
        "output_paths": outputs,
    }


def test_plan_pins_true_long_gap_warmstart_and_full_training_budget():
    api("validate_plan")(plan())


def test_plan_rejects_wrong_family_or_unapproved_follow_on():
    candidate = plan()
    candidate["training_roster"] = "single:combined_part_entrance"
    with pytest.raises(ValueError, match="training_roster"):
        api("validate_plan")(candidate)
    candidate = plan()
    candidate["diagnostic_authorized"] = True
    with pytest.raises(ValueError, match="diagnostic_authorized"):
        api("validate_plan")(candidate)


def test_task_plan_must_have_exactly_512_long_gap_environments():
    task = {
        "families": ["gaps_long"] * 512,
        "difficulty_level": [0] * 147 + [1] * 365,
        "speed_level": ([0] * 111 + [1] * 102 + [2] * 100
                        + [3] * 102 + [4] * 97),
        "allowed_speed_levels": [0, 1, 2, 3, 4],
        "speed_min": 0.35,
        "speed_max": 0.8,
        "target_level": 1,
    }
    api("validate_task_plan")(task)
    task["families"][511] = "combined_part_entrance"
    with pytest.raises(ValueError, match="gaps_long"):
        api("validate_task_plan")(task)


def test_command_uses_weights_only_initialization_and_exact_budget():
    command = api("training_command")(
        plan(), 1_975_310_007, Path("/sim"),
        Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert command[command.index("--initialize-from") + 1] == plan()["initialize_from"]
    assert "--resume" not in command
    assert command[command.index("--iterations") + 1] == "300"
    assert command[command.index("--num-envs") + 1] == "512"
    assert "evaluate" not in command and "--certify" not in command


def test_child_environment_pins_long_gap_plan_and_isolated_ppo_file():
    env = api("training_environment")(
        {},
        {"GO1_POLICY_VARIANT": "map_attn_v3", "GO1_V8_TERRAIN_CONTROL": "1",
         "GO1_PRIVILEGED_ACTOR": "0", "GO1_V1333_GAP_SENSITIVITY": "1",
         "GO1_V1333_GAP_POSTURE_RELIEF": "0"},
        plan(), Path("/sim"), Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert env["GO1_TASK_PLAN"].endswith("configs/task_plan_gaps_long_a01.json")
    assert env["GO1_PPO_CONFIG_PATH"].endswith("configs/ppo_long_gap_warmstart_a01.json")
    assert env["GO1_PRIVILEGED_ACTOR"] == "0"


def test_training_completion_requires_all_positive_artifacts_and_new_metric():
    report = {
        "run": plan()["run_id"],
        "iterations": 300,
        "task_plan": "/sim/v128_work/v132_randomized/v1333/configs/task_plan_gaps_long_a01.json",
        "episodes": 512,
        "completion_rate": 0.0,
        "task_finish_reached": 256,
        "task_completed": 128,
        "task_completion_contract": "task_finish_reached and not fallen and not fell_into_gap",
        "task_completion_rate": 0.25,
        "no_fall_no_gap_rate": 0.5,
        "completion_metric_scope": "legacy success; training rollouts, not independent evaluation",
    }
    api("validate_stage_report")(report, plan(), Path(report["task_plan"]))
    report["iterations"] = 299
    with pytest.raises(ValueError, match="iterations"):
        api("validate_stage_report")(report, plan(), Path(report["task_plan"]))


def test_one_shot_authorization_is_exact():
    api("require_one_shot")("one-shot")
    with pytest.raises(ValueError, match="one-shot"):
        api("require_one_shot")("yes")
