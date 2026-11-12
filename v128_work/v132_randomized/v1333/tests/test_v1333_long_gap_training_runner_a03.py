from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_runner():
    path = ROOT / "scripts" / "train_long_gap_warmstart_v1333_a03.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("train_long_gap_warmstart_v1333_a03", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def api(name):
    assert runner is not None, "missing isolated V1333 long-gap a03 runner"
    function = getattr(runner, name, None)
    assert callable(function), f"missing long-gap a03 contract function: {name}"
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
        "base_reward_config": "3" * 64,
        "task_plan": "4" * 64,
        "ppo_config": "5" * 64,
        "runner": "6" * 64,
        "train_runner": "7" * 64,
        "sim_wrapper": "8" * 64,
        "simulator": "9" * 64,
        "completion_module": "0" * 64,
        "completion_scorer": "a" * 64,
    }
    outputs = {
        "training_log": "v128_work/v132_randomized/v1333/reports/go1-v1333-longgap-from-best-entrance-warmstart300-a03_full_train.log",
        "training_receipt": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_result_a03.json",
        "authorization_record": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_authorization_a03.json",
        "seed_audit": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_seed_a03.txt",
        "stage_report": "reports/stage_go1-v1333-longgap-from-best-entrance-warmstart300-a03.json",
        "checkpoint": "checkpoints/stage1/go1-v1333-longgap-from-best-entrance-warmstart300-a03/model_299.pt",
        "run_dir": "runs/stage1/go1-v1333-longgap-from-best-entrance-warmstart300-a03",
    }
    return {
        "schema_version": 1,
        "candidate": "v1333-gap-scan-only",
        "run_id": "go1-v1333-longgap-from-best-entrance-warmstart300-a03",
        "parent_run_id": "go1-v1333-gap-scan-only-s01-p1-d1-continue650",
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
        "initialize_from": "/config/Desktop/sim/checkpoints/stage1/go1-v1333-gap-scan-only-s01-p1-d1-continue650/model_1298.pt",
        "initialize_from_sha256": "15351c88f106f1dfc1d9226f9cdd8284050b63218f3b20369c3337ecb177d7ad",
        "initialization": "policy-value-weights-only-reset-optimizer",
        "optimizer_state_restored": False,
        "critic_warmup_iterations": 0,
        "actor_observation_dim": 1073,
        "action_dim": 12,
        "policy_variant": "map_attn_v3",
        "task_plan_path": "configs/task_plan_gaps_long_a02.json",
        "reward_config_path": "configs/rewards_stage1_gaps_v1333_longgap_a03.toml",
        "base_reward_config_path": "configs/rewards_stage1_gaps_v1333_model1298_base_a02.toml",
        "reward_changes": {
            "forward_progress.weight": [4.0, 8.0],
            "four_feet_jump_air_time.weight": [0.0, 4.0],
        },
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


def test_a03_plan_pins_best_parent_budget_and_no_follow_on_authority():
    api("validate_plan")(plan())
    candidate = plan()
    candidate["training_authorized"] = True
    with pytest.raises(ValueError, match="training_authorized"):
        api("validate_plan")(candidate)
    candidate = plan()
    candidate["initialize_from"] = "/tmp/not-the-approved-parent.pt"
    with pytest.raises(ValueError, match="initialize_from"):
        api("validate_plan")(candidate)


def test_a03_reward_delta_is_exact_and_preserves_safety_penalties():
    baseline = {
        "forward_progress": {"weight": 4.0, "max_progress_speed": 0.9},
        "four_feet_jump_air_time": {"weight": 0.0, "max_air_time": 0.55},
        "fallen": {"weight": -16.0},
        "fell_into_gap": {"weight": -2000.0},
    }
    updated = {
        "forward_progress": {"weight": 8.0, "max_progress_speed": 0.9},
        "four_feet_jump_air_time": {"weight": 4.0, "max_air_time": 0.55},
        "fallen": {"weight": -16.0},
        "fell_into_gap": {"weight": -2000.0},
    }
    api("validate_reward_delta")(baseline, updated, plan()["reward_changes"])
    updated["fell_into_gap"]["weight"] = -1000.0
    with pytest.raises(ValueError, match="reward delta"):
        api("validate_reward_delta")(baseline, updated, plan()["reward_changes"])


def test_a03_command_uses_model1298_weights_only_and_exact_training_budget():
    command = api("training_command")(
        plan(), 1_975_310_007, Path("/sim"),
        Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert command[command.index("--initialize-from") + 1] == plan()["initialize_from"]
    assert "--resume" not in command
    assert command[command.index("--iterations") + 1] == "300"
    assert command[command.index("--num-envs") + 1] == "512"


def test_a03_child_environment_selects_a03_rewards_only():
    env = api("training_environment")(
        {},
        {"GO1_POLICY_VARIANT": "map_attn_v3", "GO1_V8_TERRAIN_CONTROL": "1",
         "GO1_PRIVILEGED_ACTOR": "0", "GO1_V1333_GAP_SENSITIVITY": "1",
         "GO1_V1333_GAP_POSTURE_RELIEF": "0"},
        plan(), Path("/sim"), Path("/sim/v128_work/v132_randomized/v1333"),
    )
    assert env["GO1_TASK_PLAN"].endswith("configs/task_plan_gaps_long_a02.json")
    assert env["GO1_PPO_CONFIG_PATH"].endswith("configs/ppo_long_gap_warmstart_a01.json")
    assert env["GO1_REWARD_CONFIG_PATH"].endswith("configs/rewards_stage1_gaps_v1333_longgap_a03.toml")
    assert env["GO1_PRIVILEGED_ACTOR"] == "0"


def test_a03_stage_report_contract_is_training_rollouts_not_evaluation():
    report = {
        "run": plan()["run_id"],
        "iterations": 300,
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
