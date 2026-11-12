#!/usr/bin/env python3
"""One-shot V1333 long-gap warm-start training; never evaluates or retries."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from train_v1333 import (
    LOCK_PATH, MAX_SEED, MIN_SEED, PACKAGE_REL, ROOT, WORKSPACE,
    _assert_lock_held, _seed_roots, active_source_fingerprint,
    build_training_environment, ensure_idle, read_json, safe_path, select_fresh_seed,
    sha256, tree_fingerprint, write_json_exclusive,
)

RUN_ID = "go1-v1333-longgap-from-best-entrance-warmstart300-a03"
PARENT_RUN_ID = "go1-v1333-gap-scan-only-s01-p1-d1-continue650"
PARENT_RECEIPT = ROOT / "reports/continuation_result_v1333_c650.json"
INITIAL_CHECKPOINT = Path(
    "/config/Desktop/sim/checkpoints/stage1/"
    "go1-v1333-gap-scan-only-s01-p1-d1-continue650/model_1298.pt"
)
INITIAL_CHECKPOINT_SHA256 = "15351c88f106f1dfc1d9226f9cdd8284050b63218f3b20369c3337ecb177d7ad"
PLAN_PATH = ROOT / "configs/long_gap_warmstart_plan_a03.json"
TASK_PLAN_PATH = ROOT / "configs/task_plan_gaps_long_a02.json"
PPO_CONFIG_PATH = ROOT / "configs/ppo_long_gap_warmstart_a01.json"
BASE_REWARD_CONFIG_PATH = ROOT / "configs/rewards_stage1_gaps_v1333_model1298_base_a02.toml"
REWARD_CONFIG_PATH = ROOT / "configs/rewards_stage1_gaps_v1333_longgap_a03.toml"
COMPLETION_CONTRACT = "task_finish_reached and not fallen and not fell_into_gap"
REWARD_CHANGES = {
    "forward_progress.weight": [4.0, 8.0],
    "four_feet_jump_air_time.weight": [0.0, 4.0],
}
INPUT_KEYS = (
    "parent_receipt", "parent_checkpoint", "parent_candidate_source", "active_source",
    "candidate_source", "candidate_execution", "observation_profile", "reward_config",
    "base_reward_config", "task_plan", "ppo_config", "runner", "train_runner",
    "sim_wrapper", "simulator", "completion_module", "completion_scorer",
)
OUTPUT_PATHS = {
    "training_log": "v128_work/v132_randomized/v1333/reports/"
                    "go1-v1333-longgap-from-best-entrance-warmstart300-a03_full_train.log",
    "training_receipt": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_result_a03.json",
    "authorization_record": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_authorization_a03.json",
    "seed_audit": "v128_work/v132_randomized/v1333/reports/long_gap_warmstart_seed_a03.txt",
    "stage_report": "reports/stage_go1-v1333-longgap-from-best-entrance-warmstart300-a03.json",
    "checkpoint": "checkpoints/stage1/go1-v1333-longgap-from-best-entrance-warmstart300-a03/model_299.pt",
    "run_dir": "runs/stage1/go1-v1333-longgap-from-best-entrance-warmstart300-a03",
}
FIXED = {
    "schema_version": 1,
    "candidate": "v1333-gap-scan-only",
    "run_id": RUN_ID,
    "parent_run_id": PARENT_RUN_ID,
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
    "initialize_from": str(INITIAL_CHECKPOINT),
    "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
    "initialization": "policy-value-weights-only-reset-optimizer",
    "optimizer_state_restored": False,
    "critic_warmup_iterations": 0,
    "actor_observation_dim": 1073,
    "action_dim": 12,
    "policy_variant": "map_attn_v3",
    "task_plan_path": "configs/task_plan_gaps_long_a02.json",
    "reward_config_path": "configs/rewards_stage1_gaps_v1333_longgap_a03.toml",
    "base_reward_config_path": "configs/rewards_stage1_gaps_v1333_model1298_base_a02.toml",
    "reward_changes": REWARD_CHANGES,
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
}
FLAG_KEYS = (
    "training_authorized", "diagnostic_authorized", "evaluation_authorized",
    "holdout_authorized", "gate_authorized", "retry_authorized",
    "automatic_follow_on_authorized", "automatic_gate_authorized", "automatic_retry_authorized",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: dict) -> None:
    require(isinstance(plan, dict), "V1333 long-gap plan must be an object")
    for name, value in FIXED.items():
        require(plan.get(name) == value, f"V1333 long-gap plan field {name} mismatch")
    for name in FLAG_KEYS:
        require(plan.get(name) is False, f"V1333 long-gap plan must not authorize {name}")
    fingerprints = plan.get("input_fingerprints")
    require(isinstance(fingerprints, dict) and set(fingerprints) == set(INPUT_KEYS),
            "V1333 long-gap input fingerprint set mismatch")
    for name, value in fingerprints.items():
        require(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                f"V1333 long-gap fingerprint {name} is malformed")
    require(fingerprints["candidate_source"] != fingerprints["parent_candidate_source"],
            "V1333 long-gap plan must record candidate source drift")
    require(plan.get("output_paths") == OUTPUT_PATHS,
            "V1333 long-gap outputs are not pinned to the one-shot run")
    expected = set(FIXED) | set(FLAG_KEYS) | {"input_fingerprints", "output_paths"}
    require(set(plan) == expected, "V1333 long-gap plan contains unknown or missing fields")


def validate_task_plan(task: dict) -> None:
    require(isinstance(task, dict), "long-gap task plan must be an object")
    families = task.get("families")
    require(isinstance(families, list) and len(families) == 512
            and set(families) == {"gaps_long"},
            "task plan must assign gaps_long to all 512 environments")
    difficulty = task.get("difficulty_level")
    speed = task.get("speed_level")
    require(isinstance(difficulty, list) and len(difficulty) == 512,
            "long-gap difficulty levels must have 512 assignments")
    require(isinstance(speed, list) and len(speed) == 512,
            "long-gap speed levels must have 512 assignments")
    require(Counter(difficulty) == Counter({0: 147, 1: 365}),
            "long-gap difficulty-level distribution changed")
    require(Counter(speed) == Counter({0: 111, 1: 102, 2: 100, 3: 102, 4: 97}),
            "long-gap speed-level distribution changed")
    require(task.get("allowed_speed_levels") == [0, 1, 2, 3, 4]
            and task.get("target_level") == 1
            and task.get("speed_min") == 0.35 and task.get("speed_max") == 0.8,
            "long-gap categorical speed/task-level contract changed")


def validate_reward_delta(baseline: dict, updated: dict, expected_changes: dict) -> None:
    require(set(baseline) == set(updated), "reward delta changes the reward-term set")
    actual_changes = {}
    for term, old_values in baseline.items():
        new_values = updated[term]
        require(set(old_values) == set(new_values), f"reward delta changes fields for {term}")
        for field, old_value in old_values.items():
            new_value = new_values[field]
            if old_value != new_value:
                actual_changes[f"{term}.{field}"] = [old_value, new_value]
    require(actual_changes == expected_changes,
            "reward delta differs from the approved airtime/forward-progress changes")


def require_one_shot(value: str | None) -> None:
    require(value == "one-shot", "V1333 long-gap training requires exact one-shot authorization")


def training_command(plan: dict, seed: int, workspace: Path, candidate_root: Path) -> list[str]:
    validate_plan(plan)
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED,
            "V1333 long-gap seed is outside the audited range")
    workspace, candidate_root = Path(workspace).resolve(), Path(candidate_root).resolve()
    require(candidate_root == workspace / PACKAGE_REL, "V1333 long-gap candidate root mismatch")
    return [
        "bash", str(candidate_root / "scripts/sim_v1333.sh"), "train",
        "--stage", plan["stage"], "--terrain", plan["terrain"], "--headless",
        "--device", "cuda:0", "--num-envs", str(plan["num_envs"]),
        "--iterations", str(plan["iterations"]), "--seed", str(seed),
        "--episode-seconds", str(plan["episode_seconds"]),
        "--course-difficulty-min", str(plan["difficulty_min"]),
        "--course-difficulty-max", str(plan["difficulty_max"]),
        "--speed-min", str(plan["speed_min"]), "--speed-max", str(plan["speed_max"]),
        "--run-name", plan["run_id"], "--initialize-from", plan["initialize_from"],
        "--critic-warmup-iterations", str(plan["critic_warmup_iterations"]),
    ]


def training_environment(base_environment: dict[str, str], execution_settings: dict[str, str],
                         plan: dict, workspace: Path, candidate_root: Path) -> dict[str, str]:
    validate_plan(plan)
    workspace, candidate_root = Path(workspace).resolve(), Path(candidate_root).resolve()
    environment = build_training_environment(
        base_environment, execution_settings, workspace, candidate_root)
    environment.update({
        "GO1_TASK_PLAN": str((candidate_root / plan["task_plan_path"]).resolve()),
        "GO1_PPO_CONFIG_PATH": str((candidate_root / plan["ppo_config_path"]).resolve()),
        "GO1_REWARD_CONFIG_PATH": str((candidate_root / plan["reward_config_path"]).resolve()),
        "GO1_POLICY_VARIANT": plan["policy_variant"],
        "GO1_PRIVILEGED_ACTOR": "0",
        "GO1_RUN_ID": plan["run_id"],
    })
    environment.pop("V1333_LONG_GAP_TRAIN_AUTHORIZED", None)
    return environment


def validate_stage_report(report: dict, plan: dict, task_plan_path: Path) -> None:
    validate_plan(plan)
    require(isinstance(report, dict), "long-gap stage report must be an object")
    require(report.get("run") == plan["run_id"], "long-gap stage report run id mismatch")
    require(report.get("iterations") == plan["iterations"],
            "long-gap stage report iterations mismatch")
    require(report.get("task_plan") == str(Path(task_plan_path).resolve()),
            "long-gap stage report task plan mismatch")
    episodes = report.get("episodes")
    require(type(episodes) is int and episodes > 0, "long-gap stage report has no completed episodes")
    require(report.get("task_completion_contract") == COMPLETION_CONTRACT,
            "long-gap stage report is missing the approved completion contract")
    require(report.get("completion_metric_scope") ==
            "legacy success; training rollouts, not independent evaluation",
            "legacy success scope is not clearly labeled")
    for name in ("task_finish_reached", "task_completed"):
        count = report.get(name)
        require(type(count) is int and 0 <= count <= episodes,
                f"long-gap stage report {name} count is invalid")
    for name in ("completion_rate", "task_completion_rate", "no_fall_no_gap_rate"):
        value = report.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0,
                f"long-gap stage report {name} is invalid")


def actual_fingerprints(plan: dict, parent: dict) -> dict[str, str]:
    return {
        "parent_receipt": sha256(PARENT_RECEIPT),
        "parent_checkpoint": sha256(INITIAL_CHECKPOINT),
        "parent_candidate_source": parent.get("candidate_source_fingerprint_after", ""),
        "active_source": active_source_fingerprint(WORKSPACE),
        "candidate_source": tree_fingerprint(ROOT / "source/go1_hier_sim"),
        "candidate_execution": sha256(ROOT / "configs/candidate_execution.json"),
        "observation_profile": sha256(ROOT / "configs/observation_profile.json"),
        "reward_config": sha256(ROOT / plan["reward_config_path"]),
        "base_reward_config": sha256(ROOT / plan["base_reward_config_path"]),
        "task_plan": sha256(ROOT / plan["task_plan_path"]),
        "ppo_config": sha256(ROOT / plan["ppo_config_path"]),
        "runner": sha256(Path(__file__).resolve()),
        "train_runner": sha256(ROOT / "scripts/train_v1333.py"),
        "sim_wrapper": sha256(ROOT / "scripts/sim_v1333.sh"),
        "simulator": sha256(WORKSPACE / "scripts/sim.py"),
        "completion_module": sha256(ROOT / "source/go1_hier_sim/go1_hier_sim/envs/long_gap_completion_v1333.py"),
        "completion_scorer": sha256(ROOT / "scripts/score_long_gap_trace_v1333.py"),
    }


def _validate_ppo(path: Path, plan: dict) -> None:
    ppo = read_json(path)
    expected = plan["ppo_contract"]
    for name in ("warm_start_actor_learning_rate", "warm_start_learning_rate",
                 "warm_start_clip_param", "warm_start_entropy_coef", "warm_start_noise_std"):
        require(ppo.get(name) == expected[name],
                "isolated long-gap PPO warm-start value changed: " + name)
    require(ppo.get("locomotion", {}).get("num_steps_per_env") == expected["num_steps_per_env"]
            and ppo.get("num_learning_epochs") == expected["num_learning_epochs"]
            and ppo.get("num_mini_batches") == expected["num_mini_batches"],
            "isolated long-gap PPO rollout/update shape changed")


def validate_bundle() -> tuple[dict, dict, dict, dict]:
    require(ROOT == WORKSPACE / PACKAGE_REL and ROOT.is_dir() and not ROOT.is_symlink(),
            "V1333 long-gap runner requires the isolated candidate root")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    parent = read_json(PARENT_RECEIPT)
    require(parent.get("status") == "trained-unverified"
            and parent.get("run_id") == PARENT_RUN_ID
            and parent.get("mode") == "full-checkpoint-resume"
            and parent.get("checkpoint_path") == str(INITIAL_CHECKPOINT)
            and parent.get("checkpoint_sha256") == INITIAL_CHECKPOINT_SHA256
            and parent.get("optimizer_state_restored") is True,
            "long-gap best entrance-gap parent receipt/checkpoint contract mismatch")
    require(INITIAL_CHECKPOINT.is_file() and not INITIAL_CHECKPOINT.is_symlink()
            and sha256(INITIAL_CHECKPOINT) == INITIAL_CHECKPOINT_SHA256,
            "long-gap parent checkpoint hash mismatch")
    require(parent.get("candidate_source_fingerprint_after")
            == plan["input_fingerprints"]["parent_candidate_source"],
            "long-gap parent source fingerprint mismatch")
    execution = read_json(ROOT / "configs/candidate_execution.json")
    settings = execution.get("train_environment")
    require(isinstance(settings, dict)
            and execution.get("candidate") == "v1333-gap-scan-only"
            and settings.get("GO1_POLICY_VARIANT") == "map_attn_v3"
            and settings.get("GO1_V8_TERRAIN_CONTROL") == "1"
            and settings.get("GO1_V1333_GAP_SENSITIVITY") == "1"
            and settings.get("GO1_V1333_GAP_POSTURE_RELIEF") == "0"
            and settings.get("GO1_PRIVILEGED_ACTOR") == "0",
            "long-gap actor-visible scan-only execution settings changed")
    profile = read_json(ROOT / "configs/observation_profile.json")
    require(profile.get("actor_observation_dim") == 1073 and profile.get("action_dim") == 12,
            "long-gap observation/action contract changed")
    task = read_json(ROOT / plan["task_plan_path"])
    validate_task_plan(task)
    _validate_ppo(ROOT / plan["ppo_config_path"], plan)
    sys.path.insert(0, str(ROOT / "source/go1_hier_sim"))
    from go1_hier_sim.checkpoints import load_checkpoint
    from go1_hier_sim.reward_config import load_stage1_rewards
    checkpoint = load_checkpoint(str(INITIAL_CHECKPOINT), expected_stage="stage1")
    metadata = checkpoint.get("metadata", {})
    require(metadata.get("iteration") == 1298 and metadata.get("policy_variant") == "map_attn_v3"
            and isinstance(checkpoint.get("models", {}).get("policy"), dict),
            "saved best entrance-gap parent model architecture/iteration mismatch")
    baseline_rewards = load_stage1_rewards(ROOT / plan["base_reward_config_path"])
    current_rewards = load_stage1_rewards(ROOT / plan["reward_config_path"])
    require(metadata.get("reward_config") == baseline_rewards,
            "saved best entrance-gap checkpoint does not match its recorded baseline rewards")
    validate_reward_delta(baseline_rewards, current_rewards, plan["reward_changes"])
    goal = current_rewards.get("goal_reached", {})
    require({name: goal.get(name) for name in ("start_offset", "end_offset", "half_width")}
            == plan["finish_region"],
            "long-gap finish region differs from the reward geometry contract")
    for path in (WORKSPACE / "scripts/sim.py", WORKSPACE / "configs/ppo_gaps_long.json",
                 ROOT / "scripts/sim_v1333.sh", ROOT / "scripts/train_v1333.py",
                 ROOT / "scripts/score_long_gap_trace_v1333.py"):
        require(path.is_file() and not path.is_symlink(),
                "long-gap training dependency missing or symlinked: " + str(path))
    actual = actual_fingerprints(plan, parent)
    require(actual == plan["input_fingerprints"],
            "long-gap training inputs changed; plan fingerprints do not match")
    require(actual["active_source"] == parent.get("active_source_fingerprint_after"),
            "active simulator source differs from the parent training")
    paths = {key: safe_path(WORKSPACE, relative)
             for key, relative in plan["output_paths"].items()}
    require(len(set(paths.values())) == len(paths), "long-gap output paths collide")
    return plan, execution, parent, paths


def _require_outputs_absent(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        require(not path.exists() and not path.is_symlink(),
                f"long-gap output already exists ({name}): {path}")


def _validate_outputs(paths: dict[str, Path], plan: dict) -> dict:
    log_path, report_path = paths["training_log"], paths["stage_report"]
    checkpoint, run_dir = paths["checkpoint"], paths["run_dir"]
    require(log_path.is_file() and not log_path.is_symlink(), "long-gap training log missing")
    require(report_path.is_file() and not report_path.is_symlink(), "long-gap stage report missing")
    require(checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0,
            "long-gap final model_299 checkpoint missing or empty")
    require(run_dir.is_dir() and not run_dir.is_symlink(), "long-gap training run directory missing")
    report = read_json(report_path)
    task_path = (ROOT / plan["task_plan_path"]).resolve()
    validate_stage_report(report, plan, task_path)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require("TRAINING FINISHED;" in log and f"STAGE REPORT: {report_path}" in log,
            "long-gap training log lacks positive completion markers")
    sys.path.insert(0, str(ROOT / "source/go1_hier_sim"))
    from go1_hier_sim.checkpoints import load_checkpoint
    final = load_checkpoint(str(checkpoint), expected_stage="stage1")
    metadata = final.get("metadata", {})
    require(metadata.get("iteration") == plan["final_checkpoint_iteration"],
            "long-gap checkpoint iteration is not 299")
    require(metadata.get("policy_variant") == "map_attn_v3"
            and metadata.get("upstream", {}).get("initialized_from") == plan["initialize_from"],
            "long-gap checkpoint warm-start provenance mismatch")
    return {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "stage_report_path": str(report_path), "stage_report_sha256": sha256(report_path),
        "training_log_path": str(log_path), "training_log_sha256": sha256(log_path),
        "training_episodes": report["episodes"],
        "legacy_success_rate_training_rollouts": report["completion_rate"],
        "task_completion_rate_training_rollouts": report["task_completion_rate"],
        "no_fall_no_gap_rate_training_rollouts": report["no_fall_no_gap_rate"],
        "task_completion_contract": report["task_completion_contract"],
    }


def run_training() -> dict:
    require_one_shot(os.environ.get("V1333_LONG_GAP_TRAIN_AUTHORIZED"))
    _assert_lock_held()
    plan, execution, parent, paths = validate_bundle()
    _require_outputs_absent(paths)
    ensure_idle()
    active_before = active_source_fingerprint(WORKSPACE)
    candidate_before = tree_fingerprint(ROOT / "source/go1_hier_sim")
    seed, files_scanned, bytes_scanned = select_fresh_seed(_seed_roots(WORKSPACE))
    seed_path = paths["seed_audit"]
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("x", encoding="utf-8") as stream:
        stream.write("V1333_LONG_GAP_WARMSTART_SEED_AUDIT_A03_V1\n")
        stream.write(f"SEED {seed}\nAUDIT_UTC {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"FILES_SCANNED {files_scanned}\nBYTES_SCANNED {bytes_scanned}\nHITS []\n")
        stream.write("RESULT UNUSED_IN_AUDITED_TEXT\n")
    authorization_path = paths["authorization_record"]
    authorization = {
        "status": "consumed-one-shot",
        "authorization_scope": "one V1333 gaps_long weights-only warm-start run from saved model_1298; 512 environments x 300 iterations with approved A03 airtime/forward-progress rewards",
        "user_authorized": True,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": plan["run_id"],
        "parent_run_id": PARENT_RUN_ID,
        "train_seed": seed,
        "initialize_from": str(INITIAL_CHECKPOINT),
        "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
        "optimizer_state_restored": False,
        "optimizer_reset": True,
        "parent_receipt_sha256": sha256(PARENT_RECEIPT),
        "training_plan_sha256": sha256(PLAN_PATH),
        "input_fingerprints": plan["input_fingerprints"],
        "seed_audit_path": str(seed_path),
        "seed_audit_sha256": sha256(seed_path),
        "task_family": "gaps_long",
        "task_completion_contract": COMPLETION_CONTRACT,
        "diagnostic_authorized": False,
        "evaluation_authorized": False,
        "holdout_authorized": False,
        "formal_gate_authorized": False,
        "automatic_retry_authorized": False,
    }
    write_json_exclusive(authorization_path, authorization)
    environment = training_environment(os.environ.copy(), execution["train_environment"],
                                       plan, WORKSPACE, ROOT)
    environment["V1333_TRAIN_CHILD"] = "1"
    environment.pop("V1333_LONG_GAP_TRAIN_AUTHORIZED", None)
    command = training_command(plan, seed, WORKSPACE, ROOT)
    log_path = paths["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launched = False
    try:
        with log_path.open("x", encoding="utf-8") as log_stream:
            launched = True
            print(f"V1333_LONG_GAP_TRAINING_STARTED run_id={RUN_ID} seed={seed}", flush=True)
            result = subprocess.run(command, cwd=str(WORKSPACE), env=environment,
                                    stdout=log_stream, stderr=subprocess.STDOUT,
                                    text=True, check=False, pass_fds=(9,))
        require(result.returncode == 0,
                f"V1333 long-gap training child exited {result.returncode}")
        outputs = _validate_outputs(paths, plan)
        active_after = active_source_fingerprint(WORKSPACE)
        candidate_after = tree_fingerprint(ROOT / "source/go1_hier_sim")
        require(active_after == active_before,
                "active simulator source changed during long-gap training")
        require(candidate_after == candidate_before,
                "V1333 candidate source changed during long-gap training")
        ensure_idle()
        receipt = {
            "status": "trained-unverified",
            "training_job_complete": True,
            "capability_validated": False,
            "mode": "policy-value-weights-only-warm-start-reset-optimizer",
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "train_seed": seed,
            "seed_audit_path": str(seed_path),
            "seed_audit_sha256": sha256(seed_path),
            "num_envs": plan["num_envs"],
            "iterations": plan["iterations"],
            "final_checkpoint_iteration": plan["final_checkpoint_iteration"],
            "training_roster": plan["training_roster"],
            "difficulty_range": [plan["difficulty_min"], plan["difficulty_max"]],
            "speed_range": [plan["speed_min"], plan["speed_max"]],
            "actor_observation_dim": plan["actor_observation_dim"],
            "action_dim": plan["action_dim"],
            "policy_variant": plan["policy_variant"],
            "privileged_actor": False,
            "initialize_from": str(INITIAL_CHECKPOINT),
            "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
            "optimizer_state_restored": False,
            "optimizer_reset": True,
            "candidate_source_fingerprint_before": candidate_before,
            "candidate_source_fingerprint_after": candidate_after,
            "active_source_fingerprint_before": active_before,
            "active_source_fingerprint_after": active_after,
            "authorization_record": str(authorization_path),
            "training_plan_path": str(PLAN_PATH),
            "training_plan_sha256": sha256(PLAN_PATH),
            "task_plan_path": str(TASK_PLAN_PATH),
            "task_plan_sha256": sha256(TASK_PLAN_PATH),
            "ppo_config_path": str(PPO_CONFIG_PATH),
            "ppo_config_sha256": sha256(PPO_CONFIG_PATH),
            "base_reward_config_path": str(BASE_REWARD_CONFIG_PATH),
            "base_reward_config_sha256": sha256(BASE_REWARD_CONFIG_PATH),
            "reward_config_path": str(REWARD_CONFIG_PATH),
            "reward_config_sha256": sha256(REWARD_CONFIG_PATH),
            "reward_changes": plan["reward_changes"],
            "task_completion_contract": COMPLETION_CONTRACT,
            "completion_metric_scope": "training rollouts only; not independent evaluation",
            "diagnostic_ran": False,
            "evaluation_ran": False,
            "holdout_ran": False,
            "formal_gate": False,
            "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False,
            **outputs,
        }
        receipt_path = paths["training_receipt"]
        write_json_exclusive(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if launched and not paths["training_receipt"].exists():
            failure = {
                "status": "training-failed",
                "training_job_complete": False,
                "capability_validated": False,
                "run_id": RUN_ID,
                "train_seed": seed,
                "error": f"{type(error).__name__}: {error}",
                "launch_attempted": True,
                "authorization_record": str(authorization_path),
                "candidate_source_fingerprint_before": candidate_before,
                "active_source_fingerprint_before": active_before,
                "automatic_retry_authorized": False,
                "automatic_follow_on_authorized": False,
            }
            try:
                write_json_exclusive(paths["training_receipt"], failure)
            except BaseException as receipt_error:
                print(f"V1333_LONG_GAP_FAILURE_RECEIPT_BLOCKED: {receipt_error}", file=sys.stderr)
        raise


def run_preflight() -> dict:
    _assert_lock_held()
    plan, execution, _, paths = validate_bundle()
    _require_outputs_absent(paths)
    ensure_idle()
    command = training_command(plan, MIN_SEED, WORKSPACE, ROOT)
    environment = training_environment({}, execution["train_environment"],
                                       plan, WORKSPACE, ROOT)
    require(command[command.index("--num-envs") + 1] == "512"
            and command[command.index("--iterations") + 1] == "300"
            and command[command.index("--initialize-from") + 1] == str(INITIAL_CHECKPOINT),
            "long-gap preflight command does not match approved training scope")
    require(environment.get("GO1_PRIVILEGED_ACTOR") == "0"
            and environment.get("GO1_TASK_PLAN") == str(TASK_PLAN_PATH.resolve())
            and environment.get("GO1_PPO_CONFIG_PATH") == str(PPO_CONFIG_PATH.resolve())
            and environment.get("GO1_REWARD_CONFIG_PATH") == str(REWARD_CONFIG_PATH.resolve()),
            "long-gap preflight environment does not match the isolated plan")
    return {
        "status": "preflight-passed-no-training-started",
        "run_id": plan["run_id"],
        "task_family": "gaps_long",
        "num_envs": 512,
        "iterations": 300,
        "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
        "optimizer_state_restored": False,
        "candidate_source_fingerprint": plan["input_fingerprints"]["candidate_source"],
        "active_source_fingerprint": plan["input_fingerprints"]["active_source"],
        "output_paths_absent": True,
        "training_authorized_by_plan": False,
        "diagnostic_started": False,
        "holdout_started": False,
        "formal_gate_started": False,
        "seed_selected": False,
    }


def main() -> int:
    if sys.argv[1:] == ["preflight"]:
        try:
            print(json.dumps(run_preflight(), sort_keys=True))
        except BaseException as error:
            print(f"V1333_LONG_GAP_PREFLIGHT_BLOCKED: {type(error).__name__}: {error}",
                  file=sys.stderr)
            return 1
        return 0
    if sys.argv[1:] != ["train"]:
        print("V1333 long-gap runner accepts only preflight or one-shot train", file=sys.stderr)
        return 2
    try:
        receipt = run_training()
    except BaseException as error:
        print(f"V1333_LONG_GAP_TRAINING_BLOCKED: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1
    print(json.dumps({
        "status": receipt["status"],
        "run_id": receipt["run_id"],
        "train_seed": receipt["train_seed"],
        "training_job_complete": receipt["training_job_complete"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "task_completion_rate_training_rollouts": receipt["task_completion_rate_training_rollouts"],
        "diagnostic_ran": receipt["diagnostic_ran"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
