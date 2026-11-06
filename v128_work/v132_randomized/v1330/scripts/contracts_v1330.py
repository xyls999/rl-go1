"""Fail-closed contracts for the single training-only V1330 run."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

RUN_ID = "go1-v1330-gait-gap-aware-s01-p1-d1"
DIAGNOSTIC_RUN_ID = RUN_ID + "-diagnostic-a01"
PACKAGE_REL = Path("v128_work/v132_randomized/v1330")
NUM_ENVS = 512
ITERATIONS = 650
EVALUATION_EPISODES = 512
CRITIC_WARMUP_ITERATIONS = 0
MIN_SEED = 1_000_000_000
MAX_SEED = 2_147_483_647
FORBIDDEN_SEEDS = frozenset({
    42, 2026, 2028, 2029, 1_189_662_175, 1_242_088_145, 1_607_361_419,
    1_752_494_079, 1_802_284_695, 1_812_341_965,
})
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED = {
    "run_id": RUN_ID,
    "diagnostic_run_id": DIAGNOSTIC_RUN_ID,
    "num_envs": NUM_ENVS,
    "iterations": ITERATIONS,
    "critic_warmup_iterations": CRITIC_WARMUP_ITERATIONS,
    "terrain": "lp_acrl",
    "training_roster": "single:combined_part_entrance",
    "difficulty_min": 0.0,
    "difficulty_max": 0.6,
    "speed_min": 0.35,
    "speed_max": 0.8,
    "episode_seconds": 200,
    "initialization": "random",
    "run_authorized": True,
    "training_authorized": True,
    "diagnostic_authorized": False,
    "automatic_gate_authorized": False,
    "automatic_retry_authorized": False,
    "automatic_follow_on_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(value, label: str) -> None:
    require(isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256 digest")


def _require_fresh_seed(seed, purpose: str) -> None:
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED and seed not in FORBIDDEN_SEEDS,
            f"{purpose} seed is outside the approved fresh-seed range")


def validate_train_contract(config: dict, spec: dict) -> None:
    require(isinstance(config, dict) and isinstance(spec, dict),
            "V1330 training config and spec must be JSON objects")
    for name, expected in EXPECTED.items():
        require(config.get(name) == expected and spec.get(name) == expected,
                f"V1330 fixed execution contract changed: {name}")
    for forbidden in ("initial_checkpoint", "initial_checkpoint_sha256", "checkpoint",
                      "resume", "initialize_from"):
        require(forbidden not in config and forbidden not in spec,
                "scratch training cannot reference a prior model: " + forbidden)
    for value in (config, spec):
        require(value.get("final_holdout", False) is False and
                value.get("final_gate_passed", False) is False,
                "V1330 training cannot authorize a holdout or gate")
    _require_fresh_seed(spec.get("train_seed"), "training")
    for name in ("manifest_sha256", "task_plan_sha256", "reward_config_sha256",
                 "observation_profile_sha256", "train_seed_audit_sha256"):
        _require_hash(spec.get(name), name)
    for name in ("task_plan_path", "reward_config_path", "observation_profile_path",
                 "train_seed_audit_path", "training_log_path", "training_receipt_path",
                 "checkpoint_path", "stage_report_path"):
        require(isinstance(spec.get(name), str) and spec[name],
                f"missing V1330 training input/output path: {name}")


def train_command(workspace: Path, config: dict, spec: dict, plan: Path) -> list[str]:
    validate_train_contract(config, spec)
    expected_plan = workspace / PACKAGE_REL / spec["task_plan_path"]
    require(plan == expected_plan and plan.is_file() and not plan.is_symlink(),
            "V1330 task plan path missing or changed")
    return [
        str(workspace / "sim.sh"), "train", "--stage", "stage1", "--terrain", "lp_acrl",
        "--headless", "--device", "cuda:0", "--num-envs", str(NUM_ENVS),
        "--iterations", str(ITERATIONS), "--seed", str(spec["train_seed"]),
        "--episode-seconds", str(config["episode_seconds"]),
        "--course-difficulty-min", str(config["difficulty_min"]),
        "--course-difficulty-max", str(config["difficulty_max"]),
        "--speed-min", str(config["speed_min"]), "--speed-max", str(config["speed_max"]),
        "--run-name", RUN_ID,
    ]


def validate_diagnostic_contract(config: dict, spec: dict, training_receipt: dict,
                                 checkpoint: Path) -> None:
    require(isinstance(config, dict) and isinstance(spec, dict),
            "V1330 diagnostic config and spec must be JSON objects")
    validate_train_contract(config, spec)
    require(config.get("run_authorized") is True and
            config.get("diagnostic_authorized") is True and
            spec.get("run_authorized") is True and
            spec.get("diagnostic_authorized") is True,
            "V1330 diagnostic evaluation is not explicitly authorized")
    require(config.get("automatic_gate_authorized") is False and
            config.get("automatic_retry_authorized") is False and
            config.get("automatic_follow_on_authorized") is False and
            spec.get("automatic_gate_authorized") is False and
            spec.get("automatic_retry_authorized") is False and
            spec.get("automatic_follow_on_authorized") is False,
            "V1330 diagnostic cannot authorize a gate, retry, or follow-on")
    _require_fresh_seed(spec.get("diagnostic_seed"), "diagnostic evaluation")
    require(spec["diagnostic_seed"] != spec.get("train_seed"),
            "V1330 diagnostic seed must be independent from training seed")
    _require_hash(spec.get("diagnostic_seed_audit_sha256"), "diagnostic_seed_audit_sha256")
    evaluation = config.get("diagnostic_evaluation")
    require(isinstance(evaluation, dict), "V1330 diagnostic evaluation contract missing")
    expected_eval = {
        "stage": "stage1", "terrain": "lp_acrl", "num_envs": NUM_ENVS,
        "episodes": EVALUATION_EPISODES, "episode_seconds": 200,
        "strict_first_episode": True, "certify": False,
        "run_id": DIAGNOSTIC_RUN_ID,
        "trace_path": spec.get("diagnostic_trace_path"),
        "log_path": spec.get("diagnostic_log_path"),
        "receipt_path": spec.get("diagnostic_receipt_path"),
        "report_path": spec.get("diagnostic_report_path"),
    }
    for name, expected in expected_eval.items():
        require(evaluation.get(name) == expected,
                f"V1330 diagnostic evaluation contract changed: {name}")
    require(training_receipt.get("status") == "trained-unverified" and
            training_receipt.get("run_id") == RUN_ID,
            "V1330 diagnostic requires the successful one-run training receipt")
    expected_checkpoint = checkpoint
    require(checkpoint.is_file() and not checkpoint.is_symlink() and
            training_receipt.get("checkpoint_path") == str(expected_checkpoint),
            "V1330 diagnostic checkpoint does not match the training receipt")
    _require_hash(training_receipt.get("checkpoint_sha256"), "training checkpoint sha256")
    require(digest(checkpoint) == training_receipt["checkpoint_sha256"],
            "V1330 checkpoint hash differs from the training receipt")
    for name in ("diagnostic_seed_audit_path", "diagnostic_trace_path",
                 "diagnostic_log_path", "diagnostic_receipt_path", "diagnostic_report_path"):
        require(isinstance(spec.get(name), str) and spec[name],
                f"missing V1330 diagnostic path: {name}")


def diagnostic_command(workspace: Path, config: dict, spec: dict,
                        checkpoint: Path, trace: Path,
                        training_receipt: dict) -> list[str]:
    expected_checkpoint = workspace / spec["checkpoint_path"]
    expected_trace = workspace / PACKAGE_REL / spec["diagnostic_trace_path"]
    require(checkpoint.resolve() == expected_checkpoint.resolve(),
            "V1330 evaluation checkpoint path changed")
    require(trace.resolve() == expected_trace.resolve(),
            "V1330 episode-trace path changed")
    validate_diagnostic_contract(config, spec, training_receipt, checkpoint)
    evaluation = config["diagnostic_evaluation"]
    return [
        str(workspace / "sim.sh"), "evaluate", "--stage", evaluation["stage"],
        "--terrain", evaluation["terrain"], "--headless", "--device", "cuda:0",
        "--num-envs", str(evaluation["num_envs"]),
        "--episodes", str(evaluation["episodes"]),
        "--seed", str(spec["diagnostic_seed"]),
        "--episode-seconds", str(evaluation["episode_seconds"]),
        "--course-difficulty-min", str(config["difficulty_min"]),
        "--course-difficulty-max", str(config["difficulty_max"]),
        "--speed-min", str(config["speed_min"]), "--speed-max", str(config["speed_max"]),
        "--checkpoint", str(checkpoint), "--episode-trace", str(trace),
        "--strict-first-episode",
    ]
