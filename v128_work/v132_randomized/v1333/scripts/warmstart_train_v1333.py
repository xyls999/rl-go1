"""One-shot V1333 reward pilot: policy-only initialization, fresh optimizer."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from train_v1333 import (
    WORKSPACE, ROOT, PACKAGE_REL, MIN_SEED, MAX_SEED, _assert_lock_held,
    active_source_fingerprint, tree_fingerprint, ensure_idle, safe_path, sha256,
    read_json, write_json_exclusive, build_training_environment,
    select_fresh_seed, _seed_roots,
)

RUN_ID = "go1-v1333-gap-scan-fc015-warmstart200-a02"
PARENT_RUN_ID = "go1-v1333-gap-scan-only-s01-p1-d1-continue650"
INITIAL_CHECKPOINT = Path("/config/Desktop/sim/checkpoints/stage1/go1-v1333-gap-scan-only-s01-p1-d1-continue650/model_1298.pt")
INITIAL_CHECKPOINT_SHA256 = "15351c88f106f1dfc1d9226f9cdd8284050b63218f3b20369c3337ecb177d7ad"
PLAN_PATH = ROOT / "configs/reward_pilot_fc015_warmstart_plan_v1333.json"
PARENT_RECEIPT = ROOT / "reports/continuation_result_v1333_c650.json"
SOURCE_POLICY = "pin-parent-and-current-fingerprints; mismatch-is-recorded"
INPUT_KEYS = ("parent_receipt", "parent_candidate_source", "active_source",
              "candidate_source", "reward_config", "task_plan", "candidate_execution",
              "observation_profile", "ppo_config", "runner", "train_runner",
              "sim_wrapper", "simulator")
OUTPUT_PATHS = {
    "training_log": "v128_work/v132_randomized/v1333/reports/go1-v1333-gap-scan-fc015-warmstart200-a02_full_train.log",
    "pilot_receipt": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_result_fc015_a02.json",
    "authorization_record": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_authorization_fc015_a02.json",
    "seed_audit": "v128_work/v132_randomized/v1333/reports/reward_pilot_warmstart_seed_fc015_a02.txt",
    "stage_report": "reports/stage_go1-v1333-gap-scan-fc015-warmstart200-a02.json",
    "checkpoint": "checkpoints/stage1/go1-v1333-gap-scan-fc015-warmstart200-a02/model_199.pt",
    "run_dir": "runs/stage1/go1-v1333-gap-scan-fc015-warmstart200-a02",
}
OUTPUT_KEYS = tuple(OUTPUT_PATHS)
PLAN_FIXED = {
    "schema_version": 1, "candidate": "v1333-gap-scan-only", "run_id": RUN_ID,
    "parent_run_id": PARENT_RUN_ID, "stage": "stage1", "terrain": "lp_acrl",
    "training_roster": "single:combined_part_entrance", "num_envs": 512,
    "episode_seconds": 200, "difficulty_min": 0.0, "difficulty_max": 0.6,
    "speed_min": 0.35, "speed_max": 0.8,
    "initialize_from": str(INITIAL_CHECKPOINT),
    "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
    "initialization": "policy-weights-only-reset-optimizer",
    "iterations": 200, "final_checkpoint_iteration": 199,
    "critic_warmup_iterations": 0, "optimizer_state_restored": False,
    "training_authorized": False,
    "task_plan_path": "configs/task_plan.json",
    "reward_config_path": "configs/rewards_stage1_gaps_v1333.toml",
    "source_policy": SOURCE_POLICY, "seed_policy": "fresh-unused-audited",
}
FLAG_KEYS = ("diagnostic_authorized", "evaluation_authorized", "holdout_authorized",
             "gate_authorized", "retry_authorized", "automatic_follow_on_authorized",
             "automatic_gate_authorized", "automatic_retry_authorized")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_warmstart_plan(plan: dict) -> None:
    require(isinstance(plan, dict), "V1333 warm-start plan must be an object")
    for name, value in PLAN_FIXED.items():
        message = ("optimizer state must be reset" if name == "optimizer_state_restored"
                   else f"V1333 warm-start field {name} mismatch")
        require(plan.get(name) == value, message)
    for name in FLAG_KEYS:
        require(plan.get(name) is False, f"V1333 warm-start must not authorize {name}")
    fingerprints = plan.get("input_fingerprints")
    require(isinstance(fingerprints, dict) and set(fingerprints) == set(INPUT_KEYS),
            "V1333 warm-start input fingerprint set mismatch")
    for name, value in fingerprints.items():
        require(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                f"V1333 warm-start fingerprint {name} is malformed")
    require(fingerprints["parent_candidate_source"] != fingerprints["candidate_source"],
            "V1333 warm-start plan must record source drift")
    require(plan.get("output_paths") == OUTPUT_PATHS,
            "V1333 warm-start output paths are not pinned to the isolated run")
    expected_keys = set(PLAN_FIXED) | set(FLAG_KEYS) | {"input_fingerprints", "output_paths"}
    require(set(plan) == expected_keys, "V1333 warm-start plan contains unknown fields")


def warmstart_command(plan: dict, seed: int, workspace: Path, candidate_root: Path) -> list[str]:
    require_warmstart_plan(plan)
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED,
            "V1333 warm-start seed out of range")
    workspace, candidate_root = Path(workspace).resolve(), Path(candidate_root).resolve()
    require(candidate_root == workspace / PACKAGE_REL, "V1333 warm-start candidate root mismatch")
    return ["bash", str(candidate_root / "scripts/sim_v1333.sh"), "train",
            "--stage", plan["stage"], "--terrain", plan["terrain"], "--headless",
            "--device", "cuda:0", "--num-envs", str(plan["num_envs"]),
            "--iterations", str(plan["iterations"]), "--seed", str(seed),
            "--episode-seconds", str(plan["episode_seconds"]),
            "--course-difficulty-min", str(plan["difficulty_min"]),
            "--course-difficulty-max", str(plan["difficulty_max"]),
            "--speed-min", str(plan["speed_min"]), "--speed-max", str(plan["speed_max"]),
            "--run-name", plan["run_id"], "--initialize-from", plan["initialize_from"],
            "--critic-warmup-iterations", str(plan["critic_warmup_iterations"])]


def _actual_fingerprints(plan: dict, parent: dict) -> dict[str, str]:
    return {
        "parent_receipt": sha256(PARENT_RECEIPT),
        "parent_candidate_source": parent.get("candidate_source_fingerprint_after", ""),
        "active_source": active_source_fingerprint(WORKSPACE),
        "candidate_source": tree_fingerprint(ROOT / "source/go1_hier_sim"),
        "reward_config": sha256(ROOT / plan["reward_config_path"]),
        "task_plan": sha256(ROOT / plan["task_plan_path"]),
        "candidate_execution": sha256(ROOT / "configs/candidate_execution.json"),
        "observation_profile": sha256(ROOT / "configs/observation_profile.json"),
        "ppo_config": sha256(WORKSPACE / "configs/ppo_gaps_long.json"),
        "runner": sha256(Path(__file__).resolve()),
        "train_runner": sha256(ROOT / "scripts/train_v1333.py"),
        "sim_wrapper": sha256(ROOT / "scripts/sim_v1333.sh"),
        "simulator": sha256(WORKSPACE / "scripts/sim.py"),
    }


def validate_warmstart_bundle() -> tuple[dict, dict, dict]:
    require(ROOT == WORKSPACE / PACKAGE_REL and ROOT.is_dir() and not ROOT.is_symlink(),
            "V1333 warm-start candidate root mismatch")
    plan = read_json(PLAN_PATH)
    require_warmstart_plan(plan)
    parent = read_json(PARENT_RECEIPT)
    require(parent.get("status") == "trained-unverified"
            and parent.get("run_id") == PARENT_RUN_ID
            and parent.get("mode") == "full-checkpoint-resume"
            and parent.get("checkpoint_path") == str(INITIAL_CHECKPOINT)
            and parent.get("checkpoint_sha256") == INITIAL_CHECKPOINT_SHA256
            and parent.get("optimizer_state_restored") is True,
            "V1333 warm-start parent receipt/checkpoint mismatch")
    actual = _actual_fingerprints(plan, parent)
    require(actual == plan["input_fingerprints"],
            "V1333 warm-start inputs changed; expected/actual fingerprints differ")
    require(actual["active_source"] == parent.get("active_source_fingerprint_after"),
            "V1333 active source no longer matches the parent training")
    require(actual["parent_candidate_source"] == parent.get("candidate_source_fingerprint_after"),
            "V1333 warm-start parent source is not attested")
    require(actual["candidate_source"] != actual["parent_candidate_source"],
            "V1333 warm-start source drift must remain explicitly recorded")
    execution = read_json(ROOT / "configs/candidate_execution.json")
    settings = execution.get("train_environment")
    require(execution.get("candidate") == "v1333-gap-scan-only"
            and execution.get("training_authorized") is False
            and execution.get("run_authorized") is False
            and isinstance(settings, dict)
            and settings.get("GO1_V1333_GAP_SENSITIVITY") == "1"
            and settings.get("GO1_V1333_GAP_POSTURE_RELIEF") == "0"
            and settings.get("GO1_PRIVILEGED_ACTOR") == "0",
            "V1333 warm-start must preserve scan-only actor-visible settings")
    profile = read_json(ROOT / "configs/observation_profile.json")
    require(profile.get("actor_observation_dim") == 1073 and profile.get("action_dim") == 12,
            "V1333 warm-start observation/action contract changed")
    task = read_json(ROOT / plan["task_plan_path"])
    require(isinstance(task.get("families"), list) and len(task["families"]) == 512
            and set(task["families"]) == {"combined_part_entrance"}
            and task.get("target_level") == 1,
            "V1333 warm-start task plan changed")
    package_path = str(ROOT / "source/go1_hier_sim")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from go1_hier_sim.checkpoints import load_checkpoint
    from go1_hier_sim.reward_config import load_stage1_rewards
    checkpoint = load_checkpoint(str(INITIAL_CHECKPOINT), expected_stage="stage1")
    metadata = checkpoint.get("metadata", {})
    require(metadata.get("iteration") == 1298 and metadata.get("policy_variant") == "map_attn_v3",
            "V1333 warm-start source checkpoint metadata mismatch")
    require("policy" in checkpoint.get("models", {}),
            "V1333 warm-start source checkpoint has no policy weights")
    previous_rewards = metadata.get("reward_config")
    current_rewards = load_stage1_rewards(ROOT / plan["reward_config_path"])
    require(isinstance(previous_rewards, dict) and set(previous_rewards) == set(current_rewards),
            "V1333 warm-start reward contract mismatch")
    differences = []
    for term in current_rewards:
        require(set(previous_rewards[term]) == set(current_rewards[term]),
                "V1333 warm-start reward fields changed for " + term)
        for field, value in current_rewards[term].items():
            old = previous_rewards[term][field]
            if old != value:
                differences.append((term, field, old, value))
    require(differences == [("flight_clearance", "weight", 0.0, 0.15)],
            "V1333 warm-start may change only flight_clearance.weight to 0.15")
    for name in ("scripts/sim.py", "configs/ppo_gaps_long.json"):
        path = WORKSPACE / name
        require(path.is_file() and not path.is_symlink(), "required simulator input missing: " + str(path))
    for name in ("scripts/sim_v1333.sh", "scripts/train_v1333.py"):
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "required V1333 input missing: " + str(path))
    outputs = {key: safe_path(WORKSPACE, value) for key, value in plan["output_paths"].items()}
    require(len(set(outputs.values())) == len(outputs), "V1333 warm-start output paths overlap")
    return plan, execution, parent


def _outputs_absent(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        require(not path.exists() and not path.is_symlink(),
                f"V1333 warm-start output already exists ({key}): {path}")


def _validate_outputs(paths: dict[str, Path], plan: dict) -> dict:
    log_path, report_path = paths["training_log"], paths["stage_report"]
    checkpoint, run_dir = paths["checkpoint"], paths["run_dir"]
    require(log_path.is_file() and not log_path.is_symlink(), "V1333 warm-start log missing")
    require(report_path.is_file() and not report_path.is_symlink(), "V1333 warm-start stage report missing")
    require(checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0,
            "V1333 warm-start final checkpoint missing or empty")
    require(run_dir.is_dir() and not run_dir.is_symlink(), "V1333 warm-start run directory missing")
    report = read_json(report_path)
    task_path = str((ROOT / plan["task_plan_path"]).resolve())
    require(report.get("run") == plan["run_id"] and report.get("iterations") == plan["iterations"]
            and report.get("task_plan") == task_path
            and type(report.get("episodes")) is int and report["episodes"] > 0,
            "V1333 warm-start stage report does not match requested run")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    require("TRAINING FINISHED;" in text and f"STAGE REPORT: {report_path}" in text,
            "V1333 warm-start log lacks completion markers")
    package_path = str(ROOT / "source/go1_hier_sim")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from go1_hier_sim.checkpoints import load_checkpoint
    final_checkpoint = load_checkpoint(str(checkpoint), expected_stage="stage1")
    metadata = final_checkpoint.get("metadata", {})
    require(metadata.get("iteration") == plan["final_checkpoint_iteration"],
            "V1333 warm-start checkpoint iteration mismatch")
    require(metadata.get("upstream", {}).get("initialized_from") == str(INITIAL_CHECKPOINT),
            "V1333 warm-start checkpoint does not record the initialized policy")
    return {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "stage_report_path": str(report_path), "stage_report_sha256": sha256(report_path),
        "training_log_path": str(log_path), "training_log_sha256": sha256(log_path),
        "training_episodes": report["episodes"],
        "completion_rate": report.get("completion_rate"),
        "survival_rate": report.get("survival_rate"),
        "mean_command_speed": report.get("mean_command_speed"),
        "mean_achieved_speed": report.get("mean_achieved_speed"),
    }


def run_warmstart_pilot() -> dict:
    require(os.environ.get("V1333_FC015_WARMSTART_AUTHORIZED") == "one-shot",
            "V1333 warm-start pilot requires exact one-shot authorization")
    _assert_lock_held()
    plan, execution, parent = validate_warmstart_bundle()
    paths = {key: safe_path(WORKSPACE, value) for key, value in plan["output_paths"].items()}
    _outputs_absent(paths)
    ensure_idle()
    active_before = active_source_fingerprint(WORKSPACE)
    candidate_before = tree_fingerprint(ROOT / "source/go1_hier_sim")
    seed, files_scanned, bytes_scanned = select_fresh_seed(_seed_roots(WORKSPACE))
    seed_path = paths["seed_audit"]
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("x", encoding="utf-8") as stream:
        stream.write("V1333_WARMSTART_SEED_AUDIT_V1\nPURPOSE reward-warmstart-training\n")
        stream.write(f"SEED {seed}\nAUDIT_UTC {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"FILES_SCANNED {files_scanned}\nBYTES_SCANNED {bytes_scanned}\nHITS []\n")
        stream.write("RESULT UNUSED_IN_AUDITED_TEXT\n")
    authorization_path = paths["authorization_record"]
    authorization = {
        "status": "consumed-one-shot", "authorization_scope": "one V1333 flight-clearance policy-only warm-start; 200 iterations",
        "user_authorized": True, "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": plan["run_id"], "parent_run_id": plan["parent_run_id"],
        "train_seed": seed, "initialize_from": str(INITIAL_CHECKPOINT),
        "initialize_from_sha256": sha256(INITIAL_CHECKPOINT),
        "optimizer_state_restored": False, "optimizer_reset": True,
        "parent_receipt_sha256": sha256(PARENT_RECEIPT),
        "pilot_plan_sha256": sha256(PLAN_PATH),
        "reward_config_sha256": plan["input_fingerprints"]["reward_config"],
        "parent_candidate_source_fingerprint": plan["input_fingerprints"]["parent_candidate_source"],
        "current_candidate_source_fingerprint": plan["input_fingerprints"]["candidate_source"],
        "input_fingerprints": plan["input_fingerprints"],
        "reward_change_only": {"flight_clearance.weight": [0.0, 0.15]},
        "seed_audit_path": str(seed_path), "seed_audit_sha256": sha256(seed_path),
        "active_source_fingerprint_before": active_before,
        "candidate_source_fingerprint_before": candidate_before,
        "GO1_PRIVILEGED_ACTOR": "0", "GO1_V1333_GAP_SENSITIVITY": "1",
        "GO1_V1333_GAP_POSTURE_RELIEF": "0",
        "diagnostic_authorized": False, "evaluation_authorized": False,
        "holdout_authorized": False, "gate_authorized": False, "retry_authorized": False,
        "automatic_follow_on_authorized": False,
    }
    write_json_exclusive(authorization_path, authorization)
    environment = build_training_environment(os.environ.copy(), execution["train_environment"],
                                            WORKSPACE, ROOT)
    environment.pop("V1333_FC015_WARMSTART_AUTHORIZED", None)
    environment.pop("V1333_REWARD_PILOT_AUTHORIZED", None)
    environment["GO1_RUN_ID"] = plan["run_id"]
    environment["V1333_TRAIN_CHILD"] = "1"
    command = warmstart_command(plan, seed, WORKSPACE, ROOT)
    log_path = paths["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launched = False
    try:
        with log_path.open("x", encoding="utf-8") as log_stream:
            launched = True
            print(f"V1333_WARMSTART_PILOT_STARTED run_id={plan['run_id']} seed={seed}", flush=True)
            result = subprocess.run(command, cwd=str(WORKSPACE), env=environment,
                                    stdout=log_stream, stderr=subprocess.STDOUT,
                                    text=True, check=False, pass_fds=(9,))
        require(result.returncode == 0,
                "V1333 warm-start pilot process exited nonzero: " + str(result.returncode))
        outputs = _validate_outputs(paths, plan)
        active_after = active_source_fingerprint(WORKSPACE)
        candidate_after = tree_fingerprint(ROOT / "source/go1_hier_sim")
        require(active_after == active_before, "active source changed during V1333 warm-start pilot")
        require(candidate_after == candidate_before, "V1333 candidate source changed during warm-start pilot")
        current_plan, _, _ = validate_warmstart_bundle()
        require(current_plan == plan, "V1333 warm-start inputs changed during training")
        ensure_idle()
        receipt = {
            "status": "trained-unverified", "mode": "policy-weights-only-warm-start",
            "run_id": plan["run_id"], "parent_run_id": plan["parent_run_id"],
            "train_seed": seed, "num_envs": plan["num_envs"], "iterations": plan["iterations"],
            "initialize_from": str(INITIAL_CHECKPOINT),
            "initialize_from_sha256": sha256(INITIAL_CHECKPOINT),
            "optimizer_state_restored": False, "optimizer_reset": True,
            "pilot_plan_sha256": sha256(PLAN_PATH),
            "reward_config_sha256": plan["input_fingerprints"]["reward_config"],
            "reward_change_only": {"flight_clearance.weight": [0.0, 0.15]},
            "actor_observation_dim": 1073, "action_dim": 12,
            "actor_observation": "ray-visible; privileged actor disabled",
            "scan_sensitivity_enabled": True, "gap_posture_relief": 0.0,
            "parent_candidate_source_fingerprint": plan["input_fingerprints"]["parent_candidate_source"],
            "candidate_source_fingerprint_before": candidate_before,
            "candidate_source_fingerprint_after": candidate_after,
            "active_source_fingerprint_before": active_before,
            "active_source_fingerprint_after": active_after,
            "authorization_record": str(authorization_path),
            "diagnostic_ran": False, "evaluation_ran": False, "holdout_ran": False,
            "formal_gate": False, "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False, **outputs,
        }
        write_json_exclusive(paths["pilot_receipt"], receipt)
        return receipt
    except BaseException as error:
        receipt_path = paths["pilot_receipt"]
        if launched and not receipt_path.exists():
            failure = {
                "status": "warmstart-pilot-failed", "run_id": plan["run_id"],
                "parent_run_id": plan["parent_run_id"], "train_seed": seed,
                "initialize_from": str(INITIAL_CHECKPOINT),
                "initialize_from_sha256": INITIAL_CHECKPOINT_SHA256,
                "optimizer_state_restored": False,
                "error": f"{type(error).__name__}: {error}", "launch_attempted": True,
                "authorization_record": str(authorization_path),
                "automatic_retry_authorized": False, "automatic_follow_on_authorized": False,
            }
            try:
                write_json_exclusive(receipt_path, failure)
            except BaseException as receipt_error:
                print(f"V1333_WARMSTART_FAILURE_RECEIPT_BLOCKED: {receipt_error}", file=sys.stderr)
        raise


def main() -> int:
    if sys.argv[1:] != ["warmstart"]:
        print("V1333 warm-start runner accepts only one-shot training", file=sys.stderr)
        return 2
    try:
        receipt = run_warmstart_pilot()
    except BaseException as error:
        print(f"V1333_WARMSTART_BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": receipt["status"], "run_id": receipt["run_id"],
                      "train_seed": receipt["train_seed"],
                      "checkpoint_sha256": receipt["checkpoint_sha256"],
                      "completion_rate": receipt.get("completion_rate"),
                      "survival_rate": receipt.get("survival_rate"),
                      "evaluation_ran": False, "formal_gate": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
