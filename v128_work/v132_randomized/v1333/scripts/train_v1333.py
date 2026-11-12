"""One-shot, scan-only V1333 training runner; never evaluates or retries."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

WORKSPACE = Path(os.environ.get("V1333_SIM_WORKSPACE", "/config/Desktop/sim")).resolve()
PACKAGE_REL = Path("v128_work/v132_randomized/v1333")
ROOT = WORKSPACE / PACKAGE_REL
LOCK_PATH = WORKSPACE / "reports" / "v7_sim.lock"
RUN_ID = "go1-v1333-gap-scan-only-s01-p1-d1"
MIN_SEED, MAX_SEED = 1_000_000_000, 2_147_483_647
OUTPUT_KEYS = ("training_log", "training_receipt", "authorization_record", "seed_audit",
               "stage_report", "checkpoint", "run_dir")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_train_plan(plan: dict) -> None:
    require(isinstance(plan, dict), "V1333 plan must be an object")
    require(plan.get("candidate") == "v1333-gap-scan-only" and plan.get("run_id") == RUN_ID,
            "V1333 plan candidate or run id mismatch")
    fixed = {"stage": "stage1", "terrain": "lp_acrl", "training_roster": "single:combined_part_entrance",
             "num_envs": 512, "iterations": 650, "episode_seconds": 200,
             "difficulty_min": 0.0, "difficulty_max": 0.6, "speed_min": 0.35, "speed_max": 0.8,
             "initialization": "random", "critic_warmup_iterations": 0,
             "actor_observation_dim": 1073, "action_dim": 12,
             "task_plan_path": "configs/task_plan.json",
             "reward_config_path": "configs/rewards_stage1_gaps_v1333.toml"}
    for name, value in fixed.items():
        require(plan.get(name) == value, f"V1333 plan field {name} must equal {value!r}")
    require(plan.get("seed_policy") == "fresh-unused-audited"
            and plan.get("seed_min") == MIN_SEED and plan.get("seed_max") == MAX_SEED,
            "V1333 requires a fresh, audited seed")
    for name in ("diagnostic_authorized", "evaluation_authorized", "holdout_authorized",
                 "gate_authorized", "retry_authorized", "automatic_follow_on_authorized",
                 "automatic_gate_authorized", "automatic_retry_authorized"):
        require(plan.get(name) is False, f"V1333 training plan must not authorize {name}")
    require(not any(name in plan for name in ("checkpoint", "initial_checkpoint", "initialize_from", "resume", "warm_start")),
            "V1333 scan-semantic change requires random initialization")
    outputs = plan.get("output_paths")
    require(isinstance(outputs, dict) and set(outputs) == set(OUTPUT_KEYS),
            "V1333 output path set must match the one-run contract")


def build_training_environment(base_environment: dict[str, str], execution_settings: dict[str, str],
                               workspace: Path, candidate_root: Path) -> dict[str, str]:
    workspace = Path(workspace).resolve()
    candidate_root = Path(candidate_root).resolve()
    require(candidate_root == workspace / PACKAGE_REL, "V1333 candidate root mismatch")
    require(execution_settings.get("GO1_PRIVILEGED_ACTOR") == "0",
            "V1333 scan-only training requires actor-visible observations")
    require(execution_settings.get("GO1_STAGE1_DIRECTION_ONLY") is None,
            "V1333 direction-only mode is forbidden")
    environment = {name: value for name, value in base_environment.items()
                   if not name.startswith("GO1_") and name != "V1333_TRAIN_AUTHORIZED"}
    for name, value in execution_settings.items():
        require(isinstance(name, str) and name.startswith("GO1_") and isinstance(value, str),
                "V1333 execution settings must be string GO1_* entries")
        if name.endswith("_PATH") and not Path(value).is_absolute():
            value = str((workspace / value).resolve())
        environment[name] = value
    environment.update({
        "GO1_SIM_WORKSPACE": str(workspace),
        "GO1_V1333_BASE_SOURCE": str(workspace / "source" / "go1_hier_sim"),
        "GO1_TASK_PLAN": str((candidate_root / "configs/task_plan.json").resolve()),
        "GO1_REWARD_CONFIG_PATH": str((candidate_root / "configs/rewards_stage1_gaps_v1333.toml").resolve()),
        "GO1_V1333_GAP_SENSITIVITY": "1",
        "GO1_V1333_GAP_POSTURE_RELIEF": "0",
        "GO1_PRIVILEGED_ACTOR": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    environment.pop("V1333_TRAIN_AUTHORIZED", None)
    return environment


def training_command(plan: dict, seed: int, workspace: Path, candidate_root: Path) -> list[str]:
    validate_train_plan(plan)
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED, "V1333 seed out of range")
    workspace, candidate_root = Path(workspace).resolve(), Path(candidate_root).resolve()
    require(candidate_root == workspace / PACKAGE_REL, "V1333 command candidate root mismatch")
    return ["bash", str(candidate_root / "scripts/sim_v1333.sh"), "train",
            "--stage", plan["stage"], "--terrain", plan["terrain"], "--headless",
            "--device", "cuda:0", "--num-envs", str(plan["num_envs"]),
            "--iterations", str(plan["iterations"]), "--seed", str(seed),
            "--episode-seconds", str(plan["episode_seconds"]),
            "--course-difficulty-min", str(plan["difficulty_min"]),
            "--course-difficulty-max", str(plan["difficulty_max"]),
            "--speed-min", str(plan["speed_min"]), "--speed-max", str(plan["speed_max"]),
            "--run-name", plan["run_id"]]


def require_one_shot(value: str | None) -> None:
    require(value == "one-shot", "V1333 training requires exact one-shot authorization")


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "required JSON missing or symlinked: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON root must be an object: " + str(path))
    return value


def safe_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "output path escapes workspace: " + relative)
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "path contains a symlink: " + str(cursor))
    resolved = (root / value).resolve(strict=False)
    require(resolved == root.resolve() or root.resolve() in resolved.parents,
            "path escapes its root: " + relative)
    return root / value


def _assert_lock_held() -> None:
    descriptor = Path(f"/proc/{os.getpid()}/fd/9")
    require(descriptor.exists(), "V1333 lock descriptor 9 is not inherited")
    require(Path(os.readlink(str(descriptor))).resolve() == LOCK_PATH.resolve(),
            "V1333 runner does not hold the required simulator lock")


def tree_fingerprint(root: Path) -> str:
    require(root.is_dir() and not root.is_symlink(), "fingerprint root missing or symlinked: " + str(root))
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        content_hash = hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else sha256(path)
        digest.update(rel + b"\0" + content_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def active_source_fingerprint(workspace: Path) -> str:
    return tree_fingerprint(Path(workspace) / "source")


def ensure_idle() -> None:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        cells = [part.strip() for part in line.split(",")]
        require(len(cells) == 2 and all(item.isdigit() for item in cells),
                "cannot parse GPU-idle snapshot")
        rows.append((int(cells[0]), int(cells[1])))
    require([index for index, _ in rows] == [0, 1, 2, 3], "V1333 expects four visible GPUs")
    require(all(memory <= 256 for _, memory in rows), "V1333 refuses non-idle GPU snapshot: " + repr(rows))
    ps = subprocess.run(["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True, check=True).stdout
    markers = ("isaaclab.sh", "/scripts/sim.py", "isaac-sim", "/kit", "omni.kit")
    active = [line.strip() for line in ps.splitlines() if any(m in line.lower() for m in markers)]
    require(not active, "V1333 refuses to overlap another Isaac process: " + " | ".join(active[:5]))


def _seed_roots(workspace: Path) -> list[Path]:
    return [workspace / name for name in ("reports", "configs", "v128_work", "source", "scripts")]


def _seed_seen(seed: int, roots: list[Path]) -> tuple[bool, int, int]:
    pattern = re.compile(r"(?<![0-9])" + re.escape(str(seed)) + r"(?![0-9])")
    text_suffixes = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    skipped = {".git", ".venv", "venv", "node_modules", "__pycache__", "checkpoints", "runs", "tensorboard"}
    scanned = byte_count = 0
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            if any(part in skipped for part in path.parts):
                continue
            data = path.read_bytes()
            scanned += 1
            byte_count += len(data)
            if pattern.search(path.name) or pattern.search(data.decode("utf-8", errors="replace")):
                return True, scanned, byte_count
    return False, scanned, byte_count


def select_fresh_seed(roots: list[Path]) -> tuple[int, int, int]:
    forbidden = {42, 2026, 2028, 2029, 1189662175, 1242088145, 1427789363,
                 1607361419, 1752494079, 1802284695, 1812341965}
    for _ in range(512):
        seed = secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED
        if seed in forbidden:
            continue
        seen, scanned, byte_count = _seed_seen(seed, roots)
        if not seen:
            return seed, scanned, byte_count
    raise ValueError("V1333 could not select a fresh seed after 512 attempts")


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def update_manifest(receipt_path: Path, receipt: dict) -> None:
    manifest_path = ROOT / "configs/manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "code-only-untrained", "V1333 manifest already finalized")
    manifest["status"] = receipt["status"]
    manifest["training_authorized"] = False
    manifest["run_authorized"] = False
    manifest["training_result"] = {
        "path": receipt_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(receipt_path), "run_id": receipt["run_id"]}
    tmp = manifest_path.with_name("manifest.json.tmp")
    require(not tmp.exists() and not tmp.is_symlink(), "V1333 manifest temp file already exists")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    tmp.replace(manifest_path)

def validate_bundle() -> tuple[dict, dict, dict]:
    require(ROOT == WORKSPACE / PACKAGE_REL and ROOT.is_dir() and not ROOT.is_symlink(),
            "V1333 runner requires its fixed isolated candidate root")
    manifest = read_json(ROOT / "configs/manifest.json")
    require(manifest.get("candidate") == "v1333-gap-scan-only"
            and manifest.get("status") == "code-only-untrained",
            "V1333 candidate is not code-only-untrained")
    plan = read_json(ROOT / "configs/train_plan_v1333.json")
    validate_train_plan(plan)
    execution = read_json(ROOT / "configs/candidate_execution.json")
    require(execution.get("candidate") == "v1333-gap-scan-only"
            and execution.get("training_authorized") is False
            and execution.get("run_authorized") is False,
            "V1333 execution config must remain non-authorizing")
    for name in ("diagnostic_authorized", "evaluation_authorized", "holdout_authorized",
                 "gate_authorized", "retry_authorized", "automatic_follow_on_authorized",
                 "automatic_gate_authorized", "automatic_retry_authorized"):
        require(execution.get(name) is False, "V1333 config may not authorize " + name)
    settings = execution.get("train_environment")
    require(isinstance(settings, dict)
            and settings.get("GO1_V1333_GAP_SENSITIVITY") == "1"
            and settings.get("GO1_V1333_GAP_POSTURE_RELIEF") == "0"
            and settings.get("GO1_PRIVILEGED_ACTOR") == "0",
            "V1333 run must be scan-only with actor-visible observations")
    profile = read_json(ROOT / "configs/observation_profile.json")
    require(profile.get("actor_observation_dim") == 1073 and profile.get("action_dim") == 12,
            "V1333 observation/action contract changed")
    task = read_json(ROOT / plan["task_plan_path"])
    families = task.get("families")
    require(isinstance(families, list) and len(families) == 512
            and set(families) == {"combined_part_entrance"},
            "V1333 task plan is not the 512-environment entrance roster")
    for name in ("scripts/sim.py", "sim.sh", "configs/ppo_gaps_long.json"):
        path = WORKSPACE / name
        require(path.is_file() and not path.is_symlink(), "required simulator input missing: " + str(path))
    return manifest, plan, execution


def _output_paths(plan: dict) -> dict[str, Path]:
    return {key: safe_path(WORKSPACE, relative) for key, relative in plan["output_paths"].items()}


def _require_outputs_absent(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        require(not path.exists() and not path.is_symlink(),
                f"V1333 one-shot output already exists ({key}): {path}")


def _validate_outputs(paths: dict[str, Path], plan: dict) -> dict:
    log_path, report_path = paths["training_log"], paths["stage_report"]
    checkpoint, run_dir = paths["checkpoint"], paths["run_dir"]
    require(log_path.is_file() and not log_path.is_symlink(), "V1333 training log missing")
    require(report_path.is_file() and not report_path.is_symlink(), "V1333 stage report missing")
    require(checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0,
            "V1333 final checkpoint missing or empty")
    require(run_dir.is_dir() and not run_dir.is_symlink(), "V1333 training run directory missing")
    report = read_json(report_path)
    task_path = (ROOT / plan["task_plan_path"]).resolve()
    require(report.get("run") == plan["run_id"] and report.get("iterations") == plan["iterations"]
            and report.get("task_plan") == str(task_path)
            and type(report.get("episodes")) is int and report["episodes"] > 0,
            "V1333 stage report does not match requested training run")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require("TRAINING FINISHED;" in log and f"STAGE REPORT: {report_path}" in log,
            "V1333 training log lacks positive completion markers")
    return {"checkpoint_path":str(checkpoint), "checkpoint_sha256":sha256(checkpoint),
            "stage_report_path":str(report_path), "stage_report_sha256":sha256(report_path),
            "training_log_path":str(log_path), "training_log_sha256":sha256(log_path),
            "training_episodes":report["episodes"], "completion_rate":report.get("completion_rate"),
            "survival_rate":report.get("survival_rate"),
            "mean_command_speed":report.get("mean_command_speed"),
            "mean_achieved_speed":report.get("mean_achieved_speed")}

def run_training() -> dict:
    require_one_shot(os.environ.get("V1333_TRAIN_AUTHORIZED"))
    _assert_lock_held()
    manifest, plan, execution = validate_bundle()
    paths = _output_paths(plan)
    _require_outputs_absent(paths)
    ensure_idle()
    active_before = active_source_fingerprint(WORKSPACE)
    candidate_before = tree_fingerprint(ROOT / "source/go1_hier_sim")
    seed, files_scanned, bytes_scanned = select_fresh_seed(_seed_roots(WORKSPACE))
    seed_path = paths["seed_audit"]
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("x", encoding="utf-8") as stream:
        stream.write("V1333_SEED_AUDIT_V1\nPURPOSE training\n")
        stream.write(f"SEED {seed}\nAUDIT_UTC {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"FILES_SCANNED {files_scanned}\nBYTES_SCANNED {bytes_scanned}\nHITS []\n")
        stream.write("RESULT UNUSED_IN_AUDITED_TEXT\n")
    plan_path = ROOT / "configs/train_plan_v1333.json"
    manifest_path = ROOT / "configs/manifest.json"
    auth_path = paths["authorization_record"]
    authorization = {
        "status":"consumed-one-shot", "authorization_scope":"one V1333 scan-only training run",
        "user_authorized":True, "authorized_at_utc":datetime.now(timezone.utc).isoformat(),
        "run_id":plan["run_id"], "train_seed":seed,
        "manifest_sha256":sha256(manifest_path), "training_plan_sha256":sha256(plan_path),
        "seed_audit_path":str(seed_path), "seed_audit_sha256":sha256(seed_path),
        "active_source_fingerprint_before":active_before,
        "candidate_source_fingerprint_before":candidate_before,
        "actor_observation":"ray-visible; GO1_PRIVILEGED_ACTOR=0",
        "GO1_V1333_GAP_SENSITIVITY":"1", "GO1_V1333_GAP_POSTURE_RELIEF":"0",
        "diagnostic_authorized":False, "evaluation_authorized":False,
        "holdout_authorized":False, "gate_authorized":False, "retry_authorized":False,
        "automatic_follow_on_authorized":False,
    }
    write_json_exclusive(auth_path, authorization)
    environment = build_training_environment(os.environ.copy(), execution["train_environment"],
                                            WORKSPACE, ROOT)
    environment["GO1_RUN_ID"] = plan["run_id"]
    environment["V1333_TRAIN_CHILD"] = "1"
    command = training_command(plan, seed, WORKSPACE, ROOT)
    log_path = paths["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_attempted = False
    try:
        with log_path.open("x", encoding="utf-8") as log_stream:
            launch_attempted = True
            print(f"V1333_TRAINING_STARTED run_id={plan['run_id']} seed={seed}", flush=True)
            result = subprocess.run(command, cwd=str(WORKSPACE), env=environment,
                                    stdout=log_stream, stderr=subprocess.STDOUT,
                                    text=True, check=False, pass_fds=(9,))
        require(result.returncode == 0, "V1333 training process exited nonzero: " + str(result.returncode))
        outputs = _validate_outputs(paths, plan)
        active_after = active_source_fingerprint(WORKSPACE)
        candidate_after = tree_fingerprint(ROOT / "source/go1_hier_sim")
        require(active_after == active_before, "active simulator source changed during V1333 training")
        require(candidate_after == candidate_before, "V1333 candidate source changed during training")
        ensure_idle()
        receipt = {
            "status":"trained-unverified", "mode":"train", "run_id":plan["run_id"],
            "train_seed":seed, "seed_audit_path":str(seed_path),
            "seed_audit_sha256":sha256(seed_path), "num_envs":plan["num_envs"],
            "iterations":plan["iterations"], "terrain":plan["terrain"],
            "initialization":"random", "actor_observation_dim":plan["actor_observation_dim"],
            "action_dim":plan["action_dim"], "actor_observation":"ray-visible; privileged actor disabled",
            "scan_sensitivity_enabled":True, "gap_posture_relief":0.0,
            "active_source_fingerprint_before":active_before,
            "active_source_fingerprint_after":active_after,
            "candidate_source_fingerprint_before":candidate_before,
            "candidate_source_fingerprint_after":candidate_after,
            "authorization_record":str(auth_path), "diagnostic_ran":False,
            "evaluation_ran":False, "holdout_ran":False, "formal_gate":False,
            "automatic_retry_authorized":False, "automatic_follow_on_authorized":False,
            **outputs,
        }
        receipt_path = paths["training_receipt"]
        write_json_exclusive(receipt_path, receipt)
        update_manifest(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt_path = paths["training_receipt"]
        if launch_attempted and not receipt_path.exists():
            try:
                active_after = active_source_fingerprint(WORKSPACE)
            except BaseException:
                active_after = None
            failure = {
                "status":"training-failed", "mode":"train", "run_id":plan["run_id"],
                "train_seed":seed, "error":f"{type(error).__name__}: {error}",
                "launch_attempted":True, "authorization_record":str(auth_path),
                "active_source_fingerprint_before":active_before,
                "active_source_fingerprint_after":active_after,
                "automatic_retry_authorized":False, "automatic_follow_on_authorized":False,
            }
            try:
                write_json_exclusive(receipt_path, failure)
                update_manifest(receipt_path, failure)
            except BaseException as receipt_error:
                print(f"V1333_FAILURE_RECEIPT_BLOCKED: {receipt_error}", file=sys.stderr)
        raise


def main() -> int:
    if sys.argv[1:] != ["train"]:
        print("V1333 supports only one-shot training", file=sys.stderr)
        return 2
    try:
        receipt = run_training()
    except BaseException as error:
        print(f"V1333_TRAINING_BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status":receipt["status"], "run_id":receipt["run_id"],
                      "train_seed":receipt["train_seed"],
                      "checkpoint_sha256":receipt["checkpoint_sha256"],
                      "training_log_sha256":receipt["training_log_sha256"],
                      "diagnostic_ran":False, "formal_gate":False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
