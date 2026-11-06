"""Pinned inputs, remote idle checks, and reversible source overlay for V1330."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

from contracts_v1330 import (
    DIAGNOSTIC_RUN_ID, EXPECTED, FORBIDDEN_SEEDS, ITERATIONS, MAX_SEED, MIN_SEED,
    NUM_ENVS, PACKAGE_REL, RUN_ID, EVALUATION_EPISODES, _require_hash,
    require, validate_diagnostic_contract, validate_train_contract,
)
from seed_audit_v1330 import scan_seed

WORKSPACE = Path("/config/Desktop/sim")
IDLE_GPU_MIB_MAX = 256
EXPECTED_SOURCE_FILES = frozenset({
    "go1_hier_sim/go1_hier_sim/cli.py",
    "go1_hier_sim/go1_hier_sim/envs/config.py",
    "go1_hier_sim/go1_hier_sim/envs/edge_contact_v1323.py",
    "go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py",
    "go1_hier_sim/go1_hier_sim/envs/episode_trace_v12.py",
    "go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py",
    "go1_hier_sim/go1_hier_sim/envs/gap_score_v1321.py",
    "go1_hier_sim/go1_hier_sim/envs/geometry.py",
    "go1_hier_sim/go1_hier_sim/envs/jump_airtime_v1325.py",
    "go1_hier_sim/go1_hier_sim/envs/landing_stability_v131.py",
    "go1_hier_sim/go1_hier_sim/envs/runtime.py",
    "go1_hier_sim/go1_hier_sim/envs/speed_autonomy_v1324.py",
    "go1_hier_sim/go1_hier_sim/reward_config.py",
    "go1_hier_sim/go1_hier_sim/strict_gate_v1322.py",
})
EXPECTED_RUNNER_FILES = frozenset({
    "scripts/contracts_v1330.py", "scripts/seed_audit_v1330.py",
    "scripts/preflight_v1330.py", "scripts/production_v1330.py",
    "scripts/verify_and_run_v1330.sh",
})
EXPECTED_LAUNCHERS = frozenset({"sim.sh", "scripts/sim.py"})
EXPECTED_INPUTS = frozenset({"configs/ppo_gaps_long.json"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    parsed = Path(relative)
    require(not parsed.is_absolute() and ".." not in parsed.parts and parsed.parts,
            "unsafe relative path: " + str(relative))
    target = root / parsed
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    require(resolved_target == resolved_root or resolved_root in resolved_target.parents,
            "path escapes approved root: " + str(relative))
    current = root
    for part in parsed.parts:
        current = current / part
        require(not current.is_symlink(), "symlinked path component: " + str(current))
    return target


def verified_file(path: Path, expected: str, label: str) -> None:
    _require_hash(expected, label + " sha256")
    require(path.is_file() and not path.is_symlink() and digest(path) == expected,
            f"{label} missing, symlinked, or SHA mismatch: {path}")


def lock_fd_matches(fd: int, lock_path: Path) -> bool:
    try:
        if lock_path.is_symlink() or not lock_path.is_file():
            return False
        opened = os.fstat(fd)
        on_disk = lock_path.stat()
        return (stat.S_ISREG(opened.st_mode) and stat.S_ISREG(on_disk.st_mode) and
                opened.st_dev == on_disk.st_dev and opened.st_ino == on_disk.st_ino)
    except OSError:
        return False


def _check_no_symlink_parents(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    current = path
    while current != resolved_root:
        require(not current.is_symlink(), "symlinked path component: " + str(current))
        current = current.parent
        resolved = current.resolve(strict=False)
        require(resolved_root == resolved or resolved_root in resolved.parents,
                "path escapes workspace: " + str(path))


def check_targets(rows: list[tuple[Path, Path, str | None, str]], workspace: Path) -> None:
    seen: set[Path] = set()
    for candidate, live, baseline_sha, candidate_sha in rows:
        resolved_live = live.resolve(strict=False)
        require(resolved_live not in seen and workspace.resolve() in resolved_live.parents and
                candidate.resolve(strict=False) != resolved_live,
                "unsafe or duplicate V1330 overlay target: " + str(live))
        seen.add(resolved_live)
        _check_no_symlink_parents(candidate, workspace)
        _check_no_symlink_parents(live, workspace)
        verified_file(candidate, candidate_sha, "V1330 source candidate")
        if baseline_sha is None:
            require(not live.exists() and not live.is_symlink(),
                    "unexpected live helper exists: " + str(live))
        else:
            verified_file(live, baseline_sha, "live-source baseline")


def atomic_copy(source: Path, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v1330-", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        require(digest(temporary) == expected_sha,
                "V1330 temporary source copy hash mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class Overlay:
    """Install only the pinned V1330 source files and restore exact baselines."""

    def __init__(self, rows: list[tuple[Path, Path, str | None, str]], workspace: Path):
        self.rows = rows
        self.workspace = workspace
        self.backup: Path | None = None
        self.installed: list[tuple[Path, str | None, str]] = []
        self.restored = False
        self.restoration: list[dict] = []

    def __enter__(self):
        check_targets(self.rows, self.workspace)
        reports = self.workspace / "reports"
        require(reports.is_dir() and not reports.is_symlink(),
                "remote reports directory missing/symlinked")
        require(not list(reports.glob("v1330-overlay-*")),
                "incomplete V1330 source overlay backup exists")
        self.backup = Path(tempfile.mkdtemp(prefix="v1330-overlay-", dir=reports))
        try:
            for index, (_candidate, live, baseline_sha, _candidate_sha) in enumerate(self.rows):
                if baseline_sha is not None:
                    saved = self.backup / str(index)
                    shutil.copy2(live, saved)
                    require(digest(saved) == baseline_sha,
                            "V1330 overlay backup hash mismatch")
            for candidate, live, baseline_sha, candidate_sha in self.rows:
                self.installed.append((live, baseline_sha, candidate_sha))
                atomic_copy(candidate, live, candidate_sha)
                require(digest(live) == candidate_sha,
                        "installed V1330 source hash mismatch")
        except BaseException:
            self.__exit__(*__import__("sys").exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        issues = []
        records = []
        for index, (live, baseline_sha, candidate_sha) in reversed(list(enumerate(self.installed))):
            try:
                if baseline_sha is None:
                    if live.exists():
                        require(digest(live) == candidate_sha,
                                "modified V1330 new helper cannot be removed: " + str(live))
                        live.unlink()
                    require(not live.exists(), "V1330 new helper remains: " + str(live))
                    records.append({"path": str(live.relative_to(self.workspace)),
                                    "expected_baseline_sha256": None,
                                    "restored_sha256": None, "new_helper_absent": True})
                else:
                    require(self.backup is not None, "V1330 overlay backup missing")
                    saved = self.backup / str(index)
                    require(saved.is_file() and digest(saved) == baseline_sha,
                            "V1330 overlay backup missing or changed: " + str(live))
                    atomic_copy(saved, live, baseline_sha)
                    require(digest(live) == baseline_sha,
                            "V1330 source restoration hash mismatch")
                    records.append({"path": str(live.relative_to(self.workspace)),
                                    "expected_baseline_sha256": baseline_sha,
                                    "restored_sha256": digest(live), "new_helper_absent": False})
            except (OSError, ValueError, RuntimeError) as error:
                issues.append(str(error))
        self.restoration = list(reversed(records))
        if self.backup is not None and not issues:
            shutil.rmtree(self.backup)
        self.restored = not issues
        if issues:
            raise RuntimeError("V1330 SOURCE RESTORE FAILED; backup retained: " + "; ".join(issues))
        return False


def isaac_process_in_listing(listing: str) -> bool:
    launch = re.compile(r"(?:^|[\s/])(?:scripts/sim\.py|scripts/run_teacher_v\d+\.py|"
                        r"scripts/torchrun_sim\.sh|sim\.sh)(?:\s|$)")
    for line in listing.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        comm, arguments = parts
        if (re.fullmatch(r"python(?:\d+(?:\.\d+)?)?|torchrun|bash|sh|zsh", comm)
                and launch.search(arguments)):
            return True
        if comm in {"kit", "isaac-sim", "isaacsim"} and re.search(r"isaac(?:sim|-sim|lab)", arguments):
            return True
    return False


def validate_idle(gpu_memory_mib: list[int], process_listing: str) -> None:
    require(len(gpu_memory_mib) == 4 and
            all(type(value) is int and 0 <= value <= IDLE_GPU_MIB_MAX
                for value in gpu_memory_mib),
            "one or more of four GPUs exceed the idle-memory threshold")
    require(not isaac_process_in_listing(process_listing),
            "Isaac or a training/evaluation launcher is already running")


def gpu_usage_mib() -> list[int]:
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, check=True).stdout
    rows = [line.strip().split(",") for line in raw.splitlines() if line.strip()]
    require(len(rows) == 4, "GPU count != 4")
    return [int(row[1].strip()) for row in rows]


def ensure_idle() -> None:
    listing = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True,
                             text=True, check=True).stdout
    validate_idle(gpu_usage_mib(), listing)


def require_outputs_absent(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        for parent in path.parents:
            if parent == Path("/"):
                break
            require(not parent.is_symlink(), f"symlinked output parent {label}: {parent}")
        require(not path.exists() and not path.is_symlink(),
                f"V1330 output already exists ({label}): {path}")


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(),
            "required V1330 JSON missing/symlinked: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "V1330 JSON root is not an object: " + str(path))
    return value


def verify_external_inputs(workspace: Path, pins: dict[str, str]) -> None:
    require(isinstance(pins, dict) and pins, "V1330 external input pin map is empty")
    for relative, expected_sha in pins.items():
        verified_file(safe_path(workspace, relative), expected_sha,
                      "pinned remote simulator input")


def source_targets(root: Path, workspace: Path, manifest: dict):
    rows = []
    for row in manifest.get("sources", []):
        candidate = safe_path(root, row.get("candidate", ""))
        live = safe_path(workspace, row.get("live", ""))
        require(row.get("candidate") == row.get("live"),
                "V1330 candidate/live source paths differ")
        baseline = manifest.get("baseline", {}).get(row["live"])
        rows.append((candidate, live, baseline, row.get("candidate_sha256", "")))
    return rows


def _contains_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _validate_task_plan(plan: dict) -> None:
    require(not _contains_key(plan, "seed"), "V1330 fixed task plan must not pin a seed")
    require(isinstance(plan.get("families"), list) and len(plan["families"]) == NUM_ENVS and
            set(plan["families"]) == {"combined_part_entrance"},
            "V1330 task plan must contain exactly 512 entrance slots")
    for name in ("difficulty", "difficulty_level", "speed_level"):
        values = plan.get(name)
        require(isinstance(values, list) and len(values) == NUM_ENVS,
                "V1330 task plan slot count changed: " + name)
        require(all(type(value) in (int, float) and 0 <= value <= 4 for value in values),
                "V1330 task plan contains an invalid value: " + name)


def validate_bundle(root: Path, workspace: Path, *, live: bool = True) -> dict:
    require(root == workspace / PACKAGE_REL and workspace == WORKSPACE,
            "V1330 requires the fixed remote workspace/package path")
    require(root.is_dir() and not root.is_symlink(), "V1330 package missing/symlinked")
    manifest_path = root / "configs/manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and
            manifest.get("status") == "authorized-one-shot" and
            manifest.get("run_authorized") is True and
            manifest.get("training_authorized") is True and
            manifest.get("diagnostic_authorized") is False and
            manifest.get("automatic_gate_authorized") is False and
            manifest.get("automatic_retry_authorized") is False and
            manifest.get("automatic_follow_on_authorized") is False and
            manifest.get("source_base") == "v1327",
            "V1330 manifest status/authorization/source base mismatch")
    execution = manifest.get("execution")
    require(isinstance(execution, dict), "V1330 execution contract missing")
    for name, expected in EXPECTED.items():
        require(execution.get(name) == expected,
                "V1330 manifest fixed execution value changed: " + name)
    require(manifest.get("execution", {}).get("diagnostic_evaluation", {}).get("certify") is False,
            "V1330 diagnostic cannot certify a checkpoint")
    require(manifest.get("execution", {}).get("diagnostic_evaluation", {}).get("episodes") == EVALUATION_EPISODES,
            "V1330 diagnostic episode budget changed")

    runner_pins = manifest.get("runner_sha256")
    require(isinstance(runner_pins, dict) and set(runner_pins) == EXPECTED_RUNNER_FILES,
            "V1330 runner bundle pin set is incomplete or expanded")
    for relative, expected_sha in runner_pins.items():
        verified_file(safe_path(root, relative), expected_sha, "V1330 runner file")

    base_manifest_path = safe_path(
        workspace, "v128_work/v132_randomized/v1327/configs/manifest.json",
    )
    verified_file(base_manifest_path, manifest.get("source_base_manifest_sha256", ""),
                  "V1327 source-base manifest")
    base_manifest = read_json(base_manifest_path)
    v1328_manifest_path = safe_path(
        workspace, "v128_work/v132_randomized/v1328/configs/manifest.json",
    )
    verified_file(v1328_manifest_path, manifest.get("base_v1328_manifest_sha256", ""),
                  "V1328 base manifest")
    v1328_manifest = read_json(v1328_manifest_path)
    code_only_path = safe_path(root, "provenance/code_only_manifest.original.json")
    verified_file(code_only_path,
                  manifest.get("provenance", {}).get("code_only_manifest_original_sha256", ""),
                  "V1330 original code-only manifest")
    speed_reference = safe_path(
        workspace, "v128_work/v132_randomized/" + manifest.get("speed_reference", ""),
    )
    verified_file(speed_reference, manifest.get("speed_reference_config_sha256", ""),
                  "V1330 speed-reference config")
    v1328_reward_path = safe_path(workspace,
                                  "v128_work/v132_randomized/v1328/" +
                                  v1328_manifest.get("reward_config", {}).get("path", ""))
    verified_file(v1328_reward_path, manifest.get("base_reward_config_sha256", ""),
                  "V1328 inherited reward config")
    for name in ("observation_profile", "task_plan"):
        inherited = v1328_manifest.get(name, {})
        inherited_path = safe_path(
            workspace, "v128_work/v132_randomized/v1328/" + inherited.get("path", ""),
        )
        verified_file(inherited_path, inherited.get("sha256", ""),
                      "V1328 inherited " + name)
        require(manifest.get("inherited_config_sha256", {}).get(
                    "observation_profile.json" if name == "observation_profile" else "task_plan.json"
                ) == inherited.get("sha256"),
                "V1330 inherited " + name + " differs from V1328")
    require(manifest.get("base_reward_config_sha256") ==
            v1328_manifest.get("reward_config", {}).get("sha256") and
            manifest.get("base_v1328_manifest_sha256") == digest(v1328_manifest_path) and
            manifest.get("candidate_reward_config_sha256") ==
            manifest.get("reward_config", {}).get("sha256") and
            manifest.get("inherited_config_sha256") == {
                "observation_profile.json": manifest.get("observation_profile", {}).get("sha256"),
                "task_plan.json": manifest.get("task_plan", {}).get("sha256"),
            }, "V1330 V1328 lineage/configuration pins changed")
    source_hashes = manifest.get("source_files_sha256")
    base_source_hashes = base_manifest.get("source_files_sha256", {})
    source_deltas = {
        "go1_hier_sim/go1_hier_sim/envs/config.py",
        "go1_hier_sim/go1_hier_sim/envs/runtime.py",
        "go1_hier_sim/go1_hier_sim/reward_config.py",
        "go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py",
    }
    require(isinstance(source_hashes, dict) and set(source_hashes) == EXPECTED_SOURCE_FILES and
            set(base_source_hashes) == EXPECTED_SOURCE_FILES - {
                "go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py"
            } and
            {name for name in base_source_hashes if source_hashes[name] != base_source_hashes[name]} ==
            source_deltas - {"go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py"} and
            "go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py" not in base_source_hashes,
            "V1330 source snapshot contains an unapproved source delta")
    source_rows = manifest.get("sources")
    require(isinstance(source_rows, list) and len(source_rows) == len(EXPECTED_SOURCE_FILES),
            "V1330 source overlay list is incomplete")
    seen = set()
    for row in source_rows:
        candidate_rel = row.get("candidate", "")
        name = candidate_rel.removeprefix("source/")
        require(name in EXPECTED_SOURCE_FILES and name not in seen and
                row.get("live") == candidate_rel and
                row.get("candidate_sha256") == source_hashes[name],
                "V1330 source mapping/hash changed")
        seen.add(name)
        verified_file(safe_path(root, candidate_rel), source_hashes[name],
                      "V1330 source snapshot")
    require(seen == EXPECTED_SOURCE_FILES, "V1330 source snapshot has missing paths")
    baseline = manifest.get("baseline")
    expected_baseline = dict(base_manifest.get("baseline", {}))
    expected_baseline["source/go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py"] = None
    require(isinstance(baseline, dict) and baseline == expected_baseline and
            set(baseline) == {row["live"] for row in source_rows},
            "V1330 live-source baseline set changed")
    for value in baseline.values():
        require(value is None or (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)),
                "V1330 baseline source hash malformed")

    for name, expected in (("observation_profile", "configs/observation_profile.json"),
                           ("reward_config", "configs/rewards_stage1_gaps_v1330_gait_gap.toml"),
                           ("task_plan", "configs/task_plan.json")):
        entry = manifest.get(name, {})
        require(entry.get("path") == expected,
                "V1330 manifest input path changed: " + name)
        verified_file(safe_path(root, expected), entry.get("sha256", ""),
                      "V1330 " + name)
    profile = read_json(root / "configs/observation_profile.json")
    require(profile.get("actor_observation_dim") == 1073 and profile.get("action_dim") == 12 and
            profile.get("environment_overrides", {}).get("GO1_MULTIRES_SCAN") == "1" and
            profile.get("terrain_scan", {}).get("coarse_grid") == [8, 16] and
            profile.get("terrain_scan", {}).get("fine_grid") == [16, 24],
            "V1330 observation/interface contract changed")
    _validate_task_plan(read_json(root / "configs/task_plan.json"))

    input_pins = manifest.get("input_files")
    require(isinstance(input_pins, dict) and set(input_pins) == EXPECTED_INPUTS and
            input_pins == base_manifest.get("input_files") == v1328_manifest.get("input_files"),
            "V1330 external input pin set/hash changed")
    require(execution.get("required_idle_gpu_mib_max") == IDLE_GPU_MIB_MAX and
            execution.get("launchers") == base_manifest.get("execution", {}).get("launchers") and
            set(execution.get("launchers", {})) == EXPECTED_LAUNCHERS,
            "V1330 host/launcher contract changed")
    verify_external_inputs(workspace, {**input_pins, **execution["launchers"]})
    expected_env = {
        "GO1_FALL_RELATIVE": "1", "GO1_POLICY_VARIANT": "map_attn_v3",
        "GO1_PPO_CONFIG_PATH": "configs/ppo_gaps_long.json",
        "GO1_PRIVILEGED_ACTOR": "1",
        "GO1_REWARD_CONFIG_PATH": "v128_work/v132_randomized/v1330/configs/rewards_stage1_gaps_v1330_gait_gap.toml",
        "GO1_SOFT_RESPAWN": "1", "GO1_V128_FILTER_OBS": "1",
        "GO1_V29_FILTER_OBS": "0", "GO1_V19_MIN_WIDTH": "0.05",
        "GO1_V19_PRELAUNCH_REWARD": "1", "GO1_V19_PRELAUNCH_SCALE": "4",
        "GO1_V8_LONG_GAP_THRESHOLD": "0.05", "GO1_V8_TERRAIN_CONTROL": "1",
        "GO1_MULTIRES_SCAN": "1", "GO1_V121_STALL_GROW": "1",
        "GO1_V121_BACKWARD": "1",
    }
    require(execution.get("train_environment") == expected_env and
            not any(name in expected_env for name in ("GO1_WATCHDOG", "GO1_AUTO_P2", "GO1_V29_TEACHER_DISTILL")),
            "V1330 train environment changed or grants automation")

    spec_path = root / "configs/train_execution.json"
    spec = read_json(spec_path)
    expected_evaluation = {
        "stage": "stage1", "terrain": "lp_acrl", "num_envs": 512,
        "episodes": 512, "episode_seconds": 200, "strict_first_episode": True,
        "certify": False, "run_id": DIAGNOSTIC_RUN_ID,
        "trace_path": spec.get("diagnostic_trace_path"),
        "log_path": spec.get("diagnostic_log_path"),
        "receipt_path": spec.get("diagnostic_receipt_path"),
        "report_path": spec.get("diagnostic_report_path"),
    }
    require(execution.get("diagnostic_evaluation") == expected_evaluation and
            spec.get("manifest_sha256") == digest(manifest_path) and
            all(spec.get(name) == execution.get(name) for name in EXPECTED) and
            spec.get("task_plan_path") == manifest["task_plan"]["path"] and
            spec.get("task_plan_sha256") == manifest["task_plan"]["sha256"] and
            spec.get("reward_config_path") == manifest["reward_config"]["path"] and
            spec.get("reward_config_sha256") == manifest["reward_config"]["sha256"] and
            spec.get("observation_profile_path") == manifest["observation_profile"]["path"] and
            spec.get("observation_profile_sha256") == manifest["observation_profile"]["sha256"],
            "V1330 train spec is not pinned to the manifest inputs")
    archive_path = root / "reports/authorized_run_manifest.json"
    verified_file(archive_path, digest(manifest_path), "archived authorized manifest")
    if live:
        check_targets(source_targets(root, workspace, manifest), workspace)
    return manifest


def audit_roots(workspace: Path) -> list[Path]:
    return [workspace / name for name in ("reports", "configs", "v128_work", "source", "scripts")]


def training_paths(workspace: Path, root: Path, spec: dict) -> dict[str, Path]:
    require(spec.get("run_id") == RUN_ID, "V1330 run ID changed")
    return {
        "checkpoint_dir": workspace / "checkpoints/stage1" / RUN_ID,
        "run_dir": workspace / "runs/stage1" / RUN_ID,
        "checkpoint": safe_path(workspace, spec["checkpoint_path"]),
        "stage_report": safe_path(workspace, spec["stage_report_path"]),
        "log": safe_path(root, spec["training_log_path"]),
        "receipt": safe_path(root, spec["training_receipt_path"]),
    }


def diagnostic_paths(workspace: Path, root: Path, spec: dict) -> dict[str, Path]:
    require(spec.get("diagnostic_run_id") == DIAGNOSTIC_RUN_ID,
            "V1330 diagnostic run ID changed")
    return {
        "seed_audit": safe_path(root, spec["diagnostic_seed_audit_path"]),
        "trace": safe_path(root, spec["diagnostic_trace_path"]),
        "log": safe_path(root, spec["diagnostic_log_path"]),
        "receipt": safe_path(root, spec["diagnostic_receipt_path"]),
        "report": safe_path(workspace, spec["diagnostic_report_path"]),
    }


def _validate_seed_receipt(root: Path, workspace: Path, spec: dict,
                           purpose: str, seed_name: str, path_name: str,
                           sha_name: str) -> Path:
    seed = spec.get(seed_name)
    require(type(seed) is int and seed not in FORBIDDEN_SEEDS,
            f"V1330 {purpose} seed missing or forbidden")
    audit = safe_path(root, spec.get(path_name, ""))
    expected_sha = spec.get(sha_name)
    verified_file(audit, expected_sha, f"V1330 {purpose} fresh-seed audit")
    text = audit.read_text(encoding="utf-8")
    require(text.startswith("V1330_SEED_AUDIT_V1\n") and
            f"PURPOSE {purpose}\n" in text and f"SEED {seed}\n" in text and
            re.search(r"^FILES_SCANNED [1-9][0-9]*$", text, re.MULTILINE) and
            re.search(r"^BYTES_SCANNED [0-9]+$", text, re.MULTILINE) and
            "HITS []\n" in text and "RESULT UNUSED_IN_AUDITED_TEXT\n" in text,
            f"V1330 {purpose} seed audit receipt is incomplete")
    hits, _files, _size = scan_seed(
        seed, audit_roots(workspace), ignore_paths=[audit, root / "configs/train_execution.json"],
    )
    require(not hits, f"V1330 {purpose} seed now appears in prior text: " + ", ".join(hits[:5]))
    return audit


def validate_unseeded_run(root: Path, workspace: Path, manifest: dict, spec: dict) -> None:
    require(spec.get("train_seed") is None and spec.get("train_seed_audit_sha256") is None and
            spec.get("diagnostic_seed") is None and spec.get("diagnostic_seed_audit_sha256") is None,
            "V1330 seeds must be unset before fresh-seed audit")
    paths = training_paths(workspace, root, spec)
    eval_paths = diagnostic_paths(workspace, root, spec)
    require_outputs_absent({**paths, **eval_paths,
                            "training_seed_audit": safe_path(root, spec["train_seed_audit_path"]),
                            "diagnostic_seed_audit": safe_path(root, spec["diagnostic_seed_audit_path"])})


def validate_seeded_run(root: Path, workspace: Path, manifest: dict,
                        spec: dict) -> tuple[dict, dict[str, Path]]:
    validate_train_contract(manifest["execution"], spec)
    require(spec.get("manifest_sha256") == digest(root / "configs/manifest.json"),
            "V1330 training spec manifest lineage mismatch")
    require(spec.get("task_plan_sha256") == manifest["task_plan"]["sha256"] and
            spec.get("reward_config_sha256") == manifest["reward_config"]["sha256"] and
            spec.get("observation_profile_sha256") == manifest["observation_profile"]["sha256"],
            "V1330 training input provenance changed")
    train_audit = _validate_seed_receipt(
        root, workspace, spec, "training", "train_seed", "train_seed_audit_path",
        "train_seed_audit_sha256",
    )
    require(spec.get("diagnostic_seed") is None and
            spec.get("diagnostic_seed_audit_sha256") is None,
            "V1330 training-only run cannot carry a diagnostic seed")
    plan = safe_path(root, spec["task_plan_path"])
    verified_file(plan, spec["task_plan_sha256"], "V1330 task plan")
    verified_file(safe_path(root, spec["reward_config_path"]),
                  spec["reward_config_sha256"], "V1330 reward config")
    verified_file(safe_path(root, spec["observation_profile_path"]),
                  spec["observation_profile_sha256"], "V1330 observation profile")
    paths = training_paths(workspace, root, spec)
    eval_paths = diagnostic_paths(workspace, root, spec)
    require_outputs_absent({**paths,
                            "diagnostic_trace": eval_paths["trace"],
                            "diagnostic_log": eval_paths["log"],
                            "diagnostic_receipt": eval_paths["receipt"],
                            "diagnostic_report": eval_paths["report"]})
    return {"task_plan": plan, "train_seed_audit": train_audit}, paths


def read_successful_training_receipt(root: Path, workspace: Path,
                                     manifest: dict, spec: dict) -> tuple[dict, dict[str, Path]]:
    train_paths = training_paths(workspace, root, spec)
    receipt_path = train_paths["receipt"]
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "trained-unverified" and
            receipt.get("run_id") == RUN_ID and receipt.get("restored") is True and
            receipt.get("manifest_sha256") == digest(root / "reports/authorized_run_manifest.json") and
            receipt.get("train_spec_sha256") == digest(root / "configs/train_execution.json") and
            receipt.get("checkpoint_path") == str(train_paths["checkpoint"]) and
            receipt.get("formal_gate") is False and
            receipt.get("automatic_gate_authorized") is False and
            receipt.get("automatic_retry_authorized") is False and
            receipt.get("automatic_follow_on_authorized") is False,
            "V1330 successful training receipt missing or invalid")
    verified_file(train_paths["checkpoint"], receipt.get("checkpoint_sha256", ""),
                  "V1330 trained checkpoint")
    verified_file(train_paths["stage_report"], receipt.get("stage_report_sha256", ""),
                  "V1330 training stage report")
    verified_file(train_paths["log"], receipt.get("training_log_sha256", ""),
                  "V1330 training log")
    require(receipt.get("num_envs") == NUM_ENVS and receipt.get("iterations") == ITERATIONS and
            receipt.get("train_seed") == spec.get("train_seed") and
            receipt.get("train_seed_audit_sha256") == spec.get("train_seed_audit_sha256"),
            "V1330 training receipt differs from the fixed run contract")
    return receipt, train_paths


def validate_diagnostic_ready(root: Path, workspace: Path, manifest: dict,
                              spec: dict) -> tuple[dict, dict[str, Path]]:
    training_receipt, train_paths = read_successful_training_receipt(root, workspace, manifest, spec)
    eval_paths = diagnostic_paths(workspace, root, spec)
    _validate_seed_receipt(
        root, workspace, spec, "diagnostic-evaluation", "diagnostic_seed",
        "diagnostic_seed_audit_path", "diagnostic_seed_audit_sha256",
    )
    require(spec.get("diagnostic_seed") != spec.get("train_seed"),
            "V1330 diagnostic seed is not independent")
    validate_diagnostic_contract(manifest["execution"], spec, training_receipt,
                                 train_paths["checkpoint"])
    require_outputs_absent({"diagnostic_trace": eval_paths["trace"],
                            "diagnostic_log": eval_paths["log"],
                            "diagnostic_receipt": eval_paths["receipt"],
                            "diagnostic_report": eval_paths["report"]})
    return training_receipt, {**train_paths, **eval_paths}
