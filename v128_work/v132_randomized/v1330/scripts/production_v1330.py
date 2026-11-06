#!/usr/bin/env python3
"""Training-only V1330 runner: one locked fresh-seed run, no evaluation or gate."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from contracts_v1330 import (
    DIAGNOSTIC_RUN_ID, EVALUATION_EPISODES, ITERATIONS, NUM_ENVS, RUN_ID,
    diagnostic_command, require, train_command,
)
from preflight_v1330 import (
    Overlay, WORKSPACE, PACKAGE_REL, check_targets, diagnostic_paths, digest,
    ensure_idle, lock_fd_matches, read_json, require_outputs_absent, source_targets,
    training_paths, validate_bundle, validate_diagnostic_ready, validate_idle,
    validate_seeded_run, validate_unseeded_run, verify_external_inputs,
    audit_roots, safe_path, _validate_seed_receipt,
)
from seed_audit_v1330 import select_unused_seed, write_seed_audit

ALLOWED_MODES = ("preflight-seed", "audit-train-seed", "preflight-train", "train")
_active_process: subprocess.Popen | None = None


def mode_authorization(mode: str) -> str:
    require(mode in ALLOWED_MODES, "mode is not implemented in V1330")
    return {
        "audit-train-seed": "seed-audit-train-only",
        "train": "one-train-only",
    }.get(mode, "")


def validate_mode_authorization(mode: str, environment: dict[str, str]) -> None:
    require(mode in ALLOWED_MODES, "mode is not implemented in V1330")
    if mode.startswith("preflight-"):
        require(environment.get("PREFLIGHT_ONLY") == "1",
                "preflight requires PREFLIGHT_ONLY=1")
        require(not environment.get("V1330_RUN_AUTHORIZED"),
                "preflight cannot carry V1330 run authorization")
        return
    require(environment.get("PREFLIGHT_ONLY", "0") != "1",
            "production mode cannot use PREFLIGHT_ONLY")
    require(environment.get("V1330_RUN_AUTHORIZED") == mode_authorization(mode),
            "explicit V1330 mode authorization absent")
    require(environment.get("V1330_MOCK", "0") != "1",
            "V1330 mock mode is not implemented")


def _lock(workspace: Path) -> None:
    lock = workspace / "reports/v7_sim.lock"
    require(lock.is_file() and not lock.is_symlink(),
            "pre-existing regular simulator lock missing/symlinked")
    require(lock_fd_matches(9, lock), "exclusive simulator lock FD 9 missing or mismatched")


def _execution_environment(workspace: Path, root: Path, manifest: dict,
                           run_id: str, plan: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("GO1_"):
            environment.pop(name, None)
    for name, value in manifest["execution"]["train_environment"].items():
        value = str(value)
        if name.endswith("_PATH"):
            value = str(workspace / value)
        environment[name] = value
    environment.update({
        "GO1_RUN_ID": run_id,
        "GO1_TASK_PLAN": str(plan),
        "GO1_STAGE1_DIRECTION_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "ISAACLAB_PATH": "/config/Desktop/IsaacLab",
    })
    for name in (
        "WORLD_SIZE", "RANK", "LOCAL_RANK", "DISPLAY", "WAYLAND_DISPLAY",
        "PYTHONPATH", "SIM_VIRTUAL_ENV", "PREFLIGHT_ONLY", "V1330_RUN_AUTHORIZED",
        "V1330_MOCK", "SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "SSHPASS",
    ):
        environment.pop(name, None)
    return environment


def _snapshot_external_inputs(workspace: Path, manifest: dict) -> dict[str, str]:
    pins = {**manifest["input_files"], **manifest["execution"]["launchers"]}
    require(pins, "no external simulator inputs are pinned")
    verify_external_inputs(workspace, pins)
    return pins


def _write_json_exclusive(path: Path, value: dict) -> None:
    require(not path.exists() and not path.is_symlink(),
            "V1330 receipt already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_json_atomic(path: Path, value: dict, expected_current_sha: str) -> None:
    require(path.is_file() and not path.is_symlink() and digest(path) == expected_current_sha,
            "V1330 JSON changed before atomic update: " + str(path))
    fd, name = tempfile.mkstemp(prefix=".v1330-json-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_logged(command: list[str], log: Path, environment: dict[str, str],
                workspace: Path, label: str) -> None:
    global _active_process
    require(not log.exists() and not log.is_symlink(), "V1330 " + label + " log already exists")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=workspace, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        _active_process = process
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            _active_process = None


def _interrupt_handler(signum, _frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    process = _active_process
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    raise KeyboardInterrupt(f"signal {signum}")


def _signal_handlers():
    return {sig: signal.signal(sig, _interrupt_handler) for sig in (signal.SIGINT, signal.SIGTERM)}


def _restore_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _validate_train_outputs(paths: dict, plan: Path) -> dict:
    for name in ("checkpoint", "stage_report", "log"):
        path = paths[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                "V1330 training output missing/empty/symlinked: " + name)
    stage = read_json(paths["stage_report"])
    require(stage.get("run") == RUN_ID and stage.get("iterations") == ITERATIONS and
            stage.get("task_plan") == str(plan) and
            type(stage.get("episodes")) is int and stage["episodes"] > 0 and
            isinstance(stage.get("environments"), dict) and
            set(stage["environments"]) == {str(index) for index in range(NUM_ENVS)},
            "V1330 full training report does not cover the exact 512-slot/650-iteration contract")
    metrics = {}
    for name in ("completion_rate", "survival_rate"):
        value = stage.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0,
                "V1330 training report has invalid metric: " + name)
        metrics[name] = float(value)
    for name in ("mean_command_speed", "mean_achieved_speed"):
        value = stage.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0.0,
                "V1330 training report has invalid metric: " + name)
        metrics[name] = float(value)
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    require("TRAINING FINISHED" in log_text,
            "V1330 CLI did not positively report training completion")
    return {
        "stage_report_sha256": digest(paths["stage_report"]),
        "training_log_sha256": digest(paths["log"]),
        "checkpoint_sha256": digest(paths["checkpoint"]),
        "training_episodes": stage["episodes"],
        "training_metrics": metrics,
    }


def _validate_diagnostic_outputs(paths: dict, spec: dict,
                                 training_receipt: dict) -> dict:
    for name in ("trace", "report", "log"):
        path = paths[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                "V1330 diagnostic output missing/empty/symlinked: " + name)
    report = read_json(paths["report"])
    require(report.get("stage") == "stage1" and report.get("terrain") == "lp_acrl" and
            report.get("seed") == spec.get("diagnostic_seed") and
            report.get("episodes") == 512 and
            report.get("checkpoint_sha256") == training_receipt.get("checkpoint_sha256"),
            "V1330 diagnostic report does not match its fixed run/checkpoint/seed")
    environments = report.get("environments")
    expected_envs = {str(index) for index in range(512)}
    require(isinstance(environments, dict) and set(environments) == expected_envs and
            all(type(environments[key].get("episodes")) is int and
                environments[key]["episodes"] == 1 for key in expected_envs),
            "V1330 diagnostic report does not contain one episode for each of 512 environments")
    rows = []
    with paths["trace"].open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(line.strip(), f"V1330 diagnostic trace has a blank row at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"V1330 diagnostic trace JSON invalid at row {line_number}") from error
            require(isinstance(row, dict), "V1330 diagnostic trace row is not an object")
            rows.append(row)
    require(len(rows) == 512, "V1330 diagnostic trace must contain exactly 512 rows")
    ids = [row.get("env_id") for row in rows]
    require(all(type(env_id) is int and 0 <= env_id < 512 for env_id in ids) and
            sorted(ids) == list(range(512)),
            "V1330 diagnostic trace has missing or duplicate environments")
    for row in rows:
        require(type(row.get("episode_index")) is int and row["episode_index"] == 0 and
                type(row.get("steps")) is int and row["steps"] > 0,
                "V1330 diagnostic trace is not each environment's first completed episode")
        for name in ("course_x", "course_y"):
            value = row.get(name)
            require(type(value) in (int, float) and math.isfinite(value),
                    "V1330 diagnostic trace has invalid failure position: " + name)
        for flag in ("success", "fallen", "collision", "out_of_bounds", "fell_into_gap"):
            require(type(row.get(flag)) is bool,
                    "V1330 diagnostic trace is missing terminal flag: " + flag)

    for row in rows:
        entry = environments[str(row["env_id"])]
        for flag in ("success", "fallen", "collision", "out_of_bounds", "fell_into_gap"):
            expected_count = int(row[flag])
            require(type(entry.get(flag)) is int and entry[flag] == expected_count,
                    "V1330 diagnostic per-environment report/trace mismatch: " + flag)

    successes = sum(row["success"] and not row["fallen"] and not row["collision"] and
                    not row["out_of_bounds"] and not row["fell_into_gap"] for row in rows)
    fallen = sum(row["fallen"] for row in rows)
    collisions = sum(row["collision"] for row in rows)
    out_of_bounds = sum(row["out_of_bounds"] for row in rows)
    gap_falls = sum(row["fell_into_gap"] for row in rows)
    rates = {
        "completion_rate": successes / 512,
        "success_rate": sum(row["success"] for row in rows) / 512,
        "fallen_rate": fallen / 512,
        "collision_rate": collisions / 512,
        "out_of_bounds_rate": out_of_bounds / 512,
        "fell_into_gap_rate": gap_falls / 512,
    }
    for name, expected in rates.items():
        observed = report.get(name)
        require(type(observed) in (int, float) and abs(observed - expected) <= 1e-9,
                "V1330 diagnostic report/trace summary mismatch: " + name)
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    require("EPISODE_TRACE:" in log_text and "REPORT:" in log_text,
            "V1330 diagnostic log does not confirm trace and report outputs")
    failure_positions = [float(row["course_x"]) for row in rows
                         if not (row["success"] and not row["fallen"] and not row["collision"]
                                 and not row["out_of_bounds"] and not row["fell_into_gap"])]
    return {
        "episodes": len(rows), "successes": successes, "fallen": fallen,
        "collisions": collisions, "out_of_bounds": out_of_bounds,
        "fell_into_gap": gap_falls, "completion_rate": rates["completion_rate"],
        "success_rate": rates["success_rate"], "failure_x_samples": failure_positions,
        "trace_sha256": digest(paths["trace"]),
        "report_sha256": digest(paths["report"]),
        "evaluation_log_sha256": digest(paths["log"]),
    }


def _seed_audit_roots(workspace: Path) -> list[Path]:
    return audit_roots(workspace)


def _write_seed_spec(spec_path: Path, spec: dict) -> None:
    current_sha = digest(spec_path)
    _write_json_atomic(spec_path, spec, current_sha)


def execute_seed_audit(mode: str, root: Path, workspace: Path, manifest: dict) -> dict:
    require(mode == "audit-train-seed", "unsupported V1330 seed-audit mode")
    spec_path = root / "configs/train_execution.json"
    spec = read_json(spec_path)
    roots = _seed_audit_roots(workspace)
    ensure_idle()
    validate_unseeded_run(root, workspace, manifest, spec)
    purpose = "training"
    seed_field = "train_seed"
    audit_field = "train_seed_audit_path"
    sha_field = "train_seed_audit_sha256"
    seed = select_unused_seed(roots)
    audit_path = root / spec[audit_field]
    record = write_seed_audit(seed, purpose, roots, audit_path)
    audit_sha = digest(audit_path)
    spec[seed_field] = seed
    spec[sha_field] = audit_sha
    _write_seed_spec(spec_path, spec)
    print(json.dumps({"status": "fresh-seed-audited", "purpose": purpose,
                      "seed": seed, "files_scanned": record["files_scanned"],
                      "bytes_scanned": record["bytes_scanned"],
                      "audit_path": str(audit_path), "audit_sha256": audit_sha}, sort_keys=True))
    return {"purpose": purpose, "seed": seed, "audit_sha256": audit_sha}


def _has_incomplete_overlay(workspace: Path) -> bool:
    reports = workspace / "reports"
    return any(list(reports.glob(f"v{version}-overlay-*")) for version in
               (1325, 1326, 1327, 1328, 1329, 1330))


def _revoke_authorization(root: Path, status: str) -> None:
    manifest_path = root / "configs/manifest.json"
    spec_path = root / "configs/train_execution.json"
    archive_path = root / "reports/authorized_run_manifest.json"
    require(archive_path.is_file() and not archive_path.is_symlink(),
            "V1330 archived authorization manifest missing/symlinked")
    authorized_sha = digest(archive_path)
    current_manifest_sha = digest(manifest_path)
    manifest = read_json(manifest_path)
    revoked = (manifest.get("run_authorized") is False and
               manifest.get("training_authorized") is False and
               manifest.get("diagnostic_authorized") is False and
               manifest.get("automatic_gate_authorized") is False and
               manifest.get("automatic_retry_authorized") is False and
               manifest.get("automatic_follow_on_authorized") is False and
               manifest.get("training_run_manifest_sha256") == authorized_sha)
    if current_manifest_sha == authorized_sha:
        require(manifest.get("run_authorized") is True,
                "V1330 authorization already revoked or changed")
        manifest.update({
            "status": status,
            "run_authorized": False,
            "training_authorized": False,
            "diagnostic_authorized": False,
            "automatic_gate_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False,
            "training_run_manifest_sha256": authorized_sha,
        })
        execution = manifest["execution"]
        execution.update({
            "run_authorized": False,
            "training_authorized": False,
            "diagnostic_authorized": False,
            "automatic_gate_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False,
        })
        _write_json_atomic(manifest_path, manifest, current_manifest_sha)
    else:
        require(revoked and manifest.get("status") == status,
                "V1330 active manifest differs from the archived authorization")
    current_spec_sha = digest(spec_path)
    spec = read_json(spec_path)
    spec.update({
        "run_authorized": False,
        "training_authorized": False,
        "diagnostic_authorized": False,
        "automatic_gate_authorized": False,
        "automatic_retry_authorized": False,
        "automatic_follow_on_authorized": False,
        "execution_status": status,
    })
    _write_json_atomic(spec_path, spec, current_spec_sha)


def execute_train(root: Path, workspace: Path, manifest: dict) -> dict:
    spec = read_json(root / "configs/train_execution.json")
    inputs, paths = validate_seeded_run(root, workspace, manifest, spec)
    plan = inputs["task_plan"]
    command = train_command(workspace, manifest["execution"], spec, plan)
    rows = source_targets(root, workspace, manifest)
    check_targets(rows, workspace)
    ensure_idle()
    require(not _has_incomplete_overlay(workspace),
            "incomplete simulator source overlay backup exists; manual recovery required")
    manifest_sha = digest(root / "configs/manifest.json")
    spec_sha = digest(root / "configs/train_execution.json")
    plan_sha = digest(plan)
    reward_path = root / spec["reward_config_path"]
    reward_sha = digest(reward_path)
    observation_path = root / spec["observation_profile_path"]
    observation_sha = digest(observation_path)
    external_inputs = _snapshot_external_inputs(workspace, manifest)
    environment = _execution_environment(workspace, root, manifest, RUN_ID, plan)
    previous = _signal_handlers()
    overlay = Overlay(rows, workspace)
    launch_attempted = False
    try:
        with overlay:
            launch_attempted = True
            _run_logged(command, paths["log"], environment, workspace, "training")
        require(overlay.restored, "V1330 source overlay did not restore after training")
        check_targets(rows, workspace)
        ensure_idle()
        require(digest(root / "configs/manifest.json") == manifest_sha and
                digest(root / "configs/train_execution.json") == spec_sha and
                digest(plan) == plan_sha == spec["task_plan_sha256"] and
                digest(reward_path) == reward_sha == spec["reward_config_sha256"] and
                digest(observation_path) == observation_sha == spec["observation_profile_sha256"],
                "V1330 pinned package input changed while training")
        verify_external_inputs(workspace, external_inputs)
        outputs = _validate_train_outputs(paths, plan)
        record = {
            "status": "trained-unverified", "mode": "train", "restored": True,
            "manifest_sha256": manifest_sha, "train_spec_sha256": spec_sha,
            "run_id": RUN_ID, "train_seed": spec["train_seed"],
            "train_seed_audit_sha256": spec["train_seed_audit_sha256"],
            "num_envs": NUM_ENVS, "iterations": ITERATIONS,
            "critic_warmup_iterations": 0, "initialization": "random",
            "checkpoint_path": str(paths["checkpoint"]),
            "checkpoint_sha256": outputs["checkpoint_sha256"],
            "task_plan_path": str(plan), "task_plan_sha256": plan_sha,
            "reward_config_path": str(reward_path), "reward_config_sha256": reward_sha,
            "observation_profile_sha256": observation_sha,
            "stage_report_sha256": outputs["stage_report_sha256"],
            "training_log_sha256": outputs["training_log_sha256"],
            "training_episodes": outputs["training_episodes"],
            "training_metrics": outputs["training_metrics"],
            "source_restoration": overlay.restoration,
            "final_holdout": False, "final_gate_passed": False,
            "diagnostic_ran": False, "formal_gate": False,
            "automatic_gate_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False,
        }
        _write_json_exclusive(paths["receipt"], record)
        _revoke_authorization(root, "trained-unverified")
        print(json.dumps({"status": record["status"], "receipt": str(paths["receipt"]),
                          "checkpoint_sha256": record["checkpoint_sha256"],
                          "stage_report_sha256": record["stage_report_sha256"],
                          "training_log_sha256": record["training_log_sha256"],
                          "restored": True, "diagnostic_ran": False, "gate_ran": False},
                         sort_keys=True))
        return record
    except BaseException as error:
        if launch_attempted:
            receipt_status = "training-failed"
            if not paths["receipt"].exists() and not paths["receipt"].is_symlink():
                restored = bool(overlay.restored)
                try:
                    _write_json_exclusive(paths["receipt"], {
                        "status": receipt_status, "mode": "train", "restored": restored,
                        "run_id": RUN_ID, "train_seed": spec.get("train_seed"),
                        "error": f"{type(error).__name__}: {error}",
                        "source_restoration": overlay.restoration if restored else [],
                        "automatic_retry_authorized": False,
                        "automatic_gate_authorized": False,
                        "automatic_follow_on_authorized": False,
                    })
                except BaseException as receipt_error:
                    print(f"V1330_FAILURE_RECEIPT_BLOCKED: {receipt_error}", file=sys.stderr)
            else:
                try:
                    receipt_status = read_json(paths["receipt"]).get("status", receipt_status)
                except BaseException:
                    pass
            if receipt_status in {"trained-unverified", "training-failed"}:
                try:
                    _revoke_authorization(root, receipt_status)
                except BaseException as revoke_error:
                    try:
                        _revoke_authorization(root, receipt_status)
                    except BaseException as retry_error:
                        print(f"V1330_AUTHORIZATION_REVOCATION_BLOCKED: {revoke_error}; {retry_error}",
                              file=sys.stderr)
        raise
    finally:
        _restore_handlers(previous)


def execute_diagnostic(root: Path, workspace: Path, manifest: dict) -> dict:
    spec = read_json(root / "configs/train_execution.json")
    training_receipt, paths = validate_diagnostic_ready(root, workspace, manifest, spec)
    plan = safe_path(root, spec["task_plan_path"])
    checkpoint = paths["checkpoint"]
    command = diagnostic_command(workspace, manifest["execution"], spec, checkpoint,
                                 paths["trace"], training_receipt)
    rows = source_targets(root, workspace, manifest)
    check_targets(rows, workspace)
    ensure_idle()
    require(not _has_incomplete_overlay(workspace),
            "incomplete simulator source overlay backup exists; manual recovery required")
    manifest_sha = digest(root / "configs/manifest.json")
    spec_sha = digest(root / "configs/train_execution.json")
    plan_sha = digest(plan)
    reward_path = root / spec["reward_config_path"]
    reward_sha = digest(reward_path)
    observation_path = root / spec["observation_profile_path"]
    observation_sha = digest(observation_path)
    external_inputs = _snapshot_external_inputs(workspace, manifest)
    environment = _execution_environment(workspace, root, manifest, DIAGNOSTIC_RUN_ID, plan)
    previous = _signal_handlers()
    overlay = Overlay(rows, workspace)
    launch_attempted = False
    try:
        with overlay:
            launch_attempted = True
            _run_logged(command, paths["log"], environment, workspace, "diagnostic evaluation")
        require(overlay.restored, "V1330 source overlay did not restore after diagnostic")
        check_targets(rows, workspace)
        ensure_idle()
        require(digest(root / "configs/manifest.json") == manifest_sha and
                digest(root / "configs/train_execution.json") == spec_sha and
                digest(plan) == plan_sha == spec["task_plan_sha256"] and
                digest(reward_path) == reward_sha == spec["reward_config_sha256"] and
                digest(observation_path) == observation_sha == spec["observation_profile_sha256"],
                "V1330 pinned package input changed during diagnostic evaluation")
        verify_external_inputs(workspace, external_inputs)
        result = _validate_diagnostic_outputs(paths, spec, training_receipt)
        record = {
            "status": "diagnostic-complete", "mode": "evaluate", "restored": True,
            "formal_gate": False, "manifest_sha256": manifest_sha,
            "train_spec_sha256": spec_sha, "run_id": DIAGNOSTIC_RUN_ID,
            "training_run_id": RUN_ID,
            "train_seed": spec["train_seed"], "diagnostic_seed": spec["diagnostic_seed"],
            "diagnostic_seed_audit_sha256": spec["diagnostic_seed_audit_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": training_receipt["checkpoint_sha256"],
            "num_envs": NUM_ENVS, "episodes": EVALUATION_EPISODES,
            "strict_first_episode": True,
            "successes": result["successes"], "fallen": result["fallen"],
            "collisions": result["collisions"], "out_of_bounds": result["out_of_bounds"],
            "fell_into_gap": result["fell_into_gap"],
            "completion_rate": result["completion_rate"],
            "success_rate": result["success_rate"],
            "failure_x_samples": result["failure_x_samples"],
            "trace_sha256": result["trace_sha256"],
            "report_sha256": result["report_sha256"],
            "evaluation_log_sha256": result["evaluation_log_sha256"],
            "source_restoration": overlay.restoration,
            "automatic_gate_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_follow_on_authorized": False,
        }
        _write_json_exclusive(paths["receipt"], record)
        _revoke_authorization(root, "diagnostic-complete")
        print(json.dumps({"status": record["status"], "receipt": str(paths["receipt"]),
                          "episodes": record["episodes"], "successes": record["successes"],
                          "completion_rate": record["completion_rate"],
                          "formal_gate": False, "authorization_revoked": True,
                          "trace_sha256": record["trace_sha256"],
                          "report_sha256": record["report_sha256"]}, sort_keys=True))
        return record
    except BaseException as error:
        if launch_attempted:
            receipt_status = "diagnostic-failed"
            if not paths["receipt"].exists() and not paths["receipt"].is_symlink():
                restored = bool(overlay.restored)
                try:
                    _write_json_exclusive(paths["receipt"], {
                        "status": receipt_status, "mode": "evaluate", "restored": restored,
                        "run_id": DIAGNOSTIC_RUN_ID, "training_run_id": RUN_ID,
                        "diagnostic_seed": spec.get("diagnostic_seed"),
                        "error": f"{type(error).__name__}: {error}",
                        "source_restoration": overlay.restoration if restored else [],
                        "formal_gate": False, "automatic_retry_authorized": False,
                        "automatic_gate_authorized": False,
                        "automatic_follow_on_authorized": False,
                    })
                except BaseException as receipt_error:
                    print(f"V1330_DIAGNOSTIC_FAILURE_RECEIPT_BLOCKED: {receipt_error}", file=sys.stderr)
            else:
                try:
                    receipt_status = read_json(paths["receipt"]).get("status", receipt_status)
                except BaseException:
                    pass
            if receipt_status in {"diagnostic-complete", "diagnostic-failed"}:
                try:
                    _revoke_authorization(root, receipt_status)
                except BaseException as revoke_error:
                    try:
                        _revoke_authorization(root, receipt_status)
                    except BaseException as retry_error:
                        print(f"V1330_AUTHORIZATION_REVOCATION_BLOCKED: {revoke_error}; {retry_error}",
                              file=sys.stderr)
        raise
    finally:
        _restore_handlers(previous)


def run(mode: str, root: Path, workspace: Path) -> None:
    validate_mode_authorization(mode, dict(os.environ))
    require(root == workspace / PACKAGE_REL and workspace == WORKSPACE,
            "V1330 fixed remote workspace/package path mismatch")
    _lock(workspace)
    manifest = validate_bundle(root, workspace, live=True)
    if mode.startswith("preflight-"):
        ensure_idle()
        require(not _has_incomplete_overlay(workspace),
                "incomplete simulator source overlay backup exists; manual recovery required")
        check_targets(source_targets(root, workspace, manifest), workspace)
        purpose = mode.removeprefix("preflight-")
        spec = read_json(root / "configs/train_execution.json")
        if purpose == "seed":
            validate_unseeded_run(root, workspace, manifest, spec)
        elif purpose == "train":
            validate_seeded_run(root, workspace, manifest, spec)
        else:
            validate_diagnostic_ready(root, workspace, manifest, spec)
        print(f"PREFLIGHT_PASS mode={purpose} read_only=true")
        return
    if mode == "audit-train-seed":
        execute_seed_audit(mode, root, workspace, manifest)
    elif mode == "train":
        execute_train(root, workspace, manifest)
    else:
        raise ValueError("unhandled training-only V1330 production mode")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        require(len(args) == 1 and args[0] in ALLOWED_MODES,
                "exactly one supported V1330 mode is required")
        run(args[0], Path(__file__).resolve().parents[1], WORKSPACE)
    except (IndexError, KeyError, OSError, TypeError, ValueError, RuntimeError,
            subprocess.CalledProcessError, KeyboardInterrupt) as error:
        print(f"V1330_BLOCKED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
