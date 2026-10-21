"""Strict, self-describing checkpoints. Engineering runs are never capability-certified."""

import hashlib
import json
import math
import os
import copy
from pathlib import Path
import tempfile

import torch

from .contracts import interface_manifest


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {name} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_checkpoint(path, stage, models, optimizer=None, metadata=None):
    if stage not in {"stage1", "stage2", "stage3a", "stage3b", "stage4"}:
        raise ValueError(f"unknown checkpoint stage {stage}")
    if not models:
        raise ValueError("checkpoint has no models")
    metadata = dict(metadata or {})
    metadata.setdefault("capability_validated", False)
    metadata.setdefault("purpose", "integration")
    if metadata["capability_validated"]:
        raise ValueError("save_checkpoint cannot certify capability; use certify_checkpoint with a passing report")
    # Metadata must be portable primitives, never arbitrary Python objects.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    states = {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
              for name, model in models.items()}
    for name, state in states.items():
        for key, value in state.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"refusing nonfinite checkpoint tensor {name}.{key}")
    payload = {
        "format": "go1_hier_sim", "stage": stage, "interfaces": interface_manifest(),
        "models": states, "model_hashes": {name: state_hash(state) for name, state in states.items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temp = Path(stream.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    write_json(path.with_suffix(".json"), {
        "format": payload["format"], "stage": stage, "interfaces": payload["interfaces"],
        "model_hashes": payload["model_hashes"], "metadata": metadata,
        "checkpoint_sha256": file_hash(path),
    })
    return path


def _is_schema3_heading_migration(interfaces):
    """Accept only the immediately preceding Stage-1 interface for V17->V18."""
    expected = copy.deepcopy(interface_manifest())
    expected["schema_version"] = 3
    expected["locomotion"]["input"] = 1071
    expected["locomotion"]["critic"] = 1078
    expected["normalization"].pop("initial_heading", None)
    return interfaces == expected


def _canonical_interfaces(interfaces):
    """Reduce a manifest to what schemas 4 and 5 have in common."""
    canon = copy.deepcopy(interfaces)
    vision = canon.get("vision") or {}
    if "obstacle" in vision:
        vision["terrain_relief"] = vision.pop("obstacle")
    for key in ("relief_rows", "relief_cols"):
        vision.pop(key, None)
    (canon.get("normalization") or {}).pop("relief", None)
    canon.pop("schema_version", None)
    return canon


def _is_schema4_relief_migration(interfaces):
    """Accept the schema-4 manifest for the frozen Stage-1 locomotion actor.

    Schema 5 renamed ``vision.obstacle`` to ``vision.terrain_relief`` and added
    the relief shape keys without changing a dimension. Only the locomotion actor
    is loaded through this path and it never reads that branch, so accepting the
    schema-4 manifest cannot change a Stage-1 function; a dimension change still
    fails the canonical comparison.
    """
    if not isinstance(interfaces, dict) or interfaces.get("schema_version") != 4:
        return False
    return _canonical_interfaces(interfaces) == _canonical_interfaces(interface_manifest())


def load_checkpoint(path, expected_stage=None, require_validated=False,
                    allow_locomotion_heading_migration=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint does not exist: {path}")
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != file_hash(path):
        raise ValueError(f"checkpoint file hash mismatch: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    compatible_interface = data.get("interfaces") == interface_manifest()
    compatible_migration = (
        allow_locomotion_heading_migration
        and expected_stage == "stage1"
        and (
            _is_schema3_heading_migration(data.get("interfaces"))
            or _is_schema4_relief_migration(data.get("interfaces"))
        )
    )
    if data.get("format") != "go1_hier_sim" or not (compatible_interface or compatible_migration):
        raise ValueError("checkpoint format, dimensions, normalization or joint order is incompatible")
    if expected_stage is not None and data["stage"] != expected_stage:
        raise ValueError(f"expected {expected_stage}, received {data['stage']}")
    for key in ("format", "stage", "interfaces", "model_hashes", "metadata"):
        if manifest.get(key) != data.get(key):
            raise ValueError(f"checkpoint manifest disagrees with payload: {key}")
    for name, state in data["models"].items():
        if data["model_hashes"].get(name) != state_hash(state):
            raise ValueError(f"model state hash mismatch: {name}")
    if require_validated and not data["metadata"].get("capability_validated", False):
        raise ValueError("checkpoint has not passed capability validation; integration tests are not training approval")
    return data


def _stage1_report_passed(report):
    return (
        report.get("protocol") == "stage1-v2" and report.get("terrain") == "course"
        and int(report.get("episodes", 0)) >= 400
        and float(report.get("completion_rate", -1)) >= 0.95
        and float(report.get("velocity_xy_rmse", math.inf)) <= 0.15
        and float(report.get("yaw_rmse", math.inf)) <= 0.25
    )


def _stage1_gate_passed(report):
    """A gate record certifies only when every measured cell holds.

    ``passed`` is set by the recorder, but it is recomputed here so a hand edited
    or stale record cannot certify a policy on its own word.
    """
    from .gate import GATE_PROTOCOL, cell_passes

    if report.get("protocol") != GATE_PROTOCOL:
        return False
    acceptance = report.get("acceptance") or {}
    cells = report.get("cells") or []
    if int(report.get("envs_per_cell", 0)) < 64 or len(cells) < 4:
        return False
    for key in ("min_completion", "max_collision", "max_fallen"):
        if key not in acceptance:
            return False
    try:
        return all(cell_passes(cell, acceptance) for cell in cells)
    except (KeyError, TypeError, ValueError):
        return False


def certify_checkpoint(source, report_path, output=None):
    """Create a separately hashed checkpoint after an external evaluation actually passes."""
    source = Path(source)
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"validation report does not exist: {report_path}")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    if report.get("checkpoint_sha256") != file_hash(source):
        raise ValueError("validation report was produced for a different checkpoint")
    data = load_checkpoint(source)
    if data["stage"] == "stage1":
        measured_pass = _stage1_report_passed(report) or _stage1_gate_passed(report)
    elif data["stage"] == "stage2":
        measured_pass = (
            report.get("protocol") == "stage2-v1" and report.get("terrain") == "mixed"
            and int(report.get("episodes", 0)) >= 200
            and float(report.get("success_rate", -1)) >= 0.90
            and float(report.get("collision_rate", math.inf)) <= 0.05
        )
    else:
        raise ValueError("automatic capability certification is defined only for Stage 1 and Stage 2")
    if report.get("passed") is not True or not measured_pass:
        raise ValueError("failed or incomplete evaluation cannot certify a checkpoint")
    data["metadata"] = dict(data["metadata"])
    data["metadata"].update({
        "capability_validated": True,
        "validation_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "validation_protocol": report.get("protocol"),
    })
    if report.get("protocol") == "stage1-gate-v1":
        cells = report["cells"]
        data["metadata"]["envelope"] = {
            "acceptance": report["acceptance"],
            "difficulty_min": min(float(cell["difficulty"]) for cell in cells),
            "difficulty_max": max(float(cell["difficulty"]) for cell in cells),
            "speed_min": min(float(cell["speed_min"]) for cell in cells),
            "speed_max": max(float(cell["speed_max"]) for cell in cells),
            "cells": len(cells),
            "envs_per_cell": int(report["envs_per_cell"]),
            "episode_seconds": float(report["episode_seconds"]),
        }
    output = Path(output) if output else source.with_name(source.stem + ".validated.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(data, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output.with_suffix(".json"), {
        "format": data["format"], "stage": data["stage"], "interfaces": data["interfaces"],
        "model_hashes": data["model_hashes"], "metadata": data["metadata"],
        "checkpoint_sha256": file_hash(output),
    })
    # Reopen both layers so a partially written or inconsistent certification cannot escape.
    load_checkpoint(output, require_validated=True)
    return output


def load_actor(path, kind, device="cpu", require_validated=False):
    from .models import ActorCore, migrate_locomotion_heading_state
    from .hier_v2.model_v2 import EncoderActorCore, detect_actor_variant

    if kind not in {"locomotion", "navigation"}:
        raise ValueError(f"unknown actor kind: {kind}")
    data = load_checkpoint(
        path, "stage1" if kind == "locomotion" else "stage2", require_validated,
        allow_locomotion_heading_migration=(kind == "locomotion"),
    )
    if kind not in data["models"]:
        raise ValueError(f"checkpoint has no {kind} actor")
    state = data["models"][kind]
    if kind == "locomotion" and _is_schema3_heading_migration(data["interfaces"]):
        # A schema-3 policy predates the fixed-heading observation columns. The
        # schema-4 relief rename needs no state change: every dimension is equal.
        state = migrate_locomotion_heading_state(state)
    actor_class = ActorCore if detect_actor_variant(state) == "v1" else EncoderActorCore
    actor = actor_class(kind).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval().requires_grad_(False)
    return actor


def load_vision(path, device="cpu", require_validated=False):
    from .models import VisionModel

    data = load_checkpoint(path, require_validated=require_validated)
    if data["stage"] not in {"stage3a", "stage3b", "stage4"} or "vision" not in data["models"]:
        raise ValueError("checkpoint is not a perception model")
    model = VisionModel().to(device)
    model.load_state_dict(data["models"]["vision"], strict=True)
    return model
