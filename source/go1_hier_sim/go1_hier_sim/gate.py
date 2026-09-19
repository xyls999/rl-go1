"""Multi-cell evaluation records that can certify a frozen Stage-1 policy.

A single report cannot describe a policy that has to hold across a range of
speeds and terrain difficulties, and ``certify_checkpoint`` used to demand a
``terrain == "course"`` report with ``passed`` set, while ``passed`` is only ever
computed for ``terrain == "mixed"`` - so Stage-1 certification was unreachable.
A gate record carries one row per (speed, difficulty) cell plus the acceptance
rule that was agreed with the user, and certification requires every cell to
hold.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

GATE_PROTOCOL = "stage1-gate-v1"
GATE_CELL_TOLERANCE = 0.1

DEFAULT_ACCEPTANCE = {"min_completion": 0.90, "max_collision": 0.10, "max_fallen": 0.05}

_ROW = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
)


def parse_gate_log(text: str) -> list[dict]:
    """Parse the ``speed difficulty completion fallen oob collision vel`` rows."""
    cells = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        speed, difficulty, completion, fallen, out_of_bounds, collision, velocity = (
            float(value) for value in match.groups()
        )
        cells.append({
            "speed": speed,
            "difficulty": difficulty,
            "completion_rate": completion,
            "fallen_rate": fallen,
            "out_of_bounds_rate": out_of_bounds,
            "collision_rate": collision,
            "velocity_xy_rmse": velocity,
        })
    return cells


def cell_bounds(speed: float, max_speed: float) -> tuple[float, float]:
    """Reproduce the band the gate script evaluates for one nominal speed."""
    return max(0.0, speed - GATE_CELL_TOLERANCE), min(max_speed, speed + GATE_CELL_TOLERANCE)


def cell_passes(cell: dict, acceptance: dict) -> bool:
    return (
        float(cell["completion_rate"]) >= float(acceptance["min_completion"])
        and float(cell["collision_rate"]) <= float(acceptance["max_collision"])
        and float(cell["fallen_rate"]) <= float(acceptance["max_fallen"])
    )


def build_gate_record(checkpoint, cells, *, envs_per_cell, episode_seconds,
                      acceptance=None, max_speed=1.0, log_bytes: bytes | None = None) -> dict:
    """Assemble a certifiable record from measured cells."""
    if not cells:
        raise ValueError("a gate record needs at least one measured cell")
    acceptance = dict(acceptance or DEFAULT_ACCEPTANCE)
    checkpoint = Path(checkpoint)
    entry = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    detailed = []
    for cell in cells:
        speed = float(cell["speed"])
        low, high = cell_bounds(speed, max_speed)
        detailed.append({
            **{key: cell[key] for key in (
                "completion_rate", "fallen_rate", "out_of_bounds_rate",
                "collision_rate", "velocity_xy_rmse")},
            "difficulty": float(cell["difficulty"]),
            "speed": speed,
            "speed_min": low,
            "speed_max": high,
        })
    record = {
        "protocol": GATE_PROTOCOL,
        "stage": "stage1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": entry,
        "envs_per_cell": int(envs_per_cell),
        "episode_seconds": float(episode_seconds),
        "acceptance": acceptance,
        "cells": detailed,
    }
    if log_bytes is not None:
        record["gate_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
    record["passed"] = all(cell_passes(cell, acceptance) for cell in detailed)
    return record


def write_gate_record(path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log", help="gate stdout log to parse")
    parser.add_argument("--cells", help="jsonl of measured cells (preferred for new runs)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--envs-per-cell", type=int, default=256)
    parser.add_argument("--episode-seconds", type=float, default=120.0)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--min-completion", type=float, default=DEFAULT_ACCEPTANCE["min_completion"])
    parser.add_argument("--max-collision", type=float, default=DEFAULT_ACCEPTANCE["max_collision"])
    parser.add_argument("--max-fallen", type=float, default=DEFAULT_ACCEPTANCE["max_fallen"])
    args = parser.parse_args(argv)

    log_bytes = None
    if args.cells:
        cells = []
        for line in Path(args.cells).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cells.append(json.loads(line))
    elif args.log:
        log_bytes = Path(args.log).read_bytes()
        cells = parse_gate_log(log_bytes.decode("utf-8", errors="replace"))
    else:
        raise ValueError("either --log or --cells is required")

    acceptance = {
        "min_completion": args.min_completion,
        "max_collision": args.max_collision,
        "max_fallen": args.max_fallen,
    }
    record = build_gate_record(
        args.checkpoint, cells, envs_per_cell=args.envs_per_cell,
        episode_seconds=args.episode_seconds, acceptance=acceptance,
        max_speed=args.max_speed, log_bytes=log_bytes,
    )
    write_gate_record(args.output, record)
    print(f"GATE RECORD: {args.output} cells={len(record['cells'])} passed={record['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
