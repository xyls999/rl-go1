"""Gate records: multi-cell acceptance must both certify and refuse correctly."""

import tempfile
import unittest
from pathlib import Path

import torch

from go1_hier_sim.checkpoints import certify_checkpoint, load_checkpoint, save_checkpoint
from go1_hier_sim.gate import (
    DEFAULT_ACCEPTANCE,
    build_gate_record,
    main as gate_main,
    parse_gate_log,
    write_gate_record,
)
from go1_hier_sim.models import ActorCore


LOG_TEXT = """checkpoint: checkpoints/stage1/run/model_699.pt
episodes/cell: 256  envs/cell: 256  device: cuda:1  episode-seconds: 120
speed    difficulty completion     fallen     oob        collision  velocity_rmse verdict
0.3      0.25     0.9336         0.0391     0.0000     0.0312     0.2996     PASS
1.0      1.0      0.9609         0.0078     0.0000     0.0312     0.1864     PASS
"""


def passing_cell(speed, difficulty, completion=0.95, fallen=0.0, collision=0.0):
    return {
        "speed": speed,
        "difficulty": difficulty,
        "completion_rate": completion,
        "fallen_rate": fallen,
        "out_of_bounds_rate": 0.0,
        "collision_rate": collision,
        "velocity_xy_rmse": 0.2,
    }


class GateRecordTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_gate_log_rows_are_parsed_without_the_header(self):
        cells = parse_gate_log(LOG_TEXT)
        self.assertEqual(len(cells), 2)
        self.assertAlmostEqual(cells[0]["speed"], 0.3)
        self.assertAlmostEqual(cells[1]["difficulty"], 1.0)
        self.assertAlmostEqual(cells[1]["completion_rate"], 0.9609)

    def test_record_fails_when_any_cell_misses_the_acceptance_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "stage1.pt"
            save_checkpoint(checkpoint, "stage1", {"locomotion": ActorCore("locomotion")})
            cells = [
                passing_cell(0.3, 0.25),
                passing_cell(1.0, 1.0, completion=0.71, fallen=0.16, collision=0.13),
            ]
            record = build_gate_record(
                checkpoint, cells, envs_per_cell=256, episode_seconds=120
            )
            self.assertFalse(record["passed"])
            self.assertEqual(record["acceptance"], DEFAULT_ACCEPTANCE)
            self.assertAlmostEqual(record["cells"][0]["speed_min"], 0.2)
            self.assertAlmostEqual(record["cells"][1]["speed_max"], 1.0)
            path = write_gate_record(Path(folder) / "gate.json", record)
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                certify_checkpoint(checkpoint, path)

    def test_all_passing_cells_certify_and_record_the_envelope(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "stage1.pt"
            save_checkpoint(checkpoint, "stage1", {"locomotion": ActorCore("locomotion")})
            cells = [
                passing_cell(speed, difficulty)
                for speed in (0.3, 1.0)
                for difficulty in (0.25, 1.0)
            ]
            record = build_gate_record(checkpoint, cells, envs_per_cell=256, episode_seconds=120)
            self.assertTrue(record["passed"])
            report = write_gate_record(folder / "gate.json", record)
            certified = certify_checkpoint(checkpoint, report)
            data = load_checkpoint(certified, "stage1", require_validated=True)
            self.assertTrue(data["metadata"]["capability_validated"])
            envelope = data["metadata"]["envelope"]
            self.assertAlmostEqual(envelope["difficulty_min"], 0.25)
            self.assertAlmostEqual(envelope["difficulty_max"], 1.0)
            self.assertAlmostEqual(envelope["speed_min"], 0.2)
            self.assertAlmostEqual(envelope["speed_max"], 1.0)
            self.assertEqual(envelope["cells"], 4)
            self.assertEqual(envelope["envs_per_cell"], 256)

    def test_record_for_a_different_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "stage1.pt"
            save_checkpoint(checkpoint, "stage1", {"locomotion": ActorCore("locomotion")})
            record = build_gate_record(
                checkpoint, [passing_cell(0.3, 0.25)] * 4, envs_per_cell=256,
                episode_seconds=120,
            )
            other = folder / "other.pt"
            save_checkpoint(other, "stage1", {"locomotion": ActorCore("locomotion")})
            record["checkpoint_sha256"] = "0" * 64
            path = write_gate_record(folder / "gate.json", record)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                certify_checkpoint(other, path)

    def test_recorder_cli_writes_a_certifiable_record(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "stage1.pt"
            save_checkpoint(checkpoint, "stage1", {"locomotion": ActorCore("locomotion")})
            log = folder / "gate.log"
            log.write_text(LOG_TEXT, encoding="utf-8")
            cells_file = folder / "cells.jsonl"
            cells_file.write_text(
                "\n".join(
                    __import__("json").dumps(passing_cell(speed, difficulty))
                    for speed in (0.3, 1.0)
                    for difficulty in (0.25, 1.0)
                ) + "\n",
                encoding="utf-8",
            )
            output = folder / "record.json"
            self.assertEqual(gate_main([
                "--checkpoint", str(checkpoint), "--log", str(log),
                "--cells", str(cells_file), "--output", str(output),
            ]), 0)
            data = load_checkpoint(certify_checkpoint(checkpoint, output), "stage1",
                                   require_validated=True)
            self.assertEqual(data["metadata"]["envelope"]["cells"], 4)


if __name__ == "__main__":
    unittest.main()
