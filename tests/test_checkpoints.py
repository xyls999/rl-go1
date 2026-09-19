import json
import copy
import unittest

import torch

from go1_hier_sim.checkpoints import certify_checkpoint, file_hash, load_checkpoint, save_checkpoint
from go1_hier_sim.contracts import interface_manifest


class CheckpointTests(unittest.TestCase):
    def test_only_exact_schema3_heading_migration_is_allowed(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            save_checkpoint(path, "stage1", {"test": torch.nn.Linear(3, 2)})
            payload = torch.load(path, map_location="cpu", weights_only=True)
            legacy = copy.deepcopy(interface_manifest())
            legacy["schema_version"] = 3
            legacy["locomotion"]["input"] = 1071
            legacy["locomotion"]["critic"] = 1078
            legacy["normalization"].pop("initial_heading")
            payload["interfaces"] = legacy
            torch.save(payload, path)
            manifest_path = path.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text())
            manifest["interfaces"] = legacy
            manifest["checkpoint_sha256"] = file_hash(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint(path, "stage1")
            loaded = load_checkpoint(
                path, "stage1", allow_locomotion_heading_migration=True
            )
            self.assertEqual(loaded["interfaces"]["schema_version"], 3)

    def test_hash_and_capability_are_enforced(self):
        with self.subTest():
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.pt"
                save_checkpoint(path, "stage1", {"test": torch.nn.Linear(3, 2)})
                data = load_checkpoint(path, "stage1")
                self.assertFalse(data["metadata"]["capability_validated"])
                with self.assertRaisesRegex(ValueError, "capability"):
                    load_checkpoint(path, require_validated=True)
                manifest = json.loads(path.with_suffix(".json").read_text())
                manifest["checkpoint_sha256"] = "wrong"
                path.with_suffix(".json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_checkpoint(path)


    def test_nonfinite_and_missing_checkpoint_fail(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(root / "missing.pt")
            model = torch.nn.Linear(1, 1)
            with torch.no_grad():
                model.weight.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                save_checkpoint(root / "bad.pt", "stage1", {"test": model})
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                save_checkpoint(root / "forged.pt", "stage1", {"test": torch.nn.Linear(1, 1)},
                                metadata={"capability_validated": True,
                                          "validation_report_sha256": "not-a-report"})

    def test_certification_requires_matching_passing_report(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "model.pt"
            report = root / "report.json"
            save_checkpoint(path, "stage1", {"test": torch.nn.Linear(2, 1)})
            report.write_text(json.dumps({"checkpoint_sha256": file_hash(path),
                                          "passed": False, "protocol": "stage1-v1"}))
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                certify_checkpoint(path, report)
            report.write_text(json.dumps({"checkpoint_sha256": file_hash(path), "passed": True,
                                          "protocol": "stage1-v2", "terrain": "course", "episodes": 400,
                                          "completion_rate": .95, "velocity_xy_rmse": .15, "yaw_rmse": .25}))
            certified = certify_checkpoint(path, report)
            self.assertTrue(load_checkpoint(certified, require_validated=True)["metadata"]["capability_validated"])


if __name__ == "__main__":
    unittest.main()
