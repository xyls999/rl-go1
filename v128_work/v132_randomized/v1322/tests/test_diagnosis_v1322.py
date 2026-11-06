"""Development-only replay must not become a final holdout or start training."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import preflight_v1322 as preflight
import production_v1322 as production


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.config = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())

    def test_explicit_dev_mode_and_command_never_train_or_certify(self):
        diagnosis = importlib.import_module('diagnostic_v1322')
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            artifacts = diagnosis.artifacts(workspace, self.manifest)
            command = diagnosis.command(workspace, self.config, artifacts)
            self.assertEqual(command[1], 'evaluate')
            self.assertEqual(command[command.index('--num-envs') + 1], '512')
            self.assertEqual(command[command.index('--episodes') + 1], '512')
            self.assertEqual(command[command.index('--seed') + 1], '2028')
            self.assertIn('--strict-first-episode', command)
            self.assertEqual(command[command.index('--episode-trace') + 1], str(artifacts['trace']))
            self.assertEqual(command[command.index('--checkpoint') + 1], str(artifacts['checkpoint']))
            self.assertEqual(command[command.index('--speed-min') + 1], '0.35')
            self.assertEqual(command[command.index('--speed-max') + 1], '0.8')
            self.assertNotIn('train', command)
            self.assertNotIn('--certify', command)
            self.assertNotIn(self.config['gate_holdout_seed'], (2028,))
            self.assertIn('dev-v1321', artifacts['run_id'])
            self.assertEqual(artifacts['report'].name, f"evaluate_stage1_gate-{artifacts['run_id']}.json")
            self.assertTrue(all('v1322' in str(artifacts[key]) for key in ('trace', 'log', 'receipt')))

    def test_frozen_receipt_lineage_rejects_missing_tampered_or_failed(self):
        diagnosis = importlib.import_module('diagnostic_v1322')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'v128_work/v132_randomized/v1322'
            (root / 'configs').mkdir(parents=True)
            (root / 'reports').mkdir()
            manifest_bytes = (ROOT / 'configs/manifest.json').read_bytes()
            (root / 'configs/manifest.json').write_bytes(manifest_bytes)
            with self.assertRaisesRegex(ValueError, 'test receipt'):
                diagnosis.validate_receipts(root, root.parents[2], self.manifest)
            names = production.attempt_artifacts(self.manifest['execution']['test_attempt'])
            for key in ('test_log', 'focus_log'):
                (root / 'reports' / names[key]).write_bytes(key.encode())
            common = {'status': 'passed', 'restored': True,
                      'manifest_sha256': preflight.digest(root / 'configs/manifest.json'),
                      'reward_sha256': self.manifest['reward_config']['sha256']}
            test = {**common, 'mode': 'test',
                    'log_sha256': preflight.digest(root / 'reports' / names['test_log']),
                    'focus_sha256': preflight.digest(root / 'reports' / names['focus_log'])}
            (root / 'reports' / names['test_receipt']).write_text(json.dumps(test))
            with self.assertRaisesRegex(ValueError, 'PhysX receipt'):
                diagnosis.validate_receipts(root, root.parents[2], self.manifest)
            for family, _ in production.FAMILIES:
                (root / 'reports' / f"{names['smoke_prefix']}{family}.log").write_text(family)
                report = root.parents[2] / 'reports' / (f"check_v1322-{self.manifest['execution']['run_id']}-"
                                                          f"a03-{family}.json")
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(family)
            families = {}
            for family, _ in production.FAMILIES:
                families[family] = {
                    'log_sha256': preflight.digest(root / 'reports' / f"{names['smoke_prefix']}{family}.log"),
                    'report_sha256': preflight.digest(root.parents[2] / 'reports' /
                        f"check_v1322-{self.manifest['execution']['run_id']}-a03-{family}.json")}
            receipt = root / 'reports' / names['smoke_receipt']
            receipt.write_text(json.dumps({**common, 'mode': 'smoke', 'families': families}))
            diagnosis.validate_receipts(root, root.parents[2], self.manifest)
            receipt.write_text(json.dumps({**common, 'mode': 'smoke', 'families': {**families, 'tire':
                {**families['tire'], 'report_sha256': '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, 'PhysX'):
                diagnosis.validate_receipts(root, root.parents[2], self.manifest)
            receipt.write_text(json.dumps({**common, 'mode': 'smoke', 'families': families}))
            (root / 'configs/manifest.json').write_bytes(manifest_bytes + b' ')
            with self.assertRaisesRegex(ValueError, 'frozen'):
                diagnosis.validate_receipts(root, root.parents[2], self.manifest)

    def test_existing_trace_or_report_fails_before_overlay(self):
        diagnosis = importlib.import_module('diagnostic_v1322')
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            artifacts = diagnosis.artifacts(workspace, self.manifest)
            for key in ('report', 'trace', 'log', 'receipt'):
                artifacts[key].parent.mkdir(parents=True, exist_ok=True)
                artifacts[key].write_text('old evidence')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'already exists'):
                    diagnosis.validate_outputs(artifacts)
                artifacts[key].unlink()

    def test_old_gate_plan_requires_pinned_provenance_and_complete_slot_metadata(self):
        diagnosis = importlib.import_module('diagnostic_v1322')
        source = ROOT / 'provenance/v1321_gate_plan.json'
        expected_sha256 = 'e16717f32563c9a1a9db72316a6eb85a51b557975080ae7c9066a71494f6e18b'
        self.assertEqual(preflight.digest(source), expected_sha256)
        with tempfile.TemporaryDirectory() as temp:
            paths = diagnosis.artifacts(Path(temp), self.manifest)
            paths['plan'].parent.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'plan missing'):
                diagnosis.validate_plan(paths, self.config)
            paths['plan'].write_bytes(source.read_bytes())
            self.assertEqual(diagnosis.validate_plan(paths, self.config), expected_sha256)
            original = json.loads(source.read_text(encoding='utf-8'))
            for field in ('instances', 'speed_level', 'difficulty_level', 'allowed_speed_levels'):
                tampered = deepcopy(original)
                del tampered[field]
                paths['plan'].write_text(json.dumps(tampered))
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    diagnosis.validate_plan(paths, self.config)
            for field, changed in (
                    ('families', ['combined'] + original['families'][1:]),
                    ('difficulty', [.7] + original['difficulty'][1:]),
                    ('difficulty_level', [3] + original['difficulty_level'][1:]),
                    ('speed_level', [4 if original['speed_level'][0] != 4 else 0] + original['speed_level'][1:]),
                    ('allowed_speed_levels', [0, 1, 2, 3, 5]),
                    ('instances', [['combined', 0, 2]] + original['instances'][1:]),
                    ('speed_min', .30),
                    ('temperature', 100.0),
            ):
                tampered = deepcopy(original)
                tampered[field] = changed
                paths['plan'].write_text(json.dumps(tampered))
                with self.subTest(changed=field), self.assertRaises(ValueError):
                    diagnosis.validate_plan(paths, self.config)
            paths['plan'].write_bytes(source.read_bytes())
            tampered = deepcopy(original)
            tampered['instances'][0][2] = 4
            paths['plan'].write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                diagnosis.validate_plan(paths, self.config)

    def test_shell_and_preflight_opt_in_without_automatic_next_teacher(self):
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text()
        self.assertIn('preflight-diagnose', shell)
        self.assertIn('dev-diagnose-only', shell)
        self.assertIn('diagnose', shell)
        self.assertIn("[[ \"$mode\" != train ]] || authorization='entrance-full-train-only'", shell)
        self.assertNotIn('junction-final-gate-only', shell)
        self.assertIn('validate_diagnosis_inputs', (ROOT / 'scripts/preflight_v1322.py').read_text())
        self.assertIn('require_strict_gate_report', (ROOT / 'scripts/production_v1322.py').read_text())
        self.assertIn('validate_gate', (ROOT / 'scripts/production_v1322.py').read_text())
        self.assertIsNone(self.config['gate_holdout_seed'])


if __name__ == '__main__':
    unittest.main()
