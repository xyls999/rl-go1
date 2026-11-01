"""Development-only replay must not become a final holdout or start training."""
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

    def test_old_gate_plan_requires_real_entrance_and_matched_speed_band(self):
        diagnosis = importlib.import_module('diagnostic_v1322')
        with tempfile.TemporaryDirectory() as temp:
            paths = diagnosis.artifacts(Path(temp), self.manifest)
            paths['plan'].parent.mkdir(parents=True)
            plan = {'families': ['combined_part_entrance'] * 512,
                    'difficulty': [0.1] * 512, 'speed_min': .35, 'speed_max': .8,
                    'instances': [['combined_part_entrance', 1, 2] for _ in range(512)]}
            with self.assertRaisesRegex(ValueError, 'plan missing'):
                diagnosis.validate_plan(paths, self.config)
            paths['plan'].write_text(json.dumps(plan))
            self.assertEqual(diagnosis.validate_plan(paths, self.config), preflight.digest(paths['plan']))
            for field, changed in (('families', ['combined'] + plan['families'][1:]),
                                   ('difficulty', [.7] + plan['difficulty'][1:]),
                                   ('speed_min', .30),
                                   ('instances', [['combined', 1, 2]] + plan['instances'][1:])):
                tampered = {**plan, field: changed}
                paths['plan'].write_text(json.dumps(tampered))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnosis.validate_plan(paths, self.config)

    def test_shell_and_preflight_opt_in_and_no_final_holdout(self):
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text()
        self.assertIn('preflight-diagnose', shell)
        self.assertIn('dev-diagnose-only', shell)
        self.assertIn('diagnose', shell)
        self.assertNotIn('train &&', shell)
        self.assertIn('validate_diagnosis_inputs', (ROOT / 'scripts/preflight_v1322.py').read_text())
        self.assertIn('require_strict_gate_report', (ROOT / 'scripts/production_v1322.py').read_text())
        self.assertIn('validate_gate', (ROOT / 'scripts/production_v1322.py').read_text())
        self.assertIsNone(self.config['gate_holdout_seed'])


if __name__ == '__main__':
    unittest.main()
