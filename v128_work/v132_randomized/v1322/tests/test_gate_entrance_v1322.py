"""One audited seed, one immutable 512-slot entrance holdout, and no auto-junction."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class EntranceHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.gate = importlib.import_module('gate_entrance_v1322')
        self.production = importlib.import_module('production_v1322')
        self.preflight = importlib.import_module('preflight_v1322')
        self.spec = json.loads((ROOT / 'configs/gate_entrance_execution.json').read_text())
        self.config = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        self.manifest = json.loads((ROOT / 'configs/manifest.json').read_text())

    def rows(self, successes):
        rows = []
        for i in range(512):
            win = i < successes
            rows.append({'env_id': i, 'episode_index': 0, 'required_gap_count': 2,
                         'credited_gap_count': 2 if win else 0, 'success': win,
                         'fallen': not win, 'collision': False, 'out_of_bounds': False,
                         'fell_into_gap': False, 'course_x': float(i),
                         'reward_events': {'safe_segment': 2 if win else 0,
                                           'supported_short': 2 if win else 0}})
        report = {'episodes': 512, 'completion_rate': successes / 512,
                  'checkpoint_sha256': self.gate.CHECKPOINT_SHA,
                  'environments': {str(i): {'episodes': 1, 'success': int(row['success']),
                                           'fallen': int(row['fallen']), 'collision': 0,
                                           'out_of_bounds': 0, 'fell_into_gap': 0}
                                   for i, row in enumerate(rows)}}
        return report, rows

    def test_holdout_contract_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            paths = self.gate.artifacts(workspace, root, self.spec)
            paths['report'].parent.mkdir(parents=True)
            paths['trace'].parent.mkdir(parents=True)
            self.gate.validate_outputs(paths)
            cmd = self.gate.command(workspace, self.config, paths, self.spec)
            for option, value in (('--episodes', '512'), ('--num-envs', '512'),
                                  ('--seed', '1812341965'), ('--checkpoint', str(paths['checkpoint'])),
                                  ('--episode-trace', str(paths['trace'])),
                                  ('--speed-min', '0.35'), ('--speed-max', '0.8')):
                self.assertEqual(cmd[cmd.index(option) + 1], value)
            self.assertIn('--strict-first-episode', cmd)
            self.assertNotIn('train', cmd)
            for name in ('report', 'trace', 'log', 'receipt'):
                with self.subTest(name=name):
                    path = paths[name]
                    path.write_text('immutable prior result')
                    with self.assertRaisesRegex(ValueError, 'already exists'):
                        self.gate.validate_outputs(paths)
                    self.assertEqual(path.read_text(), 'immutable prior result')
                    path.unlink()
            bad = dict(self.spec, holdout_seed=2028)
            with self.assertRaises(ValueError):
                self.gate.command(workspace, self.config, paths, bad)

    def test_pinned_seed_audit_precedes_config_and_binds_original_training(self):
        self.assertEqual(self.gate.SEED, 1812341965)
        self.assertEqual(self.spec['seed_audit_sha256'], self.gate.AUDIT_SHA)
        self.assertEqual(self.gate.digest(ROOT / 'provenance/holdout_seed_1812341965_audit.txt'),
                         self.gate.AUDIT_SHA)
        receipt = ROOT / 'reports/go1-v1322-real-entrance-s01-p1-d1_full_train.receipt'
        self.assertEqual(self.gate.digest(receipt), self.spec['train_receipt_sha256'])
        self.assertEqual(self.spec['checkpoint_sha256'], self.gate.CHECKPOINT_SHA)
        self.assertIsNone(self.config['gate_holdout_seed'])
        self.assertIsNone(self.manifest['execution']['holdout_seed'])
        self.assertEqual(self.manifest['execution']['train_and_gate_env']['GO1_PRIVILEGED_ACTOR'], '1')

    def test_preflight_gate_allows_existing_train_outputs_not_existing_gate_outputs(self):
        baseline = json.loads((ROOT / 'provenance/v1321_teacher_entrance.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            trained = workspace / f"checkpoints/stage1/{self.spec['trained_run_id']}"
            trained.mkdir(parents=True)
            with patch.object(self.preflight, 'validate_live_baseline'), \
                    patch.object(self.preflight, 'validate_mode_outputs') as outputs:
                self.preflight.validate_components(workspace, ROOT, self.manifest,
                                                   self.config, baseline,
                                                   production=True, mode='gate')
                outputs.assert_called_once()
                self.assertEqual(outputs.call_args.args[1], 'gate')
                with self.assertRaisesRegex(ValueError, 'output path already exists'):
                    self.preflight.validate_components(workspace, ROOT, self.manifest,
                                                       self.config, baseline,
                                                       production=True, mode='train')

    def test_exact_461_threshold_and_gap_reward_hard_fail_closed(self):
        for count, passed in ((0, False), (460, False), (461, True), (512, True)):
            with self.subTest(count=count):
                report, rows = self.rows(count)
                self.assertEqual(self.gate.score(report, rows, self.spec)['gate_passed'], passed)
                self.assertEqual(self.gate.score(report, rows, self.spec)['first_episode_successes'], count)
        report, rows = self.rows(461)
        rows[0]['credited_gap_count'] = 1
        with self.assertRaises(ValueError):
            self.gate.score(report, rows, self.spec)
        rows[0]['credited_gap_count'] = 2
        rows[0]['reward_events']['safe_segment'] = 3
        with self.assertRaisesRegex(ValueError, 'without credited real gap'):
            self.gate.score(report, rows, self.spec)
        rows[0]['reward_events']['safe_segment'] = 1
        with self.assertRaisesRegex(ValueError, 'all real gaps'):
            self.gate.score(report, rows, self.spec)
        rows[0]['reward_events']['safe_segment'] = 2
        rows[0]['out_of_bounds'] = True
        with self.assertRaisesRegex(ValueError, 'hard safety'):
            self.gate.score(report, rows, self.spec)

    def test_corrupt_terminal_duplicate_missing_and_episode_sum_block(self):
        report, rows = self.rows(461)
        for mutate in ('duplicate', 'missing', 'terminal', 'episode_sum'):
            bad = json.loads(json.dumps(report)); trace = json.loads(json.dumps(rows))
            if mutate == 'duplicate':
                trace[-1]['env_id'] = 510
            elif mutate == 'missing':
                bad['environments'].pop('511')
            elif mutate == 'terminal':
                bad['environments']['0']['success'] = 0
            else:
                bad['environments']['0']['episodes'] = 2
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.preflight.validate_gate(bad, trace, 512, self.gate.CHECKPOINT_SHA)
        report['completion_rate'] = 0.2
        with self.assertRaisesRegex(ValueError, 'completion'):
            self.preflight.validate_gate(report, rows, 512, self.gate.CHECKPOINT_SHA)

    def test_gate_result_path_checks_trace_against_report_before_scoring(self):
        report, rows = self.rows(461)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            paths = self.gate.artifacts(workspace, workspace, self.spec)
            paths['report'].parent.mkdir(parents=True)
            calls = []
            def strict(report_path, trace_path, count, checkpoint, seed, run_id):
                calls.append((count, seed, run_id))
                return report
            fake_driver = types.SimpleNamespace(require_strict_gate_report=strict)
            with patch.dict(sys.modules, {'run_teacher_v1322': fake_driver}):
                def save(trace):
                    paths['report'].write_text(json.dumps(report))
                    paths['trace'].write_text(''.join(json.dumps(row) + '\n' for row in trace))
                save(rows)
                result = self.gate.validate_result(paths, self.spec)
                self.assertTrue(result['gate_passed'])
                self.assertEqual(calls[-1], (512, self.gate.SEED, self.gate.RUN_ID))
                for tamper in ('duplicate', 'missing', 'terminal'):
                    bad = json.loads(json.dumps(rows))
                    original = report['environments'].get('511')
                    if tamper == 'duplicate':
                        bad[-1]['env_id'] = 510
                    elif tamper == 'missing':
                        bad.pop()
                    else:
                        bad[0]['success'] = False
                    save(bad)
                    with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                        self.gate.validate_result(paths, self.spec)
                    report['environments']['511'] = original

    def test_real_dispatch_signature_and_mock_sim_produce_measured_failure_only(self):
        source = (ROOT / 'scripts/production_v1322.py').read_text()
        calls = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'run_entrance_gate']
        self.assertEqual(len(calls), 1)
        inspect.signature(self.production.run_entrance_gate).bind(*([object()] * len(calls[0].args)))
        self.assertEqual([node.id for node in calls[0].args],
                         ['workspace', 'root', 'manifest', 'config', 'gate_spec',
                          'environment', 'gate_paths', 'gate_inputs'])
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            paths = self.gate.artifacts(workspace, root, self.spec)
            paths['report'].parent.mkdir(parents=True)
            paths['log'].parent.mkdir(parents=True)
            record = root / 'reports' / f"{self.spec['trained_run_id']}_full_train.receipt"
            record.write_text('pinned receipt stub')
            audit = root / 'provenance/holdout_seed_1812341965_audit.txt'
            audit.parent.mkdir()
            audit.write_text('pinned audit stub')
            reward = root / self.manifest['reward_config']['path']
            reward.parent.mkdir(parents=True)
            reward.write_bytes((ROOT / self.manifest['reward_config']['path']).read_bytes())
            def run_sim(cmd, log, env, cwd):
                self.assertEqual(cmd[:2], [str(workspace / 'sim.sh'), 'evaluate'])
                self.assertEqual(env['GO1_TASK_PLAN'], str(paths['plan']))
                self.assertEqual(env['GO1_PRIVILEGED_ACTOR'], '1')
                self.assertEqual(env['GO1_RUN_ID'], f"gate-{self.spec['gate_run_id']}")
                self.assertEqual(cwd, workspace)
                log.write_text('mock only')
                paths['report'].write_text('{}')
                paths['trace'].write_text('{}\n')
            pinned = {paths['checkpoint']: self.spec['checkpoint_sha256'],
                      paths['plan']: self.spec['source_plan_sha256'],
                      record: self.spec['train_receipt_sha256'],
                      audit: self.spec['seed_audit_sha256']}
            original_digest = self.production.digest
            with patch.object(self.production, 'run_logged', side_effect=run_sim) as launched, \
                    patch.object(self.gate, 'validate_result', return_value={
                        'first_episode_count': 512, 'first_episode_successes': 460,
                        'first_episode_completion_rate': 460 / 512,
                        'minimum_successes': 461, 'gate_passed': False,
                        'reward_events': {'supported_short': 1, 'safe_segment': 1}}), \
                    patch.object(self.production, 'digest',
                                 side_effect=lambda path: pinned[path] if path in pinned else original_digest(path)):
                result = self.production.run_entrance_gate(
                    workspace, root, self.manifest, self.config, self.spec,
                    {'GO1_PRIVILEGED_ACTOR': '1',
                     'GO1_REWARD_CONFIG_PATH': str(workspace / self.manifest['reward_config']['path'])},
                    paths, {'train_receipt_sha256': self.gate.TRAIN_RECEIPT_SHA,
                            'seed_audit_sha256': self.gate.AUDIT_SHA})
            launched.assert_called_once()
            self.assertFalse(result['gate_passed'])
            self.assertFalse(result['automatic_junction_authorized'])
            self.assertTrue(result['final_holdout'])
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text()
        self.assertIn('preflight-gate', shell)
        self.assertIn('entrance-final-gate-only', shell)
        self.assertNotIn('junction-final-gate-only', shell)


if __name__ == '__main__':
    unittest.main()
