"""The full entrance train is manually authorized and never starts a gate."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class EntranceTrainTests(unittest.TestCase):
    def setUp(self):
        self.train = importlib.import_module('train_entrance_v1322')
        self.manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.config = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        self.spec = json.loads((ROOT / 'configs/train_entrance_execution.json').read_text())

    def test_command_is_only_fresh_entrance_training_with_original_start(self):
        workspace = Path('/config/Desktop/sim')
        plan = workspace / 'reports/lp_plan_go1-v1321-real-entrance-s01-p1-d1.json'
        checkpoint = workspace / self.config['init_checkpoint']
        command = self.train.command(workspace, self.config, self.spec, checkpoint)
        self.assertEqual(command[:3], ['bash', str(workspace / 'scripts/torchrun_sim.sh'), 'train'])
        self.assertEqual(self.spec['original_torchrun_launcher_sha256'], self.train.LAUNCHER_SHA)
        for option, expected in (('--num-envs', '512'), ('--iterations', '650'),
                                 ('--critic-warmup-iterations', '50'), ('--seed', '42'),
                                 ('--speed-min', '0.35'), ('--speed-max', '0.8'),
                                 ('--initialize-from', str(checkpoint))):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(self.spec['source_plan_sha256'], self.train.SOURCE_PLAN_SHA)
        self.assertEqual(self.spec['run_id'], self.manifest['execution']['run_id'])
        self.assertEqual(self.spec['pilot_receipt_sha256'], self.train.PILOT_SHA)
        self.assertNotIn('evaluate', command)
        self.assertNotIn('distill', command)
        self.assertNotIn('--strict-first-episode', command)
        self.assertIsNone(self.config['gate_holdout_seed'])
        self.assertNotEqual(str(plan), self.spec['run_id'])

    def test_fresh_training_outputs_and_old_checkpoint_are_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            (root / 'reports').mkdir(parents=True)
            paths = self.train.artifacts(workspace, root, self.spec)
            self.train.validate_outputs(paths)
            self.assertTrue(paths['checkpoint_dir'].name.startswith('go1-v1322-'))
            self.assertNotIn('v1321-real-entrance', str(paths['checkpoint_dir']))
            old = workspace / 'checkpoints/stage1/go1-v1321-real-entrance-s01-p1-d1/model_649.pt'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'old model remains')
            self.train.validate_outputs(paths)
            self.assertEqual(old.read_bytes(), b'old model remains')
            paths['receipt'].write_text('existing receipt')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.train.validate_outputs(paths)

    def test_pilot_gain_and_each_hard_category_required(self):
        first = dict(first_episode_count=64, first_episode_completion_rate=21 / 64,
                     supported_short_events=3481, safe_segment_events=101,
                     credited_gap_count=101, collision_rate=11 / 64,
                     fallen_rate=31 / 64, fell_into_gap_rate=11 / 64,
                     out_of_bounds_rate=0)
        second = dict(first, first_episode_completion_rate=27 / 64,
                      supported_short_events=3342, safe_segment_events=111,
                      credited_gap_count=111, collision_rate=10 / 64,
                      fallen_rate=26 / 64)
        record = dict(status='pilot-diagnostic-only', mode='pilot', restored=True,
                      final_holdout=False, automatic_full_training_authorized=False,
                      full_training_started=False, pilot_completion_delta=6 / 64,
                      safety_regression_observed=False,
                      arms={'control': {'evaluation_first_episode': first},
                            'treatment': {'evaluation_first_episode': second}})
        self.train.require_pilot_improvement(record)
        for key in ('collision_rate', 'fallen_rate', 'fell_into_gap_rate',
                    'out_of_bounds_rate'):
            bad = json.loads(json.dumps(record))
            bad['arms']['treatment']['evaluation_first_episode'][key] = first[key] + 1 / 64
            with self.subTest(flag=key), self.assertRaises(ValueError):
                self.train.require_pilot_improvement(bad)
        bad = json.loads(json.dumps(record))
        bad['arms']['treatment']['evaluation_first_episode']['first_episode_completion_rate'] = 21 / 64
        with self.assertRaises(ValueError):
            self.train.require_pilot_improvement(bad)

    def test_model_649_and_actual_positive_reward_required_for_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.train.artifacts(root, root / 'v128_work/v132_randomized/v1322', self.spec)
            paths['checkpoint_dir'].mkdir(parents=True)
            paths['log'].parent.mkdir(parents=True)
            paths['stage_report'].parent.mkdir(parents=True, exist_ok=True)
            (paths['checkpoint_dir'] / 'model_0.pt').write_bytes(b'early')
            with self.assertRaisesRegex(ValueError, 'model_649'):
                self.train.finished_checkpoint(paths)
            (paths['checkpoint_dir'] / 'model_649.pt').write_bytes(b'finished')
            self.assertEqual(self.train.finished_checkpoint(paths).name, 'model_649.pt')
            paths['log'].write_text('Episode_Reward/supported_short: 0.0000\n')
            with self.assertRaisesRegex(ValueError, 'supported_short'):
                self.train.reward_activation(paths['log'])
            paths['log'].write_text('Episode_Reward/supported_short: 0.0002\n'
                                    'Episode_Reward/safe_segment: 0.0004\n')
            self.assertGreater(self.train.reward_activation(paths['log'])['supported_short'], 0)

    def test_successful_train_dispatch_records_only_unverified_teacher(self):
        production = importlib.import_module('production_v1322')
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            root.mkdir(parents=True)
            paths = self.train.artifacts(workspace, root, self.spec)
            checkpoint = workspace / 'checkpoints/original/model_299.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'original stub')
            plan = workspace / 'reports/original_plan.json'
            plan.parent.mkdir(parents=True)
            plan.write_text('pinned stub plan')
            reward = root / self.manifest['reward_config']['path']
            reward.parent.mkdir(parents=True)
            reward.write_bytes((ROOT / self.manifest['reward_config']['path']).read_bytes())
            expected = {checkpoint: self.spec['initial_checkpoint_sha256'],
                        plan: self.spec['source_plan_sha256']}
            original_digest = production.digest
            def pinned_digest(path):
                return expected.get(path, original_digest(path))
            def fake_run(command, log, environment, cwd):
                self.assertEqual(command[:3], ['bash', str(workspace / 'scripts/torchrun_sim.sh'), 'train'])
                self.assertEqual(environment['GO1_RUN_ID'], self.spec['run_id'])
                self.assertEqual(environment['NPROC_PER_NODE'], '1')
                self.assertEqual(environment['GO1_DDP_BACKEND'], 'gloo')
                self.assertEqual(environment['GO1_TASK_PLAN'], str(plan))
                self.assertNotIn('GO1_AUTO_P2', environment)
                self.assertNotIn('GO1_V29_EPS', environment)
                paths['checkpoint_dir'].mkdir(parents=True)
                (paths['checkpoint_dir'] / 'model_649.pt').write_bytes(b'new teacher stub')
                paths['stage_report'].write_text(json.dumps({
                    'run': self.spec['run_id'], 'iterations': 650, 'task_plan': str(plan),
                    'episodes': 9, 'completion_rate': 0.1,
                    'environments': {str(i): {} for i in range(512)},
                }))
                log.parent.mkdir(parents=True)
                log.write_text('Episode_Reward/supported_short: 0.0002\n'
                               'Episode_Reward/safe_segment: 0.0004\n')
            with patch.object(production, 'digest', side_effect=pinned_digest), \
                    patch.object(production, 'run_logged', side_effect=fake_run) as launched:
                result = production.run_entrance_train(
                    workspace, root, self.manifest, self.config, self.spec,
                    {'GO1_V29_EPS': 'unsafe'}, paths,
                    {'checkpoint': checkpoint, 'plan': plan, 'pilot_receipt_sha256': self.train.PILOT_SHA,
                     'diagnosis_receipt_sha256': 'diagnostic',
                     'test_receipt_sha256': 'test', 'smoke_receipt_sha256': 'smoke'})
            launched.assert_called_once()
            self.assertFalse(result['final_gate_passed'])
            self.assertFalse(result['automatic_gate_authorized'])
            self.assertFalse(result['automatic_junction_authorized'])
            self.assertEqual(result['checkpoint_path'], str(paths['checkpoint_dir'] / 'model_649.pt'))
            self.assertEqual(result['actor_updates'], 600)

    def test_preflight_and_dispatch_behaviour_preserve_no_auto_gate(self):
        production = importlib.import_module('production_v1322')
        self.assertIn('train', production.run.__code__.co_consts)
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text()
        preflight = (ROOT / 'scripts/preflight_v1322.py').read_text()
        self.assertIn('preflight-train', shell)
        self.assertIn('entrance-full-train-only', shell)
        self.assertIn('preflight-train', preflight)
        self.assertNotIn('auto-gate', shell)
        source = (ROOT / 'scripts/production_v1322.py').read_text()
        calls = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'run_entrance_train']
        self.assertEqual(len(calls), 1)
        inspect.signature(production.run_entrance_train).bind(*([object()] * len(calls[0].args)))
        self.assertEqual([node.id for node in calls[0].args],
                         ['workspace', 'root', 'manifest', 'config', 'train_spec',
                          'environment', 'train_paths', 'train_inputs'])
        self.assertIn("GO1_DDP_BACKEND='gloo'", source)
        self.assertIn("NPROC_PER_NODE='1'", source)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            root.mkdir(parents=True)
            paths = self.train.artifacts(workspace, root, self.spec)
            with patch.object(production, 'run_logged') as logged:
                # The dispatcher passes every required train argument; failures in
                # a mocked train cannot silently turn into pilot or gate execution.
                with self.assertRaisesRegex(ValueError, 'no checkpoint'):
                    production.run_entrance_train(workspace, root, self.manifest,
                                                  self.config, self.spec, {}, paths, {})
                logged.assert_not_called()


if __name__ == '__main__':
    unittest.main()
