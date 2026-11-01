"""A paired pilot must preserve every input but the supported-short TOML weight."""
import ast
import importlib
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.pilot = importlib.import_module('pilot_v1322')
        self.manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.config = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        self.spec = json.loads((ROOT / 'configs/pilot_execution.json').read_text())

    def test_immutable_plan_and_only_one_reward_weight_changes(self):
        plan = self.pilot.build_plan(ROOT, self.config, self.spec)
        self.assertEqual(len(plan['families']), 64)
        self.assertEqual(set(plan['families']), {'combined_part_entrance'})
        self.assertEqual(len(plan['instances']), 64)
        self.assertEqual(self.pilot.digest_bytes(self.pilot.json_bytes(plan)), self.spec['plan_sha256'])
        treatment = (ROOT / self.manifest['reward_config']['path']).read_bytes()
        control = self.pilot.control_toml(treatment)
        self.pilot.validate_tomls(control, treatment, self.spec, self.manifest)
        self.assertEqual(self.pilot.digest_bytes(control), self.spec['control_toml_sha256'])
        self.assertEqual(self.pilot.digest_bytes(treatment), self.spec['treatment_toml_sha256'])
        with self.assertRaises(ValueError):
            self.pilot.validate_tomls(control.replace(b'weight = 6.0', b'weight = 0.0', 1),
                                      treatment, self.spec, self.manifest)

    def test_output_symlink_and_existing_artifacts_fail_before_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            root.mkdir(parents=True)
            real_reports = workspace / 'outside'
            real_reports.mkdir()
            try:
                (root / 'reports').symlink_to(real_reports, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('symlinks unavailable on this host')
            paths = self.pilot.artifacts(workspace, root, self.spec)
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                self.pilot.validate_outputs(paths)
            (root / 'reports').unlink()
            (root / 'reports').mkdir()
            paths['receipt'].write_text('old')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.pilot.validate_outputs(paths)

    def test_commands_match_except_run_and_reward_path(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = workspace / 'pilot_plan.json'
            source = workspace / 'model_299.pt'
            for arm in ('control', 'treatment'):
                command = self.pilot.train_command(workspace, self.config, self.spec, arm, plan, source)
                self.assertEqual(command[1], 'train')
                self.assertEqual(command[command.index('--num-envs') + 1], '64')
                self.assertEqual(command[command.index('--iterations') + 1], '30')
                self.assertEqual(command[command.index('--critic-warmup-iterations') + 1], '5')
                self.assertEqual(command[command.index('--seed') + 1], '42')
                self.assertEqual(command[command.index('--initialize-from') + 1], str(source))
                self.assertEqual(command[command.index('--speed-min') + 1], '0.35')
                self.assertEqual(command[command.index('--speed-max') + 1], '0.8')
                self.assertNotIn('--strict-first-episode', command)
                self.assertNotIn('distill', command)
                evaluation = self.pilot.eval_command(workspace, self.config, self.spec, arm,
                                                      workspace / 'trained.pt', workspace / 'trace.jsonl')
                self.assertEqual(evaluation[1], 'evaluate')
                self.assertEqual(evaluation[evaluation.index('--num-envs') + 1], '64')
                self.assertEqual(evaluation[evaluation.index('--episodes') + 1], '256')
                self.assertEqual(evaluation[evaluation.index('--seed') + 1], '2028')
                for option in ('--course-difficulty-min', '--course-difficulty-max',
                               '--speed-min', '--speed-max', '--episode-seconds'):
                    self.assertEqual(evaluation[evaluation.index(option) + 1],
                                     command[command.index(option) + 1])
                self.assertNotIn('--strict-first-episode', evaluation)
            left = self.pilot.train_command(workspace, self.config, self.spec, 'control', plan, source)
            right = self.pilot.train_command(workspace, self.config, self.spec, 'treatment', plan, source)
            left[left.index('--run-name') + 1] = right[right.index('--run-name') + 1]
            self.assertEqual(left, right)

    def test_pilot_authorization_and_receipt_lineage_are_separate_from_holdout(self):
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text(encoding='utf-8')
        preflight = (ROOT / 'scripts/preflight_v1322.py').read_text(encoding='utf-8')
        runner = (ROOT / 'scripts/production_v1322.py').read_text(encoding='utf-8')
        self.assertIn('paired-pilot-only', shell)
        self.assertIn('preflight-pilot', shell)
        self.assertIn("mode == 'pilot'", preflight)
        self.assertIn("mode == 'pilot'", runner)
        self.assertIn('pilot_v1322.validate_inputs', runner)
        self.assertIn('automatic_full_training_authorized', runner)
        self.assertEqual(self.spec['frozen_manifest_sha256'], self.pilot.BASE_SHA)
        self.assertEqual(self.spec['diagnosis_receipt_sha256'], self.pilot.DIAG_SHA)
        self.assertFalse(self.spec['final_holdout'])
        self.assertIsNone(self.config['gate_holdout_seed'])
        self.assertEqual(self.spec['critic_warmup_iterations'], 5)
        self.assertEqual(self.spec['iterations'], 30)
        self.assertEqual(self.spec['evaluation_episode_quota'], 256)
        self.assertEqual(self.spec['pilot_id'], 'go1-v1322-supportedshort-pilot-a02')
        self.assertEqual(self.spec['run_ids'], {
            arm: f"{self.spec['pilot_id']}-{arm}" for arm in ('control', 'treatment')})

    def test_missing_measurements_fail_closed(self):
        with self.assertRaises(ValueError):
            self.pilot.summarize({'episodes': 0, 'environments': {}}, [], 'control', 64)
        rows = [{'env_id': n, 'episode_index': 0, 'success': False, 'fallen': True,
                 'collision': False, 'out_of_bounds': False, 'fell_into_gap': False,
                 'required_gap_count': 2, 'credited_gap_count': 0, 'course_x': 3.0,
                 'reward_events': {'supported_short': 3, 'safe_segment': 0}}
                for n in range(64)]
        report = {'episodes': 256, 'environments': {str(n): {'episodes': 1} for n in range(64)}}
        result = self.pilot.summarize(report, rows, 'treatment', 64)
        self.assertEqual(result['first_episode_completion_rate'], 0)
        self.assertEqual(result['supported_short_events'], 192)
        self.assertEqual(result['fallen_rate'], 1)
        self.assertEqual(result['mean_failure_x'], 3)
        rows[0]['reward_events'].pop('supported_short')
        with self.assertRaises(ValueError):
            self.pilot.summarize(report, rows, 'treatment', 64)
        rows[0]['reward_events']['supported_short'] = 3
        rows.append({**rows[0], 'episode_index': 1})
        with self.assertRaises(ValueError):
            self.pilot.summarize(report, rows, 'treatment', 64)

    def test_first_episode_summary_ignores_fast_repeat_before_slot_coverage(self):
        first = [{'env_id': n, 'episode_index': 0, 'success': n == 63,
                  'fallen': n != 63, 'collision': False, 'out_of_bounds': False,
                  'fell_into_gap': False, 'required_gap_count': 2,
                  'credited_gap_count': 2 if n == 63 else 0,
                  'course_x': 32.0 if n == 63 else 3.0,
                  'reward_events': {'supported_short': 1, 'safe_segment': 2 if n == 63 else 0}}
                 for n in range(64)]
        repeated = {**first[0], 'episode_index': 1, 'success': True, 'fallen': False,
                    'credited_gap_count': 2, 'course_x': 32.0,
                    'reward_events': {'supported_short': 99, 'safe_segment': 2}}
        trace = first[:63] + [repeated, first[63]]
        report = {'episodes': 256, 'environments': {
            str(n): {'episodes': 2 if n == 0 else 1} for n in range(64)}}
        result = self.pilot.summarize(report, trace, 'treatment', 64)
        self.assertEqual(result['reported_completed_episodes'], 65)
        self.assertEqual(result['first_episode_count'], 64)
        self.assertEqual(result['first_episode_completion_rate'], 1 / 64)
        self.assertEqual(result['supported_short_events'], 64)
        self.assertEqual(result['safe_segment_events'], 2)
        self.assertEqual(result['success_rate'], 1 / 64)

        # Every repeated trace must still have a contiguous per-slot index and
        # exactly match the report's actual per-slot completed-episode count.
        invalid = [({**report, 'episodes': 64}, trace),
                   (report, trace[:-1]),
                   (report, trace[:63] + [{**repeated, 'episode_index': 0}, first[63]]),
                   (report, trace[:63] + [{**repeated, 'env_id': 1}, first[63]]),
                   ({'episodes': 256, 'environments': {
                       str(n): {'episodes': 1} for n in range(64)}}, trace)]
        for bad_report, bad_trace in invalid:
            with self.subTest(report_episodes=bad_report['episodes'],
                              trace_length=len(bad_trace), repeated=bad_trace[63]):
                with self.assertRaises(ValueError):
                    self.pilot.summarize(bad_report, bad_trace, 'treatment', 64)

    def test_real_partial_47_slot_report_rejected_and_a01_preserved(self):
        # Actual a01 failure: the 64-episode request hit the 2560-step cap,
        # producing 76 trace records from only 47 slots, including fast repeats.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            paths = self.pilot.artifacts(workspace, root, self.spec)
            self.assertIn('-a02', paths['control_run_id'])
            a01 = workspace / 'checkpoints/stage1/go1-v1322-supportedshort-pilot-a01-control/model_29.pt'
            a01.parent.mkdir(parents=True)
            a01.write_bytes(b'failed pilot preserved')
            self.pilot.validate_outputs(paths)
            self.assertEqual(a01.read_bytes(), b'failed pilot preserved')
        rows = [{'env_id': n, 'episode_index': 0, 'success': False, 'fallen': True,
                 'collision': False, 'out_of_bounds': False, 'fell_into_gap': False,
                 'required_gap_count': 2, 'credited_gap_count': 0, 'course_x': 3.0,
                 'reward_events': {'supported_short': 2, 'safe_segment': 0}}
                for n in range(47)]
        rows += [{**rows[n % 47], 'episode_index': 1 + n // 47} for n in range(29)]
        counts = {str(n): sum(row['env_id'] == n for row in rows) for n in range(47)}
        with self.assertRaisesRegex(ValueError, 'episode mismatch'):
            self.pilot.summarize({'episodes': 64, 'environments': {
                key: {'episodes': count} for key, count in counts.items()}}, rows, 'control', 64)
        with self.assertRaises(ValueError):
            self.pilot.summarize({'episodes': 256, 'environments': {
                key: {'episodes': count} for key, count in counts.items()}}, rows, 'control', 64)

    def test_pilot_episode_quota_covers_configured_maximum_episode(self):
        source = (ROOT / 'source/go1_hier_sim/go1_hier_sim/cli.py').read_text(encoding='utf-8')
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_evaluation_step_limit')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<cli-limit>', 'exec'), namespace)
        command = self.pilot.eval_command(Path('/sim'), self.config, self.spec, 'control',
                                          Path('/sim/model.pt'), Path('/sim/trace.jsonl'))
        requested = int(command[command.index('--episodes') + 1])
        max_steps = int(float(command[command.index('--episode-seconds') + 1]) / .02)
        self.assertEqual(requested, 256)
        self.assertGreaterEqual(namespace['_evaluation_step_limit'](requested, 0), max_steps)
        self.assertLess(namespace['_evaluation_step_limit'](64, 0), max_steps)

    def test_incomplete_training_checkpoint_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / 'model_0.pt').write_bytes(b'first periodic save only')
            with self.assertRaisesRegex(ValueError, 'final model_29'):
                self.pilot.latest_checkpoint(checkpoint_dir)
            final = checkpoint_dir / 'model_29.pt'
            final.write_bytes(b'completed pilot checkpoint')
            self.assertEqual(self.pilot.latest_checkpoint(checkpoint_dir), final)

    def test_failed_first_arm_restores_overlay_and_never_starts_second(self):
        production = importlib.import_module('production_v1322')
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            (root / 'reports').mkdir(parents=True)
            (root / 'configs').mkdir()
            reward_name = self.manifest['reward_config']['path']
            reward = root / reward_name
            reward.parent.mkdir(parents=True, exist_ok=True)
            reward.write_bytes((ROOT / reward_name).read_bytes())
            source = workspace / 'source/go1_hier_sim/go1_hier_sim/cli.py'
            source.parent.mkdir(parents=True)
            source.write_text('original')
            candidate = workspace / 'candidate.py'
            candidate.write_text('candidate')
            (workspace / 'reports').mkdir()
            row = [(candidate, source, production.digest(source), production.digest(candidate))]
            paths = self.pilot.artifacts(workspace, root, self.spec)
            inputs = {'plan': self.pilot.build_plan(ROOT, self.config, self.spec),
                      'control_toml': self.pilot.control_toml(reward.read_bytes()),
                      'checkpoint': workspace / 'model_299.pt'}
            with patch.object(production, 'ensure_idle'), patch.object(
                    production, 'run_logged', side_effect=RuntimeError('first-arm failure')) as logged:
                with self.assertRaisesRegex(RuntimeError, 'first-arm failure'):
                    with production.Overlay(row, workspace):
                        production.run_pilot(workspace, root, self.manifest, self.config,
                                             self.spec, {}, paths, inputs)
            self.assertEqual(logged.call_count, 1)
            self.assertEqual(source.read_text(), 'original')
            self.assertFalse(paths['receipt'].exists())
            self.assertFalse(paths['treatment_train_log'].exists())
            self.assertEqual(production.digest(paths['plan']), self.spec['plan_sha256'])

    def test_pilot_dispatch_passes_validated_spec_to_eight_argument_runner(self):
        production = importlib.import_module('production_v1322')
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1322'
            (workspace / 'reports').mkdir()
            lock_path = workspace / 'reports/v7_sim.lock'
            lock_path.touch()
            (root / 'configs').mkdir(parents=True)
            (root / 'reports').mkdir()
            (root / 'provenance').mkdir()
            for relative in ('configs/manifest.json', 'configs/teacher_entrance.json',
                             'configs/pilot_execution.json',
                             'provenance/v1321_teacher_entrance.json'):
                (root / relative).write_bytes((ROOT / relative).read_bytes())
            reward = root / self.manifest['reward_config']['path']
            reward.write_bytes((ROOT / self.manifest['reward_config']['path']).read_bytes())
            receipt = root / 'reports/dispatch.receipt'
            paths = {'receipt': receipt}
            inputs = {'pinned': True}
            captured = []

            def pilot_runner(actual_workspace, actual_root, manifest, config, spec,
                             environment, actual_paths, actual_inputs):
                captured.append((actual_workspace, actual_root, manifest, config, spec,
                                 environment, actual_paths, actual_inputs))
                return {'automatic_full_training_authorized': False}

            fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *_: None)
            with patch.dict(sys.modules, {'fcntl': fake_fcntl}), patch.dict(
                    'os.environ', {'V1322_RUN_AUTHORIZED': 'paired-pilot-only'}), patch.object(
                    production.os, 'fstat', return_value=SimpleNamespace(st_ino=lock_path.stat().st_ino)), patch.object(
                    production, 'validate_components'), patch.object(
                    production, 'ensure_idle'), patch.object(
                    production, 'targets', return_value=[]), patch.object(
                    production, 'check_targets'), patch.object(
                    production.pilot_v1322, 'validate_inputs', return_value=(paths, inputs)), patch.object(
                    production, 'Overlay') as overlay_type, patch.object(
                    production, 'run_pilot', side_effect=pilot_runner):
                overlay_type.return_value.__enter__.return_value = SimpleNamespace(restored=True)
                production.run('pilot', root, workspace)

            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0][:5],
                             (workspace, root, self.manifest, self.config, self.spec))
            self.assertIs(captured[0][6], paths)
            self.assertIs(captured[0][7], inputs)
            self.assertEqual(captured[0][5]['GO1_REWARD_CONFIG_PATH'], str(reward))
            self.assertEqual(json.loads(receipt.read_text())['status'], 'pilot-diagnostic-only')

    def test_wrong_pilot_signature_is_reported_as_blocked_not_uncaught(self):
        production = importlib.import_module('production_v1322')
        with patch.object(production, 'run', side_effect=TypeError('pilot interface mismatch')):
            with patch.object(sys, 'argv', ['production_v1322.py', 'pilot']):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(production.main(), 1)
        self.assertIn('V1322_BLOCKED: pilot interface mismatch', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
