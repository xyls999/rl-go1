"""A paired pilot must preserve every input but the supported-short TOML weight."""
import importlib
import json
from pathlib import Path
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
                self.assertEqual(evaluation[evaluation.index('--episodes') + 1], '64')
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
        report = {'episodes': 64, 'environments': {str(n): {'episodes': 1} for n in range(64)}}
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
        report = {'episodes': 64, 'environments': {
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
        invalid = [({**report, 'episodes': 65}, trace),
                   (report, trace[:-1]),
                   (report, trace[:63] + [{**repeated, 'episode_index': 0}, first[63]]),
                   (report, trace[:63] + [{**repeated, 'env_id': 1}, first[63]]),
                   ({'episodes': 64, 'environments': {
                       str(n): {'episodes': 1} for n in range(64)}}, trace)]
        for bad_report, bad_trace in invalid:
            with self.subTest(report_episodes=bad_report['episodes'],
                              trace_length=len(bad_trace), repeated=bad_trace[63]):
                with self.assertRaises(ValueError):
                    self.pilot.summarize(bad_report, bad_trace, 'treatment', 64)

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


if __name__ == '__main__':
    unittest.main()
