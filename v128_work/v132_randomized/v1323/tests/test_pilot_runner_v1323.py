"""V132.3 pilot must isolate only edge-contact weight and fail closed."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.pilot = importlib.import_module('pilot_v1323')
        self.production = importlib.import_module('production_v1323')
        self.preflight = importlib.import_module('preflight_v1323')
        self.manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.spec = json.loads((ROOT / 'configs/pilot_execution.json').read_text())
        self.config = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())

    def test_pinned_plan_and_only_edge_weight_differs(self):
        plan = self.pilot.build_plan(ROOT, self.config, self.spec)
        self.assertEqual(len(plan['instances']), 64)
        self.assertEqual(set(plan['families']), {'combined_part_entrance'})
        treatment = (ROOT / self.manifest['reward_config']['path']).read_bytes()
        control = self.pilot.control_toml(treatment)
        self.pilot.validate_tomls(control, treatment, self.spec, self.manifest)
        with self.assertRaises(ValueError):
            self.pilot.validate_tomls(control.replace(b'weight = 6.0', b'weight = 0.0', 1),
                                      treatment, self.spec, self.manifest)
        old_reward = (ROOT.parent / 'v1322/configs/rewards_stage1_gaps_v131_landing_stability.toml').read_bytes()
        self.pilot.validate_against_v1322(control, old_reward)
        with self.assertRaisesRegex(ValueError, 'old reward'):
            self.pilot.validate_against_v1322(
                control.replace(b'weight = 6.0', b'weight = 1.0', 1), old_reward)

    def test_existing_output_or_candidate_sha_drift_blocks_before_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1323'
            paths = self.pilot.artifacts(workspace, root, self.spec)
            old = workspace / 'checkpoints/stage1/go1-v1322-supportedshort-pilot-a02-control/model_29.pt'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'old preserved')
            self.pilot.validate_outputs(paths)
            self.assertEqual(old.read_bytes(), b'old preserved')
            paths['receipt'].parent.mkdir(parents=True)
            paths['receipt'].write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.pilot.validate_outputs(paths)
            paths['receipt'].unlink()
            candidate = root / self.manifest['sources'][0]['candidate']
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError, 'candidate'):
                self.preflight.validate_candidate_sources(root, self.manifest)

    def test_first_episode_coverage_rejects_unmatched_repeated_rows(self):
        first = [{'env_id': i, 'episode_index': 0, 'success': i == 63,
                  'fallen': i != 63, 'collision': False, 'out_of_bounds': False,
                  'fell_into_gap': False, 'required_gap_count': 2,
                  'credited_gap_count': 2 if i == 63 else 0, 'course_x': 3.0,
                  'reward_events': {'safe_segment': 2 if i == 63 else 0,
                                    'supported_short': 1}} for i in range(64)]
        repeated = {**first[0], 'episode_index': 1, 'success': True, 'fallen': False,
                    'credited_gap_count': 2}
        report = {'episodes': 256, 'environments': {
            str(i): {'episodes': 2 if i == 0 else 1} for i in range(64)}}
        result = self.pilot.summarize(report, first[:63] + [repeated, first[63]], 'control', 64)
        self.assertEqual(result['first_episode_count'], 64)
        self.assertEqual(result['reported_completed_episodes'], 65)
        self.assertEqual(result['first_episode_completion_rate'], 1 / 64)
        with self.assertRaisesRegex(ValueError, 'episode mismatch'):
            self.pilot.summarize({'episodes': 64, 'environments': report['environments']},
                                 first[:63] + [repeated, first[63]], 'control', 64)

    def test_pilot_treatment_requires_observed_negative_edge_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'arm.log'
            log.write_text('Episode_Reward/edge_contact: 0.0000\n' * 30)
            self.assertEqual(self.production._edge_activation(log, 0.0), 0.0)
            with self.assertRaisesRegex(ValueError, 'never activated'):
                self.production._edge_activation(log, -1.0)
            log.write_text('Episode_Reward/edge_contact: -0.0100\n' +
                           'Episode_Reward/edge_contact: 0.0000\n' * 29)
            self.assertLess(self.production._edge_activation(log, -1.0), 0)
            with self.assertRaisesRegex(ValueError, 'control'):
                self.production._edge_activation(log, 0.0)

    def test_commands_match_warmstart_speed_and_256_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / 'checkpoints/stage1/go1-v1322-real-entrance-s01-p1-d1/model_649.pt'
            plan = workspace / 'plan.json'
            commands = [self.pilot.train_command(workspace, self.config, self.spec, arm, plan, source)
                        for arm in ('control', 'treatment')]
            for command in commands:
                for option, value in (('--num-envs', '64'), ('--iterations', '30'),
                                      ('--critic-warmup-iterations', '5'), ('--seed', '42'),
                                      ('--speed-min', '0.35'), ('--speed-max', '0.8'),
                                      ('--initialize-from', str(source))):
                    self.assertEqual(command[command.index(option) + 1], value)
                self.assertNotIn('--strict-first-episode', command)
            eval_cmd = self.pilot.eval_command(workspace, self.config, self.spec, 'control',
                                               workspace / 'model_29.pt', workspace / 'trace.jsonl')
            self.assertEqual(eval_cmd[eval_cmd.index('--episodes') + 1], '256')
            self.assertEqual(eval_cmd[eval_cmd.index('--seed') + 1], '2028')
            self.assertEqual(eval_cmd[eval_cmd.index('--speed-min') + 1], '0.35')
            self.assertEqual(eval_cmd[eval_cmd.index('--speed-max') + 1], '0.8')
            commands[0][commands[0].index('--run-name') + 1] = commands[1][commands[1].index('--run-name') + 1]
            self.assertEqual(commands[0], commands[1])

    def test_real_dispatch_restores_overlay_even_when_external_pilot_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'v128_work/v132_randomized/v1323'
            (workspace / 'reports').mkdir()
            live = workspace / 'source/old.py'
            live.parent.mkdir(parents=True)
            live.write_bytes(b'active original')
            helper = workspace / 'source/new.py'
            staged_a = root / 'old.py'
            staged_b = root / 'new.py'
            staged_a.parent.mkdir(parents=True)
            staged_a.write_bytes(b'candidate')
            staged_b.write_bytes(b'new helper')
            rows = [(staged_a, live, self.preflight.digest(live), self.preflight.digest(staged_a)),
                    (staged_b, helper, None, self.preflight.digest(staged_b))]
            fake = {'pilot_completion_delta': 0.1}
            with patch.object(self.production, 'run_pilot', return_value=fake):
                result = self.production.execute_mode('pilot', workspace, root, {}, {}, {},
                                                      {}, {}, {}, rows)
            self.assertEqual(result, fake)
            self.assertEqual(live.read_bytes(), b'active original')
            self.assertFalse(helper.exists())
            self.assertFalse(list((workspace / 'reports').glob('v1323-overlay-*')))
            with patch.object(self.production, 'run_pilot', side_effect=RuntimeError('pilot stopped')):
                with self.assertRaisesRegex(RuntimeError, 'pilot stopped'):
                    self.production.execute_mode('pilot', workspace, root, {}, {}, {},
                                                 {}, {}, {}, rows)
            self.assertEqual(live.read_bytes(), b'active original')
            self.assertFalse(helper.exists())

    def test_real_dispatch_pilot_and_no_other_training_mode(self):
        self.assertIn('pilot', self.production.ALLOWED_MODES)
        self.assertNotIn('train', self.production.ALLOWED_MODES)
        self.assertNotIn('gate', self.production.ALLOWED_MODES)
        self.assertNotIn('junction', self.production.ALLOWED_MODES)
        self.assertEqual(self.production.mode_authorization('pilot'), 'paired-pilot-only')
        with self.assertRaises(ValueError):
            self.production.mode_authorization('train')
        self.assertEqual(self.spec['checkpoint_sha256'],
                         '1a120f37c23f621e9f08772b1f6fc8520fabc4fb642b1da6342e08a96c7793f9')
        self.assertIs(self.spec['final_holdout'], False)


if __name__ == '__main__':
    unittest.main()
