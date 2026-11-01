"""Strict gate counts the first completed episode of each environment only."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/go1_hier_sim/go1_hier_sim"
sys.path.insert(0, str(PACKAGE))
import strict_gate_v1322


class FirstEpisodeTests(unittest.TestCase):
    def make_collector(self):
        self.assertTrue(hasattr(strict_gate_v1322, 'FirstEpisodeCollector'),
                        'FirstEpisodeCollector production symbol is missing')
        return strict_gate_v1322.FirstEpisodeCollector(num_envs=3, trace_required=True)

    def test_first_episode_per_slot_ignores_fast_repeats(self):
        collector = self.make_collector()
        collector.accept([{'env_id': 0, 'success': False}, {'env_id': 1, 'success': True}],
                         [{'env_id': 0, 'required_gap_count': 2, 'credited_gap_count': 0},
                          {'env_id': 1, 'required_gap_count': 2, 'credited_gap_count': 2}])
        self.assertFalse(collector.complete)
        collector.accept([{'env_id': 0, 'success': True}, {'env_id': 2, 'success': True}],
                         [{'env_id': 0, 'required_gap_count': 2, 'credited_gap_count': 2},
                          {'env_id': 2, 'required_gap_count': 2, 'credited_gap_count': 2}])
        self.assertTrue(collector.complete)
        records, trace = collector.finish()
        self.assertEqual([row['env_id'] for row in records], [0, 1, 2])
        self.assertEqual(len(trace), 3)
        self.assertEqual(sum(row['success'] for row in records), 2)
        self.assertEqual(trace[0]['credited_gap_count'], 0)
        self.assertEqual([row['episode_index'] for row in trace], [0, 0, 0])

    def test_missing_slot_rejected(self):
        collector = self.make_collector()
        collector.accept([{'env_id': 0}], [{'env_id': 0, 'required_gap_count': 2, 'credited_gap_count': 0}])
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            collector.finish()

    def test_duplicate_trace_id_rejected(self):
        collector = self.make_collector()
        with self.assertRaisesRegex(RuntimeError, 'trace'):
            collector.accept([{'env_id': 0}, {'env_id': 1}],
                             [{'env_id': 0}, {'env_id': 0}])

    def test_absent_trace_rejected(self):
        collector = self.make_collector()
        with self.assertRaisesRegex(RuntimeError, 'trace'):
            collector.accept([{'env_id': 0}], None)

    def test_trace_missing_required_gap_fields_rejected(self):
        collector = self.make_collector()
        with self.assertRaisesRegex(RuntimeError, 'trace'):
            collector.accept([{'env_id': 0}], [{'env_id': 0}])

    def test_duplicate_episode_id_in_same_batch_rejected(self):
        collector = self.make_collector()
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            collector.accept([{'env_id': 0}, {'env_id': 0}],
                             [{'env_id': 0}, {'env_id': 0}])

    def test_cli_opt_in_wired_to_collector_and_strict_512(self):
        cli = ast.parse((PACKAGE / 'cli.py').read_text(encoding='utf-8'))
        evaluate = next(node for node in cli.body if isinstance(node, ast.FunctionDef) and node.name == '_evaluate')
        parser = next(node for node in cli.body if isinstance(node, ast.FunctionDef) and node.name == '_parser')
        self.assertIn('--strict-first-episode', ast.get_source_segment((PACKAGE / 'cli.py').read_text(), parser))
        source = ast.get_source_segment((PACKAGE / 'cli.py').read_text(), evaluate)
        self.assertIn('FirstEpisodeCollector', source)
        self.assertIn('strict.finish()', source)
        self.assertIn('args.episodes != 512', source)
        self.assertIn('args.num_envs != 512', source)

    @staticmethod
    def driver_helpers(*names):
        source = (ROOT / 'scripts/run_teacher_v1322.py').read_text(encoding='utf-8')
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        scope = {'Path': Path, 'json': json, 'hashlib': hashlib, 'math': math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                     '<isolated-driver-helpers>', 'exec'), scope)
        return scope

    def test_two_gate_commands_use_distinct_trace_paths_and_reject_existing_trace(self):
        helpers = self.driver_helpers('gate_trace_path', 'make_gate_command',
                                      'run_gate_eval', 'holdout_seed')
        config = {'speed_min': 0.35, 'speed_max': 0.8, 'gate_holdout_seed': 31415}
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'reports').mkdir()
            helpers.update({'WORKSPACE': workspace, 'SENSOR_ENV': {}, 'os': os,
                            'disturbance_env': lambda _: {}})
            seen = []

            def fake_run(command, **kwargs):
                seen.append((command, kwargs['env']['GO1_RUN_ID']))
                return types.SimpleNamespace(returncode=0,
                    stdout=f"REPORT: {workspace / 'reports' / ('evaluate_stage1_' + kwargs['env']['GO1_RUN_ID'] + '.json')}\n")

            helpers['subprocess'] = types.SimpleNamespace(run=fake_run, PIPE=-1)
            first = helpers['gate_trace_path'](workspace, 'entrance-s01')
            first_report = helpers['run_gate_eval']('entrance-s01', workspace / 'plan.json',
                {}, config, 512, 35, workspace / 'model.pt', first)
            self.assertEqual(first_report.name, 'evaluate_stage1_gate-entrance-s01.json')
            first.write_text('first gate trace must survive second gate')
            with self.assertRaisesRegex(SystemExit, 'exists'):
                helpers['gate_trace_path'](workspace, 'entrance-s01')
            second = helpers['gate_trace_path'](workspace, 'entrance-s02')
            self.assertNotEqual(first, second)
            second_report = helpers['run_gate_eval']('entrance-s02', workspace / 'plan.json',
                {}, config, 512, 35, workspace / 'model.pt', second)
            self.assertEqual(second_report.name, 'evaluate_stage1_gate-entrance-s02.json')
            self.assertEqual([(cmd[cmd.index('--episode-trace') + 1], run)
                              for cmd, run in seen],
                             [(str(first), 'gate-entrance-s01'),
                              (str(second), 'gate-entrance-s02')])
            self.assertEqual(first.read_text(), 'first gate trace must survive second gate')
            self.assertEqual(seen[1][0][seen[1][0].index('--seed') + 1], '31415')
        source = (ROOT / 'scripts/run_teacher_v1322.py').read_text(encoding='utf-8')
        self.assertIn('require_strict_gate_report(gate_report_path, trace_path', source)
        self.assertIn('run_gate_eval(run_id, gate_plan_path, phase, config, envs,', source)

    def test_holdout_seed_missing_and_reused_rejected(self):
        holdout_seed = self.driver_helpers('holdout_seed')['holdout_seed']
        for config in ({}, {'gate_holdout_seed': 2028}, {'gate_holdout_seed': '2028'},
                       {'gate_holdout_seed': True}):
            with self.subTest(config=config), self.assertRaises((ValueError, SystemExit)):
                holdout_seed(config)
        self.assertEqual(holdout_seed({'gate_holdout_seed': 31415}), 31415)

    def test_driver_strict_command_and_report_validation(self):
        validate = self.driver_helpers('require_strict_gate_report')['require_strict_gate_report']
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / 'model.pt'
            checkpoint.write_bytes(b'checkpoint')
            report_path = directory / 'report.json'
            trace_path = directory / 'trace.jsonl'
            rows = [{'env_id': i, 'episode_index': 0, 'required_gap_count': 2,
                     'credited_gap_count': 2, 'success': i == 2, 'fallen': i == 0,
                     'collision': False, 'out_of_bounds': False, 'fell_into_gap': i == 1}
                    for i in range(3)]
            report = {'episodes': 3, 'stage': 'stage1', 'terrain': 'lp_acrl', 'seed': 31415,
                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'completion_rate': 1 / 3, 'success_rate': 1 / 3, 'fallen_rate': 1 / 3,
                      'collision_rate': 0, 'out_of_bounds_rate': 0, 'fell_into_gap_rate': 1 / 3,
                      'environments': {str(i): {'episodes': 1, 'success': int(row['success']),
                                               'collision': 0, 'out_of_bounds': 0,
                                               'fell_into_gap': int(row['fell_into_gap'])}
                                       for i, row in enumerate(rows)}}

            def save():
                report_path.write_text(json.dumps(report))
                trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

            save()
            self.assertEqual(validate(report_path, trace_path, 3, checkpoint, 31415), report)
            for field, value in (('success', 1), ('collision', 1),
                                 ('out_of_bounds', 1), ('fell_into_gap', 1)):
                with self.subTest(field=field):
                    report['environments']['0'][field] = value
                    save()
                    with self.assertRaises(SystemExit):
                        validate(report_path, trace_path, 3, checkpoint, 31415)
                    report['environments']['0'][field] = int(rows[0][field])
            for field in ('fallen', 'success', 'collision', 'out_of_bounds', 'fell_into_gap'):
                with self.subTest(missing=field):
                    original = rows[0].pop(field)
                    save()
                    with self.assertRaises(SystemExit):
                        validate(report_path, trace_path, 3, checkpoint, 31415)
                    rows[0][field] = original
            rows[0]['fallen'] = False
            save()
            with self.assertRaises(SystemExit):
                validate(report_path, trace_path, 3, checkpoint, 31415)
            rows[0]['fallen'] = True
            for field, value in (('fallen_rate', 0), ('completion_rate', 1),
                                 ('seed', 2028), ('checkpoint_sha256', '0' * 64)):
                with self.subTest(mismatch=field):
                    original = report[field]
                    report[field] = value
                    save()
                    with self.assertRaises(SystemExit):
                        validate(report_path, trace_path, 3, checkpoint, 31415)
                    report[field] = original
            report['environments']['0']['episodes'] = 2
            save()
            with self.assertRaises(SystemExit):
                validate(report_path, trace_path, 3, checkpoint, 31415)
            report['environments']['0']['episodes'] = 1
            rows[2]['env_id'] = 0
            save()
            with self.assertRaises(SystemExit):
                validate(report_path, trace_path, 3, checkpoint, 31415)
            rows[2]['env_id'] = 2
            save()
            with self.assertRaisesRegex(SystemExit, 'run provenance'):
                validate(report_path, trace_path, 3, checkpoint, 31415, 'entrance-s01')

    def test_cli_parser_option_is_opt_in(self):
        app = types.ModuleType('isaaclab.app')
        app.AppLauncher = type('AppLauncher', (), {'add_app_launcher_args': staticmethod(lambda parser: None)})
        spec = importlib.util.spec_from_file_location('isolated_cli_v1322', PACKAGE / 'cli.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'isaaclab': types.ModuleType('isaaclab'), 'isaaclab.app': app}):
            spec.loader.exec_module(module)
            parser = module._parser()
        self.assertFalse(parser.parse_args(['evaluate']).strict_first_episode)
        self.assertTrue(parser.parse_args(['evaluate', '--strict-first-episode']).strict_first_episode)


if __name__ == '__main__':
    unittest.main()
