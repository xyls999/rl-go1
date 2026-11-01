"""Strict gate counts the first completed episode of each environment only."""
import ast
import importlib.util
import json
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

    def test_driver_strict_command_and_report_validation(self):
        source = (ROOT / 'scripts/run_teacher_v1322.py').read_text(encoding='utf-8')
        self.assertIn('--strict-first-episode', source)
        self.assertIn('--episode-trace', source)
        self.assertIn('v1322_go1-v1322-real-entrance-s01-p1-d1_trace.jsonl', source)
        self.assertIn('require_strict_gate_report(', source)
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'require_strict_gate_report')
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        scope = {'Path': Path, 'json': json}
        exec(compile(module, '<isolated-driver-helper>', 'exec'), scope)
        validate = scope['require_strict_gate_report']
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / 'report.json'
            trace_path = Path(directory) / 'trace.jsonl'
            report = {'episodes': 3, 'environments': {str(i): {'episodes': 1} for i in range(3)}}
            rows = [{'env_id': i, 'episode_index': 0, 'required_gap_count': 2,
                     'credited_gap_count': 2} for i in range(3)]
            report_path.write_text(json.dumps(report))
            trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(validate(report_path, trace_path, 3), report)
            report['environments']['0']['episodes'] = 2
            report_path.write_text(json.dumps(report))
            with self.assertRaises(SystemExit):
                validate(report_path, trace_path, 3)
            report['environments']['0']['episodes'] = 1
            report_path.write_text(json.dumps(report))
            rows[2]['env_id'] = 0
            trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(SystemExit):
                validate(report_path, trace_path, 3)

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
