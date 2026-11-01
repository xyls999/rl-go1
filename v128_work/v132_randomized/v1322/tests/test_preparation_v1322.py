"""Task 3 preparation contract; no Isaac, remote writes or real overlay."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_path = ROOT / 'scripts/preflight_v1322.py'
        assert module_path.is_file(), 'RED: preflight_v1322 production validator absent'
        spec = importlib.util.spec_from_file_location('v1322_preflight', module_path)
        cls.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.guard)

    def test_pinned_sources_and_immutable_provenance(self):
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        for row in manifest['sources']:
            candidate = ROOT / row['candidate']
            self.assertEqual(self.guard.digest(candidate), row['candidate_sha256'])
            self.assertEqual(row['live'].startswith('source/go1_hier_sim/go1_hier_sim/'), True)
        provenance = json.loads((ROOT / 'provenance/v1321_manifest.json').read_text())
        self.assertEqual(self.guard.digest(ROOT / 'provenance/v1321_teacher_entrance.json'),
                         provenance['teachers']['entrance']['config_sha256'])
        self.assertEqual(manifest['baseline']['geometry.py'],
                         'f60a374596427cc61cad38b6c2e5a1da77f4ff7889906e984c944ad6f736e6b1')

    def test_configuration_preserves_parity_and_avoids_old_outputs(self):
        cfg = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        old = json.loads((ROOT / 'provenance/v1321_teacher_entrance.json').read_text())
        self.guard.validate_config(cfg, old)
        self.assertEqual(cfg['gate_holdout_seed'], None)
        self.assertNotEqual(cfg['state'], old['state'])
        self.assertNotEqual(cfg['sampler_state'], old['sampler_state'])
        self.assertNotEqual(cfg['results_file'], old['results_file'])
        for key in ('speed_min', 'speed_max', 'gate_speed_min', 'gate_speed_max',
                    'gate_family', 'training_roster', 'num_envs', 'episode_seconds'):
            self.assertEqual(cfg[key], old[key])
        self.assertEqual(cfg['phases'][0]['disturbance'], old['phases'][0]['disturbance'])
        plan = json.loads((ROOT / 'configs/pilot_plan.json').read_text())
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.guard.validate_pilot(plan, cfg, manifest)
        self.assertEqual(plan['environment_count'], 64)
        self.assertEqual(plan['arms'][0]['initial_checkpoint'], plan['arms'][1]['initial_checkpoint'])
        self.assertEqual(plan['arms'][0]['seed'], plan['arms'][1]['seed'])
        self.assertEqual(set(plan['arms'][0]) ^ set(plan['arms'][1]), set())

    def test_gate_must_have_one_hard_clean_episode_and_trace_per_slot(self):
        rows = []
        envs = {}
        for i in range(512):
            successful = i != 1
            envs[str(i)] = {'episodes': 1, 'success': int(successful), 'collision': 0,
                            'fallen': 0, 'out_of_bounds': 0, 'fell_into_gap': 0}
            rows.append({'env_id': i, 'episode_index': 0, 'required_gap_count': 2,
                         'credited_gap_count': 2 if successful else 1,
                         'success': successful, 'collision': False, 'fallen': False,
                         'out_of_bounds': False, 'fell_into_gap': False,
                         'course_x': float(i)})
        report = {'episodes': 512, 'environments': envs, 'completion_rate': 511/512,
                  'checkpoint_sha256': 'a'*64}
        self.guard.validate_gate(report, rows, 512, 'a'*64)
        with self.assertRaises(ValueError):
            self.guard.validate_gate(report, rows[:-1] + [rows[-2]], 512, 'a'*64)
        with self.assertRaises(ValueError):
            self.guard.validate_gate(report, rows[:-1] + [{**rows[-1], 'required_gap_count': None}], 512, 'a'*64)
        with self.assertRaises(ValueError):
            self.guard.validate_gate(report, rows, 512, 'b'*64)
        for missing in ('collision', 'fallen', 'out_of_bounds', 'fell_into_gap'):
            incomplete = copy.deepcopy(report)
            del incomplete['environments']['0'][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'terminal'):
                self.guard.validate_gate(incomplete, rows, 512, 'a'*64)
        with self.assertRaises(ValueError):
            self.guard.validate_gate({**report, 'completion_rate': 1}, rows, 512, 'a'*64)

    def test_cli_reports_per_slot_fallen_terminal_flag(self):
        source = (ROOT / 'source/go1_hier_sim/go1_hier_sim/cli.py').read_text()
        self.assertIn('slot["fallen"] += int(bool(item.get("fallen", False)))', source)

    def test_pilot_binds_top_level_geometry_checkpoint_and_seed(self):
        cfg = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        plan = json.loads((ROOT / 'configs/pilot_plan.json').read_text())
        self.guard.validate_pilot(plan, cfg, manifest)
        for value in ('0'*64, 'f'*64):
            tampered = copy.deepcopy(plan)
            tampered['geometry_sha256'] = value
            with self.assertRaisesRegex(ValueError, 'geometry'):
                self.guard.validate_pilot(tampered, cfg, manifest)
        for field, value in [('initial_checkpoint', 'wrong/model.pt'),
                             ('checkpoint_sha256', 'f'*64), ('seed', 2028)]:
            tampered = copy.deepcopy(plan)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'checkpoint|seed'):
                self.guard.validate_pilot(tampered, cfg, manifest)
            tampered = copy.deepcopy(plan)
            tampered['arms'][0][field if field != 'checkpoint_sha256' else 'initial_checkpoint'] = value
            with self.subTest(arm_field=field), self.assertRaises(ValueError):
                self.guard.validate_pilot(tampered, cfg, manifest)

    def test_preflight_lock_open_does_not_truncate_contended_lock(self):
        script = (ROOT / 'scripts/verify_and_train.sh').read_text()
        self.assertIn('exec 9>>"$workspace/reports/v7_sim.lock"', script)
        self.assertNotIn('exec 9>"$workspace/reports/v7_sim.lock"', script)
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / 'v7_sim.lock'
            lock.write_bytes(b'other writer owns this lock\n')
            # Same nontruncating shell open, followed by a refused lock acquisition.
            blocked = subprocess.run(['bash', '-c', 'exec 9>>"$1"; exit 86', 'bash', str(lock)],
                                     capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 86)
            self.assertEqual(lock.read_bytes(), b'other writer owns this lock\n')

    def test_mock_failure_restores_existing_and_removes_new_file(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        self.assertTrue(script.is_file())
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            sources = sandbox / 'source/go1_hier_sim/go1_hier_sim'
            (sources / 'envs').mkdir(parents=True)
            (sandbox / 'reports').mkdir()
            (sources / 'cli.py').write_bytes(b'old cli')
            (sources / 'envs/runtime.py').write_bytes(b'old runtime')
            (sources / 'strict_gate_v1322.py').write_bytes(b'existing helper')
            initial = {str(p.relative_to(sandbox)): p.read_bytes() for p in sources.rglob('*.py')}
            command = ['bash', str(script), 'mock-restore']
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1', 'V1322_MOCK_WORKSPACE': str(sandbox)}
            proc = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('RESTORED', proc.stdout)
            after = {str(p.relative_to(sandbox)): p.read_bytes() for p in sources.rglob('*.py')}
            self.assertEqual(initial, after)
            env['V1322_MOCK_WORKSPACE'] = '/config/Desktop/sim'
            denied = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertNotIn('RESTORED', denied.stdout)

    def test_mock_exclusive_lock_refuses_second_writer(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            (sandbox / 'source/go1_hier_sim/go1_hier_sim/envs').mkdir(parents=True)
            (sandbox / 'reports/.v1322_mock.lock').mkdir(parents=True)
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1', 'V1322_MOCK_WORKSPACE': str(sandbox)}
            blocked = subprocess.run(['bash', str(script), 'mock-restore'], env=env,
                                     capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 86)
            self.assertIn('lock busy', blocked.stderr)
            self.assertTrue((sandbox / 'reports/.v1322_mock.lock').is_dir())

    def test_mock_symlink_refused_without_touching_original(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            sources = sandbox / 'source/go1_hier_sim/go1_hier_sim'
            (sources / 'envs').mkdir(parents=True)
            (sandbox / 'reports').mkdir()
            target = sandbox / 'outside.py'
            target.write_bytes(b'untouched')
            try:
                (sources / 'cli.py').symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest('symlinks unavailable on this host')
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1', 'V1322_MOCK_WORKSPACE': str(sandbox)}
            denied = subprocess.run(['bash', str(script), 'mock-restore'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn('symlinked target', denied.stderr)
            self.assertTrue((sources / 'cli.py').is_symlink())
            self.assertEqual(target.read_bytes(), b'untouched')

    def test_mock_signal_restores_and_gpu_usage_rejects_busy(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            sources = sandbox / 'source/go1_hier_sim/go1_hier_sim'
            (sources / 'envs').mkdir(parents=True)
            (sandbox / 'reports').mkdir()
            (sources / 'cli.py').write_bytes(b'original')
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1', 'V1322_MOCK_WORKSPACE': str(sandbox)}
            signalled = subprocess.run(['bash', str(script), 'mock-signal'], env=env,
                                       capture_output=True, text=True)
            self.assertNotEqual(signalled.returncode, 0)
            self.assertIn('MOCK_RESTORED', signalled.stdout)
            self.assertEqual((sources / 'cli.py').read_bytes(), b'original')
            self.assertFalse((sources / 'envs/runtime.py').exists())
        with patch.object(self.guard.subprocess, 'run') as run:
            run.return_value.stdout = '0, 12\n1, 257\n'
            self.assertEqual(self.guard.gpu_usage_mib(), [12, 257])

    def test_mock_validator_detects_hash_drift_and_existing_outputs(self):
        import shutil
        root = ROOT
        manifest = json.loads((root / 'configs/manifest.json').read_text())
        cfg = json.loads((root / 'configs/teacher_entrance.json').read_text())
        old = json.loads((root / 'provenance/v1321_teacher_entrance.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            for row in manifest['sources']:
                if row['name'] not in ('strict_gate_v1322.py', 'entrance_support_v1322.py',
                                       'gap_score_v1321.py'):
                    live = workspace / row['live']
                    live.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / row['candidate'], live)
                    manifest['baseline'][row['name']] = self.guard.digest(live)
            self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)
            (workspace / cfg['state']).parent.mkdir(parents=True)
            (workspace / cfg['state']).write_text('occupied')
            with self.assertRaisesRegex(ValueError, 'output path already exists'):
                self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)
            (workspace / cfg['state']).unlink()
            manifest['sources'][0]['candidate_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'candidate source hash drift'):
                self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)

    def test_production_preflight_cannot_succeed_without_attestations(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        result = subprocess.run(['bash', str(script), 'preflight'],
                                env={'PATH': '/usr/bin:/bin', 'PREFLIGHT_ONLY': '1'},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('PREFLIGHT_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
