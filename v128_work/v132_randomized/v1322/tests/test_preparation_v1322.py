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

    def test_remote_live_audit_pins_baseline_and_receipt(self):
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        receipt = ROOT / 'provenance/remote_live_audit_v1322.txt'
        audit = receipt.read_text(encoding='utf-8')
        self.assertIn('UTC_2026-', audit)
        self.assertEqual(self.guard.digest(receipt),
                         manifest['execution']['remote_attestation_sha256'])
        self.assertIn(manifest['baseline']['cli.py'] + '  source/go1_hier_sim/go1_hier_sim/cli.py', audit)
        for row in manifest['test_overlays']:
            self.assertIn(row['baseline_sha256'] + '  ' + row['live'], audit)
        for name in ('strict_gate_v1322.py', 'entrance_support_v1322.py',
                     'gap_score_v1321.py', 'landing_stability_v131.py'):
            self.assertIn('ABSENT ' + next(row['live'] for row in manifest['sources']
                                           if row['name'] == name), audit)
        self.assertIn('LOCK_PRESENT mode=664 owner=abc bytes=0', audit)
        self.assertIn('V1322_REMOTE_ABSENT', audit)

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

    def test_production_test_smoke_contract_requires_all_runtime_imports(self):
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        names = {row['name'] for row in manifest['sources']}
        self.assertIn('landing_stability_v131.py', names)
        self.assertTrue(hasattr(self.guard, 'validate_live_baseline'))
        self.assertTrue(hasattr(self.guard, 'validate_mode_outputs'))
        self.assertTrue((ROOT / 'scripts/production_v1322.py').is_file())
        source = (ROOT / 'scripts/production_v1322.py').read_text()
        self.assertIn('sim.sh', source)
        self.assertIn('test', source)
        self.assertIn('combined_part_entrance', source)
        self.assertIn('combined_junction', source)
        self.assertIn('gaps_long', source)
        self.assertIn('tire_field', source)
        self.assertNotIn('run_teacher_v1322.py', source)

    def test_production_overlay_restores_existing_and_new_on_failure(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            import production_v1322 as production
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / 'reports').mkdir()
            existing = workspace / 'source/old.py'
            existing.parent.mkdir()
            existing.write_bytes(b'original')
            absent = workspace / 'source/new.py'
            source_a = workspace / 'candidate_a.py'
            source_b = workspace / 'candidate_b.py'
            source_a.write_bytes(b'new old')
            source_b.write_bytes(b'new helper')
            rows = [(source_a, existing, self.guard.digest(existing), self.guard.digest(source_a)),
                    (source_b, absent, None, self.guard.digest(source_b))]
            for error in (ValueError('test failure'), KeyboardInterrupt('signal')):
                with self.assertRaises(type(error)):
                    with production.Overlay(rows, workspace) as overlay:
                        self.assertEqual(existing.read_bytes(), b'new old')
                        self.assertEqual(absent.read_bytes(), b'new helper')
                        raise error
                self.assertTrue(overlay.restored)
                self.assertEqual(existing.read_bytes(), b'original')
                self.assertFalse(absent.exists())
                self.assertFalse(list((workspace / 'reports').glob('v1322-overlay-*')))
            copy = production.atomic_copy
            count = 0
            def interrupted_copy(source, destination, expected):
                nonlocal count
                count += 1
                if count == 2:
                    raise KeyboardInterrupt('interrupted before second atomic swap')
                return copy(source, destination, expected)
            with patch.object(production, 'atomic_copy', side_effect=interrupted_copy):
                with self.assertRaises(KeyboardInterrupt):
                    with production.Overlay(rows, workspace):
                        pass
            self.assertEqual(existing.read_bytes(), b'original')
            self.assertFalse(absent.exists())
            source_a.write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError, 'candidate'):
                production.check_targets(rows, workspace)
            with self.assertRaisesRegex(ValueError, 'training'):
                production.run('train', ROOT, workspace)

    def test_production_test_and_smoke_commands_no_training(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            import production_v1322 as production
        finally:
            sys.path.pop(0)
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            test_rows = production.targets(ROOT, workspace, manifest, 'test')
            smoke_rows = production.targets(ROOT, workspace, manifest, 'smoke')
            self.assertEqual(len(test_rows), len(manifest['sources']) + 4)
            self.assertEqual(len(smoke_rows), len(manifest['sources']))
            self.assertFalse(any('rewards_stage1_gaps' in str(live) for _, live, _, _ in test_rows))
            for name, _ in production.FAMILIES:
                command = production.smoke_command(workspace, name, 'unique', 9700)
                self.assertEqual(command[1:3], ['check', '--stage'])
                self.assertEqual(command[command.index('--num-envs') + 1], '512')
                self.assertNotIn('train', command)
            environment = production.common_environment(workspace, manifest)
            self.assertEqual(environment['GO1_REWARD_CONFIG_PATH'],
                             str(workspace / manifest['execution']['train_and_gate_env']['GO1_REWARD_CONFIG_PATH']))
            self.assertNotIn('GO1_AUTO_P2', environment)

    def test_production_preflight_rejects_missing_cli_pin_and_stale_mode_outputs(self):
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            live = workspace / manifest['sources'][0]['live']
            live.parent.mkdir(parents=True)
            live.write_text('active CLI')
            with self.assertRaisesRegex(ValueError, 'unexpected unpinned'):
                self.guard.validate_live_baseline(workspace, {
                    'sources': [manifest['sources'][0]], 'baseline': {'cli.py': None}}, production=True)
            (workspace / 'v128_work/v132_randomized/v1322/reports').mkdir(parents=True)
            self.guard.validate_mode_outputs(workspace, 'test', 'unique', attempt=2)
            old = workspace / 'v128_work/v132_randomized/v1322/reports/v1322_tests.log'
            old.write_text('failed first attempt preserved')
            self.guard.validate_mode_outputs(workspace, 'test', 'unique', attempt=2)
            new = workspace / 'v128_work/v132_randomized/v1322/reports/v1322_test_a02.receipt'
            new.write_text('second attempt already used')
            with self.assertRaisesRegex(ValueError, 'mode output already exists'):
                self.guard.validate_mode_outputs(workspace, 'test', 'unique', attempt=2)
            self.guard.validate_mode_outputs(workspace, 'test', 'unique', attempt=3)
            self.assertEqual(old.read_text(), 'failed first attempt preserved')
            self.assertEqual(new.read_text(), 'second attempt already used')
            with self.assertRaisesRegex(ValueError, 'not implemented'):
                self.guard.validate_mode_outputs(workspace, 'train', 'unique', attempt=3)

    def test_legacy_test_overlays_exercise_explicit_default_and_production_environments(self):
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        overlays = {row['live']: row for row in manifest['test_overlays']}
        for target, baseline in (
            ('tests/test_v8_geometry.py', 'test_v8_geometry.baseline.py'),
            ('tests/test_v8_training_warmstart.py', 'test_v8_training_warmstart.baseline.py'),
        ):
            self.assertIn(target, overlays)
            row = overlays[target]
            self.assertEqual(row['baseline_sha256'],
                             self.guard.digest(ROOT / 'provenance' / baseline))
            source = ROOT.parents[2] / row['candidate']
            self.assertTrue(source.is_file(), row['candidate'])
            self.assertEqual(self.guard.digest(source), row['candidate_sha256'])
            text = source.read_text(encoding='utf-8')
            if 'geometry' in target:
                self.assertIn('GO1_V8_LONG_GAP_THRESHOLD', text)
                self.assertIn('0.26', text)
                self.assertIn('0.05', text)
                self.assertIn('== 3', text)
                self.assertIn('== 6', text)
            else:
                self.assertIn('GO1_PPO_CONFIG_PATH', text)
                self.assertIn('ppo.json', text)
                self.assertIn('ppo_gaps_long.json', text)
                self.assertIn('[1e-7, 3e-5, 3e-6]', text)
                self.assertIn('[2e-6, 3e-5, 2e-5]', text)
                self.assertIn('len(assigned) == len(set(assigned))', text)

    def test_attempt_artifacts_preserve_failed_attempt_and_bind_smoke(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            import production_v1322 as production
        finally:
            sys.path.pop(0)
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.assertEqual(manifest['execution']['test_attempt'], 3)
        names = production.attempt_artifacts(3)
        self.assertEqual(names['test_log'], 'v1322_tests_a03.log')
        self.assertEqual(names['focus_log'], 'v1322_focus_a03.log')
        self.assertEqual(names['test_receipt'], 'v1322_test_a03.receipt')
        self.assertEqual(names['smoke_receipt'], 'v1322_physx_a03.receipt')
        self.assertNotIn(names['test_log'], ('v1322_tests.log', 'v1322_tests_a02.log'))
        with tempfile.TemporaryDirectory() as temp:
            reports = Path(temp) / 'reports'
            reports.mkdir()
            (reports / 'v1322_tests.log').write_text('first failed attempt')
            (reports / 'v1322_tests_a02.log').write_text('second failed attempt')
            self.assertEqual((reports / 'v1322_tests.log').read_text(), 'first failed attempt')
            self.assertEqual((reports / 'v1322_tests_a02.log').read_text(), 'second failed attempt')
            self.assertNotIn('v1322_tests_a02.log', names.values())

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

    def test_preflight_requires_preexisting_regular_lock_before_open(self):
        script = (ROOT / 'scripts/verify_and_train.sh').read_text()
        lock_open = script.index('exec 9>>"$workspace/reports/v7_sim.lock"')
        self.assertIn('[[ -f "$workspace/reports/v7_sim.lock" && ! -L "$workspace/reports/v7_sim.lock" ]]',
                      script[:lock_open])
        with tempfile.TemporaryDirectory() as temp:
            reports = Path(temp) / 'reports'
            reports.mkdir()
            lock = reports / 'v7_sim.lock'
            target = Path(temp) / 'target'
            target.write_text('original')
            for kind in ('absent', 'symlink'):
                if kind == 'symlink':
                    lock.symlink_to(target)
                command = ('workspace="$1"; '
                           '[[ -f "$workspace/reports/v7_sim.lock" && '
                           '! -L "$workspace/reports/v7_sim.lock" ]] || exit 86; '
                           'exec 9>>"$workspace/reports/v7_sim.lock"')
                blocked = subprocess.run(['bash', '-c', command, 'bash', temp],
                                         capture_output=True, text=True)
                self.assertEqual(blocked.returncode, 86, kind)
                self.assertEqual(lock.exists(), kind == 'symlink')
                self.assertEqual(target.read_text(), 'original')
                if kind == 'symlink':
                    lock.unlink()
            lock.write_bytes(b'keep')
            opened = subprocess.run(['bash', '-c', command, 'bash', temp],
                                    capture_output=True, text=True)
            self.assertEqual(opened.returncode, 0)
            self.assertEqual(lock.read_bytes(), b'keep')

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
            original_audit = manifest['execution']['remote_attestation_sha256']
            manifest['execution']['remote_attestation_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'read-only remote audit drift'):
                self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)
            manifest['execution']['remote_attestation_sha256'] = original_audit
            (workspace / cfg['state']).parent.mkdir(parents=True)
            (workspace / cfg['state']).write_text('occupied')
            with self.assertRaisesRegex(ValueError, 'output path already exists'):
                self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)
            (workspace / cfg['state']).unlink()
            manifest['sources'][0]['candidate_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'candidate source hash drift'):
                self.guard.validate_components(workspace, root, manifest, cfg, old, production=False)

    def test_mock_parent_symlinks_cannot_escape_sandbox(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            sandbox, outside = parent / 'sandbox', parent / 'outside'
            sandbox.mkdir()
            outside.mkdir()
            (sandbox / 'reports').mkdir()
            source = outside / 'go1_hier_sim/go1_hier_sim/envs'
            source.mkdir(parents=True)
            protected = source.parent / 'cli.py'
            protected.write_bytes(b'protected live source')
            try:
                (sandbox / 'source').symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('directory symlinks unavailable on this host')
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1',
                   'V1322_MOCK_WORKSPACE': str(sandbox)}
            result = subprocess.run(['bash', str(script), 'mock-restore'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('MOCK_DENIED', result.stderr)
            self.assertEqual(protected.read_bytes(), b'protected live source')
            self.assertFalse(list((sandbox / 'reports').glob('v1322-backup.*')))
            self.assertFalse((sandbox / 'reports/.v1322_mock.lock').exists())

    def test_mock_restore_failure_preserves_backup_for_manual_recovery(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            sources = sandbox / 'source/go1_hier_sim/go1_hier_sim'
            (sources / 'envs').mkdir(parents=True)
            (sandbox / 'reports').mkdir()
            (sources / 'cli.py').write_bytes(b'old live cli')
            env = {'PATH': '/usr/bin:/bin', 'V1322_MOCK': '1',
                   'V1322_MOCK_WORKSPACE': str(sandbox), 'V1322_MOCK_FAIL_RESTORE': '1'}
            result = subprocess.run(['bash', str(script), 'mock-restore'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('RESTORE_FAILED', result.stderr)
            self.assertEqual(len(list((sandbox / 'reports').glob('v1322-backup.*'))), 1)

    def test_idle_check_catches_versioned_python_torchrun_and_isaac(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            import production_v1322 as production
        finally:
            sys.path.pop(0)
        busy = ('python3.11 /root/env_isaaclab/bin/python3.11 /config/Desktop/sim/scripts/sim.py check',
                'python3.12 python3.12 /config/Desktop/sim/scripts/sim.py train',
                'torchrun /root/env_isaaclab/bin/torchrun --nproc_per_node=4 scripts/sim.py train',
                'bash bash /config/Desktop/sim/scripts/torchrun_sim.sh train',
                'kit /isaac-sim/kit/kit --enable isaacsim.core.api')
        for line in busy:
            with self.subTest(line=line), patch.object(production, 'gpu_usage_mib', return_value=[0]*4), \
                    patch.object(production.subprocess, 'run') as run:
                run.return_value.stdout = line + '\n'
                with self.assertRaisesRegex(ValueError, 'Isaac process'):
                    production.ensure_idle()
        with patch.object(production, 'gpu_usage_mib', return_value=[0]*4), \
                patch.object(production.subprocess, 'run') as run:
            run.return_value.stdout = 'python3.11 python3.11 unrelated_analysis.py\n'
            production.ensure_idle()

    def test_test_smoke_read_only_preflight_does_not_require_final_seed(self):
        cfg = json.loads((ROOT / 'configs/teacher_entrance.json').read_text())
        old = json.loads((ROOT / 'provenance/v1321_teacher_entrance.json').read_text())
        manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
        self.assertIsNone(cfg['gate_holdout_seed'])
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            for row in manifest['sources']:
                if row['name'] in ('strict_gate_v1322.py', 'entrance_support_v1322.py',
                                   'gap_score_v1321.py', 'landing_stability_v131.py'):
                    continue
                live = workspace / row['live']
                live.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / row['candidate'], live)
                manifest['baseline'][row['name']] = self.guard.digest(live)
            for row in manifest['test_overlays']:
                candidate = workspace / row['candidate']
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(('synthetic pinned ' + row['candidate']).encode())
                row['candidate_sha256'] = self.guard.digest(candidate)
                live = workspace / row['live']
                live.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, live)
                row['baseline_sha256'] = self.guard.digest(live)
            (workspace / 'reports').mkdir(exist_ok=True)
            self.guard.validate_read_only_preflight(workspace, ROOT, manifest, cfg, old,
                                                    mode='test', gpu_mib=[0]*4)
            with self.assertRaisesRegex(ValueError, 'final holdout'):
                self.guard.validate_components(workspace, ROOT, manifest, cfg, old,
                                               production=True, mode='final')
            with patch.object(self.guard, 'validate_test_receipt',
                              side_effect=ValueError('test receipt missing')) as receipt_check:
                with self.assertRaisesRegex(ValueError, 'test receipt'):
                    self.guard.validate_read_only_preflight(workspace, ROOT, manifest, cfg, old,
                                                            mode='smoke', gpu_mib=[0]*4)
                receipt_check.assert_called_once_with(ROOT, manifest)
            with self.assertRaisesRegex(ValueError, 'GPU'):
                self.guard.validate_read_only_preflight(workspace, ROOT, manifest, cfg, old,
                                                        mode='test', gpu_mib=[0, 0, 257, 0])
        shell = (ROOT / 'scripts/verify_and_train.sh').read_text()
        self.assertIn('preflight-test', shell)
        self.assertIn('preflight-smoke', shell)

    def test_production_preflight_cannot_succeed_without_attestations(self):
        script = ROOT / 'scripts/verify_and_train.sh'
        result = subprocess.run(['bash', str(script), 'preflight'],
                                env={'PATH': '/usr/bin:/bin', 'PREFLIGHT_ONLY': '1'},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('PREFLIGHT_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
