#!/usr/bin/env python3
"""Single-lock V132.3 test, PhysX and paired development-pilot runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from preflight_v1323 import BASE_SHA, digest, require, preflight
import pilot_v1323

ALLOWED_MODES = ('test', 'smoke', 'pilot')
FAMILIES = (('entrance', 'combined_part_entrance'), ('junction', 'combined_junction'),
            ('long', 'gaps_long'), ('tire', 'tire_field'))
_active_process: subprocess.Popen | None = None


def mode_authorization(mode: str) -> str:
    require(mode in ALLOWED_MODES, 'V132.3 does not authorize full train/final gate/other teachers')
    return 'paired-pilot-only' if mode == 'pilot' else 'test-smoke-only'


def attempt_artifacts(attempt: int) -> dict[str, str]:
    require(attempt == 1, 'unapproved attempt number')
    return {'test_log': 'v1323_tests_a01.log', 'focus_log': 'v1323_focus_a01.log',
            'test_receipt': 'v1323_test_a01.receipt',
            'smoke_receipt': 'v1323_physx_a01.receipt',
            'smoke_prefix': 'v1323_smoke_a01_'}


def targets(root: Path, workspace: Path, manifest: dict, mode: str) -> list[tuple[Path, Path, str | None, str]]:
    rows = [(root / row['candidate'], workspace / row['live'],
             manifest['baseline'].get(row['name']), row['candidate_sha256'])
            for row in manifest['sources']]
    if mode == 'test':
        rows.extend((workspace / row['candidate'], workspace / row['live'],
                     row['baseline_sha256'], row['candidate_sha256'])
                    for row in manifest['test_overlays'])
    return rows


def check_targets(rows: list[tuple[Path, Path, str | None, str]], workspace: Path) -> None:
    seen: set[Path] = set()
    for candidate, live, original, expected in rows:
        require(live not in seen and workspace in live.parents and candidate != live,
                'unsafe/duplicate overlay target: ' + str(live))
        seen.add(live)
        require(not candidate.is_symlink() and candidate.is_file() and digest(candidate) == expected,
                'unpinned overlay candidate: ' + str(candidate))
        for parent in candidate.parents:
            if parent == workspace:
                break
            require(not parent.is_symlink(), 'symlinked candidate parent: ' + str(parent))
        require(not live.is_symlink(), 'symlinked live target: ' + str(live))
        for parent in live.parents:
            if parent == workspace:
                break
            require(not parent.is_symlink(), 'symlinked overlay parent: ' + str(parent))
        if original is None:
            require(not live.exists(), 'unexpected live helper: ' + str(live))
        else:
            require(live.is_file() and digest(live) == original,
                    'live baseline SHA mismatch: ' + str(live))


def atomic_copy(source: Path, destination: Path, expected_sha256: str) -> None:
    handle, name = tempfile.mkstemp(prefix='.v1323-', dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        require(digest(temporary) == expected_sha256, 'temporary copy SHA mismatch')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class Overlay:
    def __init__(self, rows: list[tuple[Path, Path, str | None, str]], workspace: Path):
        self.rows, self.workspace = rows, workspace
        self.backup: Path | None = None
        self.installed: list[tuple[Path, str | None, str]] = []
        self.restored = False

    def __enter__(self):
        check_targets(self.rows, self.workspace)
        self.backup = Path(tempfile.mkdtemp(prefix='v1323-overlay-', dir=self.workspace / 'reports'))
        try:
            for index, (_, live, original, _) in enumerate(self.rows):
                if original is not None:
                    saved = self.backup / str(index)
                    shutil.copy2(live, saved)
                    require(digest(saved) == original, 'backup SHA mismatch: ' + str(live))
            for candidate, live, original, expected in self.rows:
                live.parent.mkdir(parents=True, exist_ok=True)
                self.installed.append((live, original, expected))
                atomic_copy(candidate, live, expected)
                require(digest(live) == expected, 'installed source SHA mismatch: ' + str(live))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        issues = []
        for live, original, expected in reversed(self.installed):
            try:
                if original is None:
                    if live.exists() and digest(live) == expected:
                        live.unlink()
                    elif live.exists():
                        raise RuntimeError('modified new helper: ' + str(live))
                    require(not live.exists(), 'new helper remains: ' + str(live))
                else:
                    index = next(i for i, row in enumerate(self.rows) if row[1] == live)
                    atomic_copy(self.backup / str(index), live, original)
                    require(digest(live) == original, 'restore SHA mismatch: ' + str(live))
            except (OSError, ValueError, RuntimeError) as error:
                issues.append(str(error))
        if self.backup is not None and not issues:
            shutil.rmtree(self.backup)
        self.restored = not issues
        if issues:
            raise RuntimeError('RESTORE FAILED; backups retained: ' + '; '.join(issues))
        return False


def isaac_process_in_listing(listing: str) -> bool:
    launch = re.compile(r'(?:^|[\s/])(?:scripts/sim\.py|scripts/run_teacher_v\d+\.py|'
                        r'scripts/torchrun_sim\.sh|sim\.sh)(?:\s|$)')
    for line in listing.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        comm, arguments = parts
        if (re.fullmatch(r'python(?:\d+(?:\.\d+)?)?|torchrun|bash|sh|zsh', comm)
                and launch.search(arguments)):
            return True
        if comm in {'kit', 'isaac-sim', 'isaacsim'} and re.search(r'isaac(?:sim|-sim|lab)', arguments):
            return True
    return False


def ensure_idle() -> None:
    from preflight_v1323 import gpu_usage_mib
    require(all(0 <= item <= 256 for item in gpu_usage_mib()), 'GPU busy')
    listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True,
                             text=True, check=True).stdout
    require(not isaac_process_in_listing(listing), 'Isaac process active')


def common_environment(workspace: Path, manifest: dict) -> dict[str, str]:
    environment = dict(os.environ)
    for name in ('GO1_V29_TEACHER_DISTILL', 'GO1_WATCHDOG', 'GO1_AUTO_P2',
                 'GO1_V29_TEACHERS', 'GO1_V23_TIRE_BRAKE', 'GO1_ACTOR_FILTER'):
        environment.pop(name, None)
    for name, value in manifest['execution']['train_and_gate_env'].items():
        environment[name] = str(workspace / value) if name.endswith('_PATH') else str(value)
    environment.update({
        'CUDA_VISIBLE_DEVICES': '0', 'GO1_DR_MASS_UP': '0.5', 'GO1_DR_PUSH_VEL': '0',
        'GO1_DR_MOTOR_PCT': '0.02', 'GO1_DR_COM_SCALE': '0.25',
        'GO1_DR_FRICTION_STATIC_MIN': '0.5', 'GO1_DR_FRICTION_STATIC_MAX': '1.3',
        'GO1_DR_FRICTION_DYNAMIC_MIN': '0.4', 'GO1_DR_FRICTION_DYNAMIC_MAX': '1.1',
        'GO1_OBS_DELAY_STEPS': '0', 'GO1_OBS_DELAY_RANDOM': '0',
        'GO1_OBS_NOISE_STD': '0', 'GO1_OBS_BIAS_RAD': '0', 'GO1_OBS_SCAN_DROPOUT': '0',
    })
    environment.pop('DISPLAY', None)
    environment.pop('WAYLAND_DISPLAY', None)
    return environment


def run_logged(command: list[str], log: Path, env: dict[str, str], workspace: Path) -> None:
    global _active_process
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=workspace, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        _active_process = process
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            _active_process = None


def run_tests(workspace: Path, root: Path, manifest: dict, env: dict[str, str]) -> dict:
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    focus = root / 'reports' / names['focus_log']
    run_logged(['/root/env_isaaclab/bin/python', '-m', 'unittest', 'discover', '-s',
                str(root / 'tests'), '-p', 'test_*v1323.py', '-v'], focus, env, workspace)
    focus_text = focus.read_text(encoding='utf-8', errors='replace')
    require(re.search(r'Ran \d+ tests?', focus_text) is not None and
            re.search(r'^OK\s*$', focus_text, re.MULTILINE) is not None,
            'V132.3 focused tests did not complete')
    log = root / 'reports' / names['test_log']
    run_logged([str(workspace / 'sim.sh'), 'test', '-q'], log, env, workspace)
    text = log.read_text(encoding='utf-8', errors='replace')
    require(re.search(r'\b\d+ passed\b', text) is not None and
            re.search(r'\b\d+ failed\b', text) is None,
            'project test suite did not report complete success')
    return {'log_sha256': digest(log), 'focus_sha256': digest(focus)}


def run_smoke(workspace: Path, root: Path, manifest: dict, env: dict[str, str]) -> dict:
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    passed = {}
    for index, (family, kind) in enumerate(FAMILIES):
        ensure_idle()
        plan = root / 'reports' / f'v1323_smoke_plan_a01_{family}.json'
        with plan.open('x', encoding='utf-8') as stream:
            json.dump({'families': [kind] * 512, 'difficulty': [.6] * 512,
                       'speed_level': [2] * 512, 'speed_min': .35, 'speed_max': .8}, stream)
        run_name = f"v1323-{manifest['execution']['run_id']}-a01-{family}"
        family_env = dict(env, GO1_TASK_PLAN=str(plan), GO1_RUN_ID=run_name)
        log = root / 'reports' / f"{names['smoke_prefix']}{family}.log"
        command = [str(workspace / 'sim.sh'), 'check', '--stage', 'stage1', '--terrain', 'lp_acrl',
                   '--headless', '--device', 'cuda:0', '--num-envs', '512',
                   '--seed', str(9700 + index), '--speed-min', '0.35', '--speed-max', '0.8']
        run_logged(command, log, family_env, workspace)
        report_path = workspace / 'reports' / f'check_{run_name}.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(report.get('passed') is True and report.get('num_envs') == 512 and
                report.get('policy_shape') == [512, 1073] and
                report.get('critic_shape') == [512, 1080], 'PhysX check failed: ' + family)
        passed[family] = {'log_sha256': digest(log), 'report_sha256': digest(report_path)}
    return {'families': passed}


def _edge_activation(log: Path, weight: float) -> float:
    text = log.read_text(encoding='utf-8', errors='replace')
    values = [float(value) for value in re.findall(
        r'Episode_Reward/edge_contact:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', text)]
    require(len(values) == 30, 'edge_contact reward telemetry incomplete')
    if weight == -1.0:
        require(min(values) < 0.0, 'edge_contact negative reward never activated')
    else:
        require(all(value == 0.0 for value in values), 'control has nonzero edge penalty')
    return min(values)


def run_pilot(workspace: Path, root: Path, manifest: dict, config: dict, spec: dict,
              env: dict[str, str], paths: dict, inputs: dict) -> dict:
    with paths['plan'].open('xb') as stream:
        stream.write(pilot_v1323.json_bytes(inputs['plan']))
    with paths['control_toml'].open('xb') as stream:
        stream.write(inputs['control_toml'])
    require(digest(paths['plan']) == spec['plan_sha256'] and
            digest(paths['control_toml']) == spec['control_toml_sha256'],
            'pilot generated plan/TOML drift')
    arms = {}
    for arm in pilot_v1323.ARMS:
        ensure_idle()
        reward = (paths['control_toml'] if arm == 'control'
                  else root / manifest['reward_config']['path'])
        require(digest(reward) == spec[f'{arm}_toml_sha256'], 'arm reward SHA drift')
        run_id = paths[f'{arm}_run_id']
        environment = dict(env, GO1_RUN_ID=run_id, GO1_TASK_PLAN=str(paths['plan']),
                           GO1_REWARD_CONFIG_PATH=str(reward))
        run_logged(pilot_v1323.train_command(workspace, config, spec, arm,
                                             paths['plan'], inputs['checkpoint']),
                   paths[f'{arm}_train_log'], environment, workspace)
        stage_path = paths[f'{arm}_stage_report']
        require(stage_path.is_file() and not stage_path.is_symlink(), 'pilot stage report absent')
        stage = json.loads(stage_path.read_text(encoding='utf-8'))
        require(stage.get('run') == run_id and stage.get('iterations') == 30 and
                stage.get('task_plan') == str(paths['plan']) and
                type(stage.get('episodes')) is int and stage['episodes'] > 0 and
                isinstance(stage.get('environments'), dict) and
                len(stage['environments']) == 64, 'pilot training report mismatch')
        model = pilot_v1323.latest_checkpoint(paths[f'{arm}_checkpoint_dir'])
        model_sha = digest(model)
        activation = _edge_activation(paths[f'{arm}_train_log'], 0.0 if arm == 'control' else -1.0)
        ensure_idle()
        eval_env = dict(environment, GO1_RUN_ID=paths[f'{arm}_eval_run_id'])
        run_logged(pilot_v1323.eval_command(workspace, config, spec, arm,
                                            model, paths[f'{arm}_trace']),
                   paths[f'{arm}_eval_log'], eval_env, workspace)
        report_path, trace_path = paths[f'{arm}_eval_report'], paths[f'{arm}_trace']
        require(report_path.is_file() and trace_path.is_file(), 'pilot report/trace absent')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(report.get('seed') == 2028 and report.get('checkpoint_sha256') == model_sha and
                report.get('stage') == 'stage1' and report.get('terrain') == 'lp_acrl',
                'pilot evaluation provenance mismatch')
        rows = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
        first = pilot_v1323.summarize(report, rows, arm, 64)
        arms[arm] = {'run_id': run_id, 'edge_contact_weight': 0.0 if arm == 'control' else -1.0,
                     'edge_reward_min': activation, 'reward_toml_sha256': digest(reward),
                     'checkpoint_path': str(model), 'checkpoint_sha256': model_sha,
                     'stage_report_sha256': digest(stage_path),
                     'training_log_sha256': digest(paths[f'{arm}_train_log']),
                     'evaluation_log_sha256': digest(paths[f'{arm}_eval_log']),
                     'evaluation_report_sha256': digest(report_path),
                     'evaluation_trace_sha256': digest(trace_path),
                     'training_episodes_repeated_not_gate': stage['episodes'],
                     'evaluation_first_episode': first}
    require(digest(inputs['checkpoint']) == spec['checkpoint_sha256'] and
            digest(paths['plan']) == spec['plan_sha256'] and
            digest(paths['control_toml']) == spec['control_toml_sha256'] and
            digest(root / manifest['reward_config']['path']) == spec['treatment_toml_sha256'],
            'pilot pinned inputs changed during arms')
    control, treatment = (arms[arm]['evaluation_first_episode'] for arm in pilot_v1323.ARMS)
    return {'arms': arms,
            'pilot_completion_delta': treatment['first_episode_completion_rate'] -
                                      control['first_episode_completion_rate'],
            'safety_regression_observed': any(treatment[key] > control[key] for key in
                ('fallen_rate', 'collision_rate', 'out_of_bounds_rate', 'fell_into_gap_rate')),
            'plan_sha256': spec['plan_sha256'],
            'control_toml_sha256': spec['control_toml_sha256'],
            'treatment_toml_sha256': spec['treatment_toml_sha256'],
            'initial_checkpoint_sha256': spec['checkpoint_sha256'],
            'train_receipt_sha256': inputs['train_receipt_sha256'],
            'test_receipt_sha256': inputs['test_receipt_sha256'],
            'smoke_receipt_sha256': inputs['smoke_receipt_sha256'],
            'automatic_full_training_authorized': False,
            'final_holdout': False, 'full_training_started': False}


def execute_mode(mode: str, workspace: Path, root: Path, manifest: dict,
                 config: dict, spec: dict, environment: dict, paths: dict,
                 inputs: dict, rows: list[tuple[Path, Path, str | None, str]]) -> dict:
    """Actual test/smoke/pilot branch under the same verified overlay lifetime."""
    with Overlay(rows, workspace) as overlay:
        if mode == 'test':
            result = run_tests(workspace, root, manifest, environment)
        elif mode == 'smoke':
            result = run_smoke(workspace, root, manifest, environment)
        elif mode == 'pilot':
            result = run_pilot(workspace, root, manifest, config, spec,
                               environment, paths, inputs)
        else:
            raise ValueError('unsupported simulator mode')
    require(overlay.restored, 'overlay not restored')
    return result


def run(mode: str, root: Path, workspace: Path) -> None:
    require(os.environ.get('V1323_RUN_AUTHORIZED') == mode_authorization(mode),
            'explicit authorization absent')
    import fcntl
    lock = workspace / 'reports/v7_sim.lock'
    require(lock.is_file() and not lock.is_symlink() and os.fstat(9).st_ino == lock.stat().st_ino,
            'exclusive simulator lock fd missing')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight(workspace, root, mode)
    manifest = json.loads((root / 'configs/manifest.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'configs/teacher_entrance.json').read_text(encoding='utf-8'))
    spec = json.loads((root / 'configs/pilot_execution.json').read_text(encoding='utf-8')) if mode == 'pilot' else None
    spec_sha = digest(root / 'configs/pilot_execution.json') if mode == 'pilot' else None
    paths, inputs = (pilot_v1323.validate_inputs(root, workspace, manifest, config, spec)
                     if mode == 'pilot' else (None, None))
    rows = targets(root, workspace, manifest, mode)
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    environment = common_environment(workspace, manifest)
    require(not list((workspace / 'reports').glob('v1323-overlay-*')),
            'prior incomplete V132.3 overlay backup; manual audit required')

    def interrupted(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        process = _active_process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise KeyboardInterrupt(f'signal {signum}')

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = execute_mode(mode, workspace, root, manifest, config, spec,
                              environment, paths, inputs, rows)
        require(digest(root / manifest['reward_config']['path']) == manifest['reward_config']['sha256'],
                'reward TOML changed during run')
        if mode == 'pilot':
            require(digest(root / 'configs/pilot_execution.json') == spec_sha,
                    'pilot spec changed during run')
        receipt = (paths['receipt'] if mode == 'pilot' else root / 'reports' /
                   names['test_receipt' if mode == 'test' else 'smoke_receipt'])
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'pilot-diagnostic-only' if mode == 'pilot' else 'passed',
                       'mode': mode, 'restored': True, 'manifest_sha256': BASE_SHA,
                       'pilot_spec_sha256': spec_sha if mode == 'pilot' else None,
                       'reward_sha256': manifest['reward_config']['sha256'], **result}, stream)
            stream.write('\n')
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    try:
        run(sys.argv[1], Path(__file__).resolve().parents[1], Path('/config/Desktop/sim'))
    except (IndexError, KeyError, OSError, ValueError, RuntimeError,
            subprocess.CalledProcessError, KeyboardInterrupt) as error:
        print(f'V1323_BLOCKED: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
