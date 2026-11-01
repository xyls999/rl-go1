#!/usr/bin/env python3
"""Guarded, explicit V132.2 test/PhysX modes; never starts training.

The caller owns the v7_sim.lock for this entire process. A mode runs only when
all live files have freshly matched their pinned baseline hashes.
"""
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

from preflight_v1322 import (digest, gpu_usage_mib, require, validate_components,
                            validate_mode_outputs)

FAMILIES = (('entrance', 'combined_part_entrance'), ('junction', 'combined_junction'),
            ('long', 'gaps_long'), ('tire', 'tire_field'))
_active_process: subprocess.Popen | None = None


def attempt_artifacts(attempt: int) -> dict[str, str]:
    require(type(attempt) is int and 1 <= attempt <= 99, 'invalid test attempt')
    tag = f'a{attempt:02d}'
    return {'test_log': f'v1322_tests_{tag}.log',
            'focus_log': f'v1322_focus_{tag}.log',
            'test_receipt': f'v1322_test_{tag}.receipt',
            'smoke_receipt': f'v1322_physx_{tag}.receipt',
            'smoke_prefix': f'v1322_smoke_{tag}_'}


def targets(root: Path, workspace: Path, manifest: dict, mode: str) -> list[tuple[Path, Path, str | None, str]]:
    rows = [(root / row['candidate'], workspace / row['live'],
             manifest['baseline'].get(row['name']), row['candidate_sha256'])
            for row in manifest['sources']]
    # The V132.2 TOML is an immutable pinned input at its unique staged path,
    # not a self-copy overlay. Never modify the unrelated live V131 TOML.
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
        require(not live.is_symlink(), 'symlinked overlay target: ' + str(live))
        for parent in live.parents:
            if parent == workspace:
                break
            require(not parent.is_symlink(), 'symlinked overlay parent: ' + str(parent))
        if original is None:
            require(not live.exists(), 'unexpected unpinned overlay target: ' + str(live))
        else:
            require(live.is_file() and digest(live) == original,
                    'unpinned live baseline: ' + str(live))


def atomic_copy(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy to a private sibling, verify bytes, then swap atomically."""
    handle, name = tempfile.mkstemp(prefix='.v1322-', dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        require(digest(temporary) == expected_sha256, 'temporary copy SHA mismatch: ' + str(source))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class Overlay:
    def __init__(self, rows: list[tuple[Path, Path, str | None, str]], workspace: Path):
        self.rows = rows
        self.workspace = workspace
        self.backup: Path | None = None
        self.installed: list[tuple[Path, str | None, str]] = []
        self.restored = False

    def __enter__(self):
        check_targets(self.rows, self.workspace)
        self.backup = Path(tempfile.mkdtemp(prefix='v1322-overlay-', dir=self.workspace / 'reports'))
        try:
            for index, (_, live, original, _) in enumerate(self.rows):
                if original is not None:
                    saved = self.backup / str(index)
                    shutil.copy2(live, saved)
                    require(digest(saved) == original, 'backup SHA mismatch: ' + str(live))
            for candidate, live, original, expected in self.rows:
                live.parent.mkdir(parents=True, exist_ok=True)
                # Record BEFORE copying, so interruption during a copy is restored.
                self.installed.append((live, original, expected))
                atomic_copy(candidate, live, expected)
                require(digest(live) == expected, 'overlay SHA mismatch: ' + str(live))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        issues = []
        for live, original, expected in reversed(self.installed):
            try:
                if original is None:
                    # Never delete a file that another actor modified unexpectedly.
                    if live.exists() and digest(live) == expected:
                        live.unlink()
                    elif live.exists():
                        raise RuntimeError('unowned modified helper: ' + str(live))
                    if live.exists():
                        raise RuntimeError('created helper remains: ' + str(live))
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
    return environment


def smoke_command(workspace: Path, family: str, run_id: str, seed: int) -> list[str]:
    return [str(workspace / 'sim.sh'), 'check', '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--device', 'cuda:0', '--num-envs', '512', '--seed', str(seed),
            '--speed-min', '0.35', '--speed-max', '0.8']


def isaac_process_in_listing(listing: str) -> bool:
    """Match versioned Python, torchrun, wrappers and Isaac Kit, not unrelated Python."""
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
    usage = gpu_usage_mib()
    require(len(usage) == 4 and all(0 <= item <= 256 for item in usage), 'GPU busy or count != 4')
    listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True,
                             text=True, check=True).stdout
    require(not isaac_process_in_listing(listing), 'Isaac process active')


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
                str(root / 'tests'), '-p', 'test_*v1322.py', '-v'], focus, env, workspace)
    focus_text = focus.read_text(encoding='utf-8', errors='replace')
    require(re.search(r'Ran \d+ tests?', focus_text) is not None and
            re.search(r'^OK\s*$', focus_text, re.MULTILINE) is not None,
            'V132.2 focused tests did not complete')
    log = root / 'reports' / names['test_log']
    run_logged([str(workspace / 'sim.sh'), 'test', '-q'], log, env, workspace)
    text = log.read_text(encoding='utf-8', errors='replace')
    require(re.search(r'\b\d+ passed\b', text) is not None and
            re.search(r'\b\d+ failed\b', text) is None,
            'project test suite did not report complete success')
    return {'log_sha256': digest(log), 'focus_sha256': digest(focus)}


def run_smoke(workspace: Path, root: Path, manifest: dict, env: dict[str, str]) -> dict:
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    tag = f"a{manifest['execution']['test_attempt']:02d}"
    passed = {}
    for index, (family, kind) in enumerate(FAMILIES):
        ensure_idle()
        plan = root / 'reports' / f'v1322_smoke_plan_{tag}_{family}.json'
        with plan.open('x', encoding='utf-8') as stream:
            json.dump({'families': [kind] * 512, 'difficulty': [.6] * 512,
                       'speed_level': [2] * 512, 'speed_min': .35, 'speed_max': .8}, stream)
        run_name = f"v1322-{manifest['execution']['run_id']}-{tag}-{family}"
        family_env = dict(env, GO1_TASK_PLAN=str(plan), GO1_RUN_ID=run_name)
        log = root / 'reports' / f"{names['smoke_prefix']}{family}.log"
        run_logged(smoke_command(workspace, family, run_name, 9700 + index), log,
                   family_env, workspace)
        report_path = workspace / 'reports' / f'check_{run_name}.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(report.get('passed') is True and report.get('num_envs') == 512 and
                report.get('policy_shape') == [512, 1073] and
                report.get('critic_shape') == [512, 1080], 'PhysX check failed: ' + family)
        passed[family] = {'log_sha256': digest(log), 'report_sha256': digest(report_path)}
    return {'families': passed}


def run(mode: str, root: Path, workspace: Path) -> None:
    require(mode in ('test', 'smoke'), 'training and gate not implemented in this runner')
    require(os.environ.get('V1322_RUN_AUTHORIZED') == 'test-smoke-only', 'explicit authorization absent')
    # fd 9 must be inherited from the shell after nontruncating flock acquisition;
    # refuse direct Python invocation that would bypass the external simulator lock.
    import fcntl
    lock_path = workspace / 'reports/v7_sim.lock'
    require(os.fstat(9).st_ino == lock_path.stat().st_ino, 'exclusive simulator lock fd missing')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'configs/manifest.json').read_text())
    config = json.loads((root / 'configs/teacher_entrance.json').read_text())
    baseline = json.loads((root / 'provenance/v1321_teacher_entrance.json').read_text())
    validate_components(workspace, root, manifest, config, baseline, production=True, mode=mode)
    require(manifest['baseline']['cli.py'] is not None,
            'remote active CLI hash has not been verified and pinned')
    require(root == workspace / 'v128_work/v132_randomized/v1322', 'runner must use staged remote root')
    require((workspace / 'reports/v7_sim.lock').is_file(), 'external simulator lock missing')
    require(not list((workspace / 'reports').glob('v1322-overlay-*')),
            'prior overlay backup remains; manual source audit required')
    ensure_idle()
    rows = targets(root, workspace, manifest, mode)
    check_targets(rows, workspace)
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    if mode == 'smoke':
        receipt = root / 'reports' / names['test_receipt']
        require(receipt.is_file(), 'full project test receipt absent')
        previous = json.loads(receipt.read_text())
        require(previous.get('status') == 'passed' and previous.get('restored') is True and
                previous.get('manifest_sha256') == digest(root / 'configs/manifest.json') and
                previous.get('log_sha256') == digest(root / 'reports' / names['test_log']) and
                previous.get('focus_sha256') == digest(root / 'reports' / names['focus_log']) and
                previous.get('reward_sha256') == manifest['reward_config']['sha256'],
                'full project test receipt/source hash mismatch')
    environment = common_environment(workspace, manifest)
    # Running code must not inherit a display or any automatic training hooks.
    environment.pop('DISPLAY', None)
    environment.pop('WAYLAND_DISPLAY', None)
    def interrupted(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        process = _active_process
        if process is not None and process.poll() is None:
            # Keep the lock and overlay installed until the entire Isaac process
            # group has stopped. Never restore live imports beneath a running sim.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise KeyboardInterrupt(f'signal {signum}')
    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with Overlay(rows, workspace) as overlay:
            result = (run_tests(workspace, root, manifest, environment) if mode == 'test' else
                      run_smoke(workspace, root, manifest, environment))
        require(overlay.restored, 'overlay not restored')
        require(digest(root / manifest['reward_config']['path']) == manifest['reward_config']['sha256'],
                'reward TOML changed during run')
        receipt = root / 'reports' / names['test_receipt' if mode == 'test' else 'smoke_receipt']
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'passed', 'mode': mode, 'restored': True,
                       'manifest_sha256': digest(root / 'configs/manifest.json'),
                       'reward_sha256': manifest['reward_config']['sha256'], **result}, stream)
            stream.write('\n')
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    try:
        mode = sys.argv[1]
        run(mode, Path(__file__).resolve().parents[1], Path('/config/Desktop/sim'))
    except (IndexError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError,
            KeyboardInterrupt) as error:
        print(f'V1322_BLOCKED: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
