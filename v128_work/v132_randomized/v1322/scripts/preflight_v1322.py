#!/usr/bin/env python3
"""CPU-only V132.2 preparation checks. No remote mutation or Isaac startup."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict, baseline: dict) -> None:
    require(config['training_roster'] == baseline['training_roster'] == 'single:combined_part_entrance',
            'entrance training roster changed')
    require(config['gate_family'] == baseline['gate_family'] == 'combined_part_entrance',
            'entrance gate family changed')
    for key in ('num_envs', 'episode_seconds', 'speed_min', 'speed_max',
                'gate_speed_min', 'gate_speed_max', 'difficulty_max', 'phases'):
        require(config[key] == baseline[key], f'train/eval baseline changed: {key}')
    require(config['num_envs'] == 512 and config['speed_min'] == config['gate_speed_min'] == .35
            and config['speed_max'] == config['gate_speed_max'] == .8,
            '512 environment/speed-band parity failed')
    require(config['max_stages'] == 1 and config['warmup_first'] == 50
            and config['phases'][0]['iterations'] == 650, 'proposed 50+600 stage altered')
    require(config['gate']['completion_gate'] == .9 and config['require_deterministic_gate'],
            'hard entrance gate altered')
    for key in ('state', 'sampler_state', 'results_file'):
        value = config[key]
        require(value != baseline[key] and value.startswith('v128_work/v132_randomized/v1322/reports/')
                and '..' not in Path(value).parts, f'old or unsafe output path: {key}')
    seed = config.get('gate_holdout_seed')
    require(seed is None or (type(seed) is int and seed > 0
                             and seed not in {42, 2026, 2028}), 'invalid final holdout seed')


def validate_pilot(plan: dict, config: dict, manifest: dict) -> None:
    require(plan['environment_count'] == 64 and plan['terrain_family'] == config['gate_family'],
            'pilot terrain or environment count changed')
    require(plan['only_reward_variable'] == 'supported_short_weight', 'pilot changes more than one variable')
    approved_geometry = next(row['candidate_sha256'] for row in manifest['sources']
                             if row['name'] == 'geometry.py')
    execution = manifest['execution']
    require(plan['geometry_sha256'] == approved_geometry, 'pilot geometry hash mismatch')
    require(plan['initial_checkpoint'] == config['init_checkpoint'] == execution['initial_checkpoint']
            and plan['checkpoint_sha256'] == execution['checkpoint_sha256'],
            'pilot checkpoint differs from approved config/manifest')
    require(type(plan['seed']) is int and plan['seed'] == execution['train_seed'],
            'pilot seed differs from approved manifest')
    control, treatment = plan['arms']
    require(set(control) == set(treatment) and
            all(control[k] == treatment[k] for k in control if k not in {'name', 'supported_short_weight'})
            and control['supported_short_weight'] == 0 and treatment['supported_short_weight'] == .8,
            'pilot does not isolate supported_short')
    require(control['safe_segment_weight'] == 6.0 and
            all(arm['initial_checkpoint'] == plan['initial_checkpoint'] and
                arm['seed'] == plan['seed'] for arm in (control, treatment)),
            'pilot checkpoint/seed/safe_segment parity failed')
    require(plan['strict_gate_command_contract']['episodes'] == 512 and
            plan['strict_gate_command_contract']['strict_first_episode'] is True,
            'strict gate contract absent')


def validate_gate(report: dict, trace: list[dict], count: int, checkpoint_sha256: str) -> None:
    """Additional fail-closed postflight: not a substitute for Task 1 driver validation."""
    expected = {str(i) for i in range(count)}
    environments = report.get('environments', {})
    require(report.get('episodes') == count and set(environments) == expected,
            'report lacks exactly one entry per slot')
    require(len(trace) == count and
            {str(row.get('env_id')) for row in trace} == expected and
            len({str(row.get('env_id')) for row in trace}) == count,
            'trace does not cover exactly one first episode per slot')
    require(report.get('checkpoint_sha256') == checkpoint_sha256,
            'report checkpoint digest differs from approved checkpoint')
    completions = 0
    for row in trace:
        slot = environments[str(row['env_id'])]
        require(row.get('episode_index') == 0 and slot.get('episodes') == 1,
                'repeated episode in report or trace')
        for key in ('success', 'collision', 'fallen', 'out_of_bounds', 'fell_into_gap',
                    'required_gap_count', 'credited_gap_count', 'course_x'):
            require(key in row and row[key] is not None, f'trace lacks terminal field: {key}')
        required, credited = row['required_gap_count'], row['credited_gap_count']
        require(type(required) is int and type(credited) is int and 2 <= required <= 5
                and 0 <= credited <= required, 'invalid entrance gap credit count')
        hard_clean = not any(row[k] for k in ('collision', 'fallen', 'out_of_bounds', 'fell_into_gap'))
        success = bool(row['success']) and hard_clean and credited == required
        terminal_keys = ('success', 'collision', 'fallen', 'out_of_bounds', 'fell_into_gap')
        require(all(key in slot and slot[key] is not None for key in terminal_keys),
                'report missing per-slot terminal flags')
        require(all(slot[key] == int(bool(row[key])) for key in terminal_keys),
                'terminal flags differ across report and trace')
        require(not bool(row['success']) or success, 'success credited without all gaps/hard safety')
        require(math.isfinite(float(row['course_x'])), 'nonfinite termination location')
        completions += int(success)
    require(sum(slot['episodes'] for slot in environments.values()) == count,
            'episode total differs from environment slots')
    require(math.isclose(report.get('completion_rate', float('nan')),
                         completions / count, abs_tol=1e-9), 'completion numerator mismatch')


def validate_live_baseline(workspace: Path, manifest: dict, *, production: bool) -> None:
    for row in manifest['sources']:
        live_path = workspace / row['live']
        expected = manifest['baseline'].get(row['name'])
        require(not live_path.is_symlink() and not any(parent.is_symlink() for parent in live_path.parents
                                                      if parent != workspace and workspace in parent.parents),
                'symlinked live source: ' + row['live'])
        if expected is None:
            require(not live_path.exists(), 'unexpected unpinned live helper: ' + row['live'])
        else:
            require(live_path.is_file() and digest(live_path) == expected,
                    'remote live SHA mismatch: ' + row['live'])
        if production:
            require(expected is not None or row['name'] not in ('cli.py', 'runtime.py', 'geometry.py', 'config.py'),
                    'required live SHA unpinned: ' + row['live'])


def validate_mode_outputs(workspace: Path, mode: str, run_id: str) -> None:
    if mode not in ('test', 'smoke'):
        raise ValueError('production mode not implemented: ' + mode)
    names = (['v1322_tests.log', 'v1322_focus.log', 'v1322_test.receipt'] if mode == 'test' else
             ['v1322_physx.receipt'] + [f'v1322_smoke_{family}.log' for family in
                                     ('entrance', 'junction', 'long', 'tire')] +
             [f'v1322_smoke_plan_{family}.json' for family in
              ('entrance', 'junction', 'long', 'tire')])
    for name in names:
        path = workspace / 'v128_work/v132_randomized/v1322/reports' / name
        require(not path.exists() and not path.is_symlink(), 'mode output already exists: ' + str(path))
    if mode == 'smoke':
        for family in ('entrance', 'junction', 'long', 'tire'):
            path = workspace / 'reports' / f'check_v1322-{run_id}-{family}.json'
            require(not path.exists() and not path.is_symlink(), 'smoke check already exists: ' + str(path))


def validate_components(workspace: Path, root: Path, manifest: dict, config: dict,
                        baseline: dict, *, production: bool, mode: str = 'final') -> None:
    validate_config(config, baseline)
    validate_pilot(json.loads((root / 'configs/pilot_plan.json').read_text()), config, manifest)
    require(digest(root / 'configs/teacher_entrance.json') == manifest['configuration_sha256'],
            'V132.2 config hash drift')
    provenance = manifest['provenance']
    require(digest(root / 'provenance/v1321_manifest.json') == provenance['v1321_manifest_sha256']
            and digest(root / 'provenance/v1321_teacher_entrance.json') == provenance['v1321_config_sha256'],
            'V132.1 provenance drift')
    require(digest(root / 'provenance/remote_live_audit_v1322.txt') ==
            manifest['execution']['remote_attestation_sha256'], 'read-only remote audit drift')
    require(digest(root / 'configs/rewards_stage1_gaps_v131_landing_stability.toml') ==
            manifest['reward_config']['sha256'], 'reward TOML hash drift')
    seen = set()
    for row in manifest['sources']:
        live = row['live']
        require(live not in seen and live.startswith('source/go1_hier_sim/go1_hier_sim/')
                and '..' not in Path(live).parts and row['candidate'] == live,
                'unsafe/duplicate source target')
        seen.add(live)
        require(digest(root / row['candidate']) == row['candidate_sha256'],
                'candidate source hash drift: ' + live)
    validate_live_baseline(workspace, manifest, production=production)
    execution = manifest['execution']
    run_id = execution['run_id']
    require(run_id.startswith('go1-v1322-') and '/' not in run_id and '..' not in run_id,
            'unsafe run ID')
    outputs = [config[key] for key in ('state', 'sampler_state', 'results_file')]
    outputs += [f'checkpoints/stage1/{run_id}', f'reports/stage_{run_id}.json',
                f'reports/lp_plan_{run_id}.json', f'reports/lp_gate_plan_{run_id}.json',
                f'reports/v1322_{run_id}_trace.jsonl']
    require(len(set(outputs)) == len(outputs), 'duplicate output paths')
    for relative in outputs:
        target = workspace / relative
        require(not target.exists() and not target.is_symlink(), 'output path already exists: ' + relative)
    env = execution['train_and_gate_env']
    require(env['GO1_REWARD_CONFIG_PATH'] ==
            'v128_work/v132_randomized/v1322/configs/rewards_stage1_gaps_v131_landing_stability.toml'
            and env['GO1_V128_FILTER_OBS'] == '1' and env['GO1_V29_FILTER_OBS'] == '0',
            'train/gate reward or observation mask changed')
    require(not any(key in env for key in ('GO1_V29_TEACHER_DISTILL', 'GO1_WATCHDOG', 'GO1_AUTO_P2')),
            'forbidden automation enabled')
    if production and mode in ('test', 'smoke'):
        validate_mode_outputs(workspace, mode, run_id)
    if production and mode not in ('test', 'smoke'):
        require(config['gate_holdout_seed'] is not None and
                config['gate_holdout_seed'] == execution['holdout_seed'] and
                execution['seed_audit_sha256'], 'final holdout seed unaudited')
        require(config['init_checkpoint'] and config['init_checkpoint'] == execution['initial_checkpoint']
                and execution['checkpoint_sha256'], 'initial checkpoint not owner-approved')
        require(digest(workspace / config['init_checkpoint']) == execution['checkpoint_sha256'],
                'initial checkpoint SHA mismatch')
        require(execution['remote_attestation_sha256'] and
                all(manifest['baseline'][name] for name in ('cli.py', 'runtime.py', 'geometry.py', 'config.py')),
                'remote live SHA/GPU/lock attestation pending')


def validate_test_receipt(root: Path, manifest: dict) -> None:
    receipt = root / 'reports/v1322_test.receipt'
    require(receipt.is_file(), 'full project test receipt absent')
    previous = json.loads(receipt.read_text(encoding='utf-8'))
    require(previous.get('status') == 'passed' and previous.get('restored') is True and
            previous.get('manifest_sha256') == digest(root / 'configs/manifest.json') and
            previous.get('log_sha256') == digest(root / 'reports/v1322_tests.log') and
            previous.get('focus_sha256') == digest(root / 'reports/v1322_focus.log') and
            previous.get('reward_sha256') == manifest['reward_config']['sha256'],
            'full project test receipt/source hash mismatch')


def validate_read_only_preflight(workspace: Path, root: Path, manifest: dict,
                                 config: dict, baseline: dict, *, mode: str,
                                 gpu_mib: list[int]) -> None:
    """Permit a test/smoke source audit without miscertifying an unchosen final seed."""
    require(mode in ('test', 'smoke'), 'only test/smoke preflight may omit the final seed')
    validate_components(workspace, root, manifest, config, baseline, production=True, mode=mode)
    require(len(gpu_mib) == 4 and all(0 <= value <= 256 for value in gpu_mib),
            'GPU busy or count != 4')
    from production_v1322 import check_targets, targets
    check_targets(targets(root, workspace, manifest, mode), workspace)
    if mode == 'smoke':
        validate_test_receipt(root, manifest)


def gpu_usage_mib() -> list[int]:
    raw = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True,
                         text=True, check=True).stdout
    lines = [line.strip().split(',') for line in raw.splitlines() if line.strip()]
    require(bool(lines), 'no GPU information')
    return [int(parts[1].strip()) for parts in lines]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    workspace = Path('/config/Desktop/sim')
    try:
        manifest = json.loads((root / 'configs/manifest.json').read_text())
        config = json.loads((root / 'configs/teacher_entrance.json').read_text())
        baseline = json.loads((root / 'provenance/v1321_teacher_entrance.json').read_text())
        require(workspace.is_dir(), 'remote workspace not mounted; remote check pending')
        mode_arg = sys.argv[1] if len(sys.argv) > 1 else 'preflight'
        require(mode_arg in ('preflight', 'preflight-test', 'preflight-smoke'),
                'unknown read-only preflight mode')
        import fcntl
        lock = workspace / 'reports/v7_sim.lock'
        require(lock.is_file() and not lock.is_symlink() and os.fstat(9).st_ino == lock.stat().st_ino,
                'exclusive simulator lock fd missing')
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = gpu_usage_mib()
        from production_v1322 import isaac_process_in_listing
        listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True,
                                 text=True, check=True).stdout
        require(not isaac_process_in_listing(listing), 'Isaac process active')
        if mode_arg == 'preflight':
            validate_components(workspace, root, manifest, config, baseline,
                                production=True, mode='final')
            # A final holdout seed still needs separate provenance and approval.
            raise ValueError('final gate and training are not implemented')
        mode = mode_arg.removeprefix('preflight-')
        validate_read_only_preflight(workspace, root, manifest, config, baseline,
                                     mode=mode, gpu_mib=usage)
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f'PREFLIGHT_BLOCKED: {exc}', file=sys.stderr)
        return 1
    print(f'PREFLIGHT_PASS mode={mode} read_only=true final_gate_authorized=false', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
