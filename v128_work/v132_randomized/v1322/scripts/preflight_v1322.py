#!/usr/bin/env python3
"""CPU-only V132.2 preparation checks. No remote mutation or Isaac startup."""
from __future__ import annotations

import hashlib
import json
import math
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


def validate_pilot(plan: dict, config: dict) -> None:
    require(plan['environment_count'] == 64 and plan['terrain_family'] == config['gate_family'],
            'pilot terrain or environment count changed')
    require(plan['only_reward_variable'] == 'supported_short_weight', 'pilot changes more than one variable')
    control, treatment = plan['arms']
    require(set(control) == set(treatment) and
            all(control[k] == treatment[k] for k in control if k not in {'name', 'supported_short_weight'})
            and control['supported_short_weight'] == 0 and treatment['supported_short_weight'] == .8,
            'pilot does not isolate supported_short')
    require(control['safe_segment_weight'] == 6.0 and
            control['initial_checkpoint'] == treatment['initial_checkpoint'] and
            control['seed'] == treatment['seed'], 'pilot checkpoint/seed/safe_segment parity failed')
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
        require(slot['success'] == int(bool(row['success'])) and
                all(slot.get(key, 0) == int(bool(row[key]))
                    for key in ('collision', 'out_of_bounds', 'fell_into_gap')),
                'terminal flags differ across report and trace')
        require(not bool(row['success']) or success, 'success credited without all gaps/hard safety')
        require(math.isfinite(float(row['course_x'])), 'nonfinite termination location')
        completions += int(success)
    require(sum(slot['episodes'] for slot in environments.values()) == count,
            'episode total differs from environment slots')
    require(math.isclose(report.get('completion_rate', float('nan')),
                         completions / count, abs_tol=1e-9), 'completion numerator mismatch')


def validate_components(workspace: Path, root: Path, manifest: dict, config: dict,
                        baseline: dict, *, production: bool) -> None:
    validate_config(config, baseline)
    validate_pilot(json.loads((root / 'configs/pilot_plan.json').read_text()), config)
    require(digest(root / 'configs/teacher_entrance.json') == manifest['configuration_sha256'],
            'V132.2 config hash drift')
    provenance = manifest['provenance']
    require(digest(root / 'provenance/v1321_manifest.json') == provenance['v1321_manifest_sha256']
            and digest(root / 'provenance/v1321_teacher_entrance.json') == provenance['v1321_config_sha256'],
            'V132.1 provenance drift')
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
        live_path = workspace / live
        expected = manifest['baseline'].get(row['name'])
        if expected is None:
            require(not live_path.exists(), 'unexpected unpinned live helper: ' + live)
        else:
            require(live_path.is_file() and digest(live_path) == expected,
                    'remote live SHA mismatch: ' + live)
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
    if production:
        require(config['gate_holdout_seed'] is not None and
                config['gate_holdout_seed'] == execution['holdout_seed'] and
                execution['seed_audit_sha256'], 'final holdout seed unaudited')
        require(config['init_checkpoint'] and config['init_checkpoint'] == execution['initial_checkpoint']
                and execution['checkpoint_sha256'], 'initial checkpoint not owner-approved')
        require(digest(workspace / config['init_checkpoint']) == execution['checkpoint_sha256'],
                'initial checkpoint SHA mismatch')
        require(execution['remote_attestation_sha256'] and all(manifest['baseline'].values()),
                'remote live SHA/GPU/lock attestation pending')


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
        validate_components(workspace, root, manifest, config, baseline, production=True)
        usage = gpu_usage_mib()
        require(all(0 <= value <= 256 for value in usage), 'GPU memory exceeds 256 MiB')
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f'PREFLIGHT_BLOCKED: {exc}', file=sys.stderr)
        return 1
    # This preparation version is never allowed to approve a live run.
    print('PREFLIGHT_BLOCKED: Task 3 remote integration and authorization pending', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
