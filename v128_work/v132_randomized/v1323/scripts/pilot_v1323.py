"""Pinned, paired 64-environment speed-autonomy diagnostic; never a gate.

Both arms start from failed V132.2 model_649 with 40 critic-only and 160 actor
iterations. Treatment receives only a forward-direction command cue and no
reward target for a chosen speed magnitude.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tomllib

from preflight_v1323 import OLD_REWARD_SHA, digest, require

BASE_SHA = '59281cf96bcbf0a5abd646b399e7bc748ebdc336613a6fef07d4fc1186db396d'
SOURCE_PLAN_SHA = 'e16717f32563c9a1a9db72316a6eb85a51b557975080ae7c9066a71494f6e18b'
MODEL_SHA = '1a120f37c23f621e9f08772b1f6fc8520fabc4fb642b1da6342e08a96c7793f9'
TRAIN_RECEIPT_SHA = 'b728dee938262b83d9bcb17f8f1e327eaf3adea0eb0d1566fc6f064fa1c9b185'
ARMS = ('control', 'treatment')
EVALUATION_EPISODE_QUOTA = 256  # 256*40 steps cover the 200s/0.02s maximum episode.


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def build_plan(root: Path, config: dict, spec: dict) -> dict:
    archived = root / 'provenance/v1321_gate_plan.json'
    require(digest(archived) == SOURCE_PLAN_SHA == spec['source_plan_sha256'],
            'pilot source plan provenance changed')
    original = json.loads(archived.read_text(encoding='utf-8'))
    n = spec['num_envs']
    require(n == 64 and original['families'] == [config['gate_family']] * 512 and
            original['speed_min'] == config['speed_min'] and
            original['speed_max'] == config['speed_max'], 'pilot entrance plan/band changed')
    per_slot = ('families', 'difficulty', 'difficulty_level', 'speed_level', 'instances')
    require(all(len(original[key]) == 512 for key in per_slot), 'pilot source plan incomplete')
    plan = {key: (value[:n] if key in per_slot else value)
            for key, value in original.items()}
    require(all(row == [config['gate_family'], plan['difficulty_level'][index],
                        plan['speed_level'][index]]
                for index, row in enumerate(plan['instances'])), 'pilot plan slot mismatch')
    require(digest_bytes(json_bytes(plan)) == spec['plan_sha256'], 'pilot plan SHA mismatch')
    return plan


def control_toml(old_reward: bytes) -> bytes:
    """Add the new optional term at zero so the V132.3 loader can sum all terms."""
    marker = b'\r\n[edge_contact]\r\n'
    require(marker not in old_reward and old_reward.endswith(b'\r\n'),
            'V132.2 control reward format changed')
    return old_reward + marker + b'weight = 0.0\r\n'


def validate_tomls(control: bytes, treatment: bytes, spec: dict, manifest: dict) -> None:
    require(digest_bytes(treatment) == spec['treatment_toml_sha256'] ==
            manifest['reward_config']['sha256'] and
            digest_bytes(control) == spec['control_toml_sha256'], 'pilot TOML SHA mismatch')
    control_cfg = tomllib.loads(control.decode('utf-8'))
    treatment_cfg = tomllib.loads(treatment.decode('utf-8'))
    require(treatment_cfg['track_linear_velocity']['weight'] == 0.0 and
            treatment_cfg['forward_velocity']['weight'] == 0.0 and
            treatment_cfg['speed_error_l1']['weight'] == 0.0 and
            treatment_cfg['no_progress']['weight'] == 0.0 and
            treatment_cfg['forward_progress']['max_progress_speed'] == 1.2 and
            treatment_cfg['fallen']['weight'] == control_cfg['fallen']['weight'] == -12.0 and
            treatment_cfg['edge_contact']['weight'] == control_cfg['edge_contact']['weight'] == 0.0,
            'speed-autonomy reward or death penalty changed')
    for section, key in (('track_linear_velocity', 'weight'),
                         ('forward_velocity', 'weight'),
                         ('speed_error_l1', 'weight'),
                         ('no_progress', 'weight'),
                         ('forward_progress', 'max_progress_speed')):
        treatment_cfg[section][key] = control_cfg[section][key]
    treatment_cfg.pop('edge_contact')
    control_cfg.pop('edge_contact')
    require(treatment_cfg == control_cfg,
            'pilot reward differs beyond speed-autonomy fields')


def validate_against_v1322(control: bytes, old_reward: bytes) -> None:
    """Control is byte-for-byte V132.2 reward; death penalty stays unchanged."""
    baseline = tomllib.loads(old_reward.decode('utf-8'))
    candidate = tomllib.loads(control.decode('utf-8'))
    require(candidate.pop('edge_contact', None) == {'weight': 0.0} and
            candidate == baseline and candidate['fallen']['weight'] == -12.0,
            'pilot control differs from the V132.2 reward baseline')


def artifacts(workspace: Path, root: Path, spec: dict) -> dict:
    pilot = spec['pilot_id']
    require(pilot == 'go1-v1323-speed-autonomy-pilot-a02', 'unapproved pilot ID')
    result = {'plan': root / 'reports' / f'{pilot}_plan.json',
              'control_toml': root / 'reports' / f'{pilot}_control.toml',
              'treatment_toml': root / 'reports' / f'{pilot}_treatment.toml',
              'receipt': root / 'reports' / f'{pilot}.receipt'}
    for arm in ARMS:
        run_id = spec['run_ids'][arm]
        require(run_id == f'{pilot}-{arm}', 'unapproved pilot run ID')
        prefix = f'{arm}_'
        result.update({prefix + 'run_id': run_id,
                       prefix + 'checkpoint_dir': workspace / 'checkpoints/stage1' / run_id,
                       prefix + 'runs_dir': workspace / 'runs/stage1' / run_id,
                       prefix + 'stage_report': workspace / 'reports' / f'stage_{run_id}.json',
                       prefix + 'train_log': root / 'reports' / f'{run_id}_train.log',
                       prefix + 'eval_run_id': f'{run_id}-dev-seed2028',
                       prefix + 'eval_log': root / 'reports' / f'{run_id}_eval.log',
                       prefix + 'eval_report': workspace / 'reports' /
                           f'evaluate_stage1_{run_id}-dev-seed2028.json',
                       prefix + 'trace': root / 'reports' / f'{run_id}_dev_trace.jsonl'})
    return result


def validate_outputs(paths: dict) -> None:
    for key, path in paths.items():
        if key.endswith('run_id'):
            continue
        require(not any(parent.is_symlink() for parent in path.parents),
                'pilot output parent is symlinked: ' + str(path))
        require(not path.exists() and not path.is_symlink(), 'pilot output already exists: ' + str(path))


def validate_inputs(root: Path, workspace: Path, manifest: dict, config: dict,
                    spec: dict) -> tuple[dict, dict]:
    from production_v1323 import attempt_artifacts
    require(digest(root / 'configs/manifest.json') == spec['frozen_manifest_sha256'] == BASE_SHA,
            'pilot source manifest changed; rerun tests and PhysX')
    require(spec['final_holdout'] is False and
            spec['automatic_full_training_authorized'] is False and
            config['gate_holdout_seed'] is None and manifest['execution']['holdout_seed'] is None,
            'pilot cannot authorize a final gate or teacher training')
    require(spec['num_envs'] == 64 and spec['evaluation_episode_quota'] == 256 and
            spec['iterations'] == 200 and spec['critic_warmup_iterations'] == 40 and
            spec['train_seed'] == 42 and spec['development_evaluation_seed'] == 2028 and
            spec['test_attempt'] == manifest['execution']['test_attempt'] == 2,
            'pilot iteration, seed or test contract changed')
    require(config['init_checkpoint'] == manifest['execution']['initial_checkpoint'] ==
            'checkpoints/stage1/go1-v1322-real-entrance-s01-p1-d1/model_649.pt' and
            manifest['execution']['checkpoint_sha256'] == spec['checkpoint_sha256'] == MODEL_SHA,
            'pilot failed model_649 provenance changed')
    model = workspace / config['init_checkpoint']
    require(model.is_file() and not model.is_symlink() and digest(model) == MODEL_SHA,
            'pilot initial checkpoint missing or changed')
    old_root = workspace / 'v128_work/v132_randomized/v1322'
    old_reward_path = old_root / 'configs/rewards_stage1_gaps_v131_landing_stability.toml'
    require(old_reward_path.is_file() and not old_reward_path.is_symlink() and
            digest(old_reward_path) == OLD_REWARD_SHA,
            'frozen V132.2 control reward drift')
    old_reward = old_reward_path.read_bytes()
    receipt = old_root / 'reports/go1-v1322-real-entrance-s01-p1-d1_full_train.receipt'
    require(receipt.is_file() and not receipt.is_symlink() and
            digest(receipt) == spec['train_receipt_sha256'] == TRAIN_RECEIPT_SHA,
            'V132.2 failed teacher training receipt drift')
    record = json.loads(receipt.read_text(encoding='utf-8'))
    require(record.get('mode') == 'train' and record.get('status') == 'trained-unverified' and
            record.get('restored') is True and record.get('checkpoint_path') == str(model) and
            record.get('checkpoint_sha256') == MODEL_SHA and
            record.get('iterations') == 650 and record.get('actor_updates') == 600 and
            record.get('train_seed') == 42 and record.get('environments') == 512 and
            record.get('final_holdout') is False,
            'failed entrance teacher model lineage invalid')
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    for kind, sha in (('test_receipt', spec['test_receipt_sha256']),
                      ('smoke_receipt', spec['smoke_receipt_sha256'])):
        require(type(sha) is str and len(sha) == 64, 'pilot test/PhysX receipt SHA missing')
        path = root / 'reports' / names[kind]
        require(path.is_file() and not path.is_symlink() and digest(path) == sha,
                'pilot test/PhysX receipt provenance changed')
        evidence = json.loads(path.read_text(encoding='utf-8'))
        require(evidence.get('status') == 'passed' and evidence.get('restored') is True and
                evidence.get('manifest_sha256') == BASE_SHA and
                evidence.get('reward_sha256') == manifest['reward_config']['sha256'],
                'pilot test/PhysX source identity changed')
    plan = build_plan(root, config, spec)
    treatment = (root / manifest['reward_config']['path']).read_bytes()
    control = control_toml(old_reward)
    validate_against_v1322(control, old_reward)
    validate_tomls(control, treatment, spec, manifest)
    paths = artifacts(workspace, root, spec)
    validate_outputs(paths)
    return paths, {'plan': plan, 'control_toml': control,
                   'treatment_toml': treatment, 'checkpoint': model,
                   'test_receipt_sha256': spec['test_receipt_sha256'],
                   'smoke_receipt_sha256': spec['smoke_receipt_sha256'],
                   'train_receipt_sha256': TRAIN_RECEIPT_SHA}


def train_command(workspace: Path, config: dict, spec: dict, arm: str, plan: Path,
                  checkpoint: Path) -> list[str]:
    require(arm in ARMS and spec['iterations'] == 200 and spec['critic_warmup_iterations'] == 40,
            'unapproved pilot iteration contract')
    return [str(workspace / 'sim.sh'), 'train', '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--device', 'cuda:0', '--num-envs', str(spec['num_envs']),
            '--iterations', str(spec['iterations']), '--critic-warmup-iterations',
            str(spec['critic_warmup_iterations']), '--seed', str(spec['train_seed']),
            '--episode-seconds', str(config['episode_seconds']),
            '--course-difficulty-min', '0.0', '--course-difficulty-max', str(config['difficulty_max']),
            '--speed-min', str(config['speed_min']), '--speed-max', str(config['speed_max']),
            '--initialize-from', str(checkpoint), '--run-name', spec['run_ids'][arm]]


def eval_command(workspace: Path, config: dict, spec: dict, arm: str, checkpoint: Path,
                 trace: Path) -> list[str]:
    require(arm in ARMS and spec['development_evaluation_seed'] == 2028 and
            spec['num_envs'] == 64 and
            spec['evaluation_episode_quota'] == EVALUATION_EPISODE_QUOTA and
            float(config['episode_seconds']) == 200.0,
            'pilot evaluation coverage or episode-time contract changed')
    # Legacy evaluation uses episodes*40 as a step cap. The quota of 256
    # allows >=10240 steps for every 200s (10000-step) first episode to end;
    # it is NOT the number of environments scored, which remains exactly 64.
    return [str(workspace / 'sim.sh'), 'evaluate', '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--device', 'cuda:0', '--num-envs', '64', '--episodes',
            str(spec['evaluation_episode_quota']),
            '--episode-trace', str(trace), '--episode-seconds', str(config['episode_seconds']),
            '--course-difficulty-min', '0.0', '--course-difficulty-max', str(config['difficulty_max']),
            '--speed-min', str(config['gate_speed_min']), '--speed-max', str(config['gate_speed_max']),
            '--seed', str(spec['development_evaluation_seed']), '--checkpoint', str(checkpoint),
            '--run-name', f"{spec['run_ids'][arm]}-dev-seed2028"]


def latest_checkpoint(checkpoint_dir: Path, iterations: int = 200) -> Path:
    # model_0 from a periodic save is not evidence of completing all 200 updates.
    require(iterations == 200, 'pilot training iteration contract changed')
    final = checkpoint_dir / f'model_{iterations - 1}.pt'
    require(final.is_file() and not final.is_symlink(),
            'pilot arm lacks the final model_199 checkpoint; do not evaluate model_0')
    return final


def summarize(report: dict, trace: list[dict], arm: str, count: int) -> dict:
    """First per-env outcomes only; legacy 64-env aggregate can contain repeats."""
    require(arm in ARMS and count == 64 and isinstance(trace, list), 'pilot evaluation shape invalid')
    expected = {str(index) for index in range(count)}
    environments = report.get('environments', {})
    require(isinstance(environments, dict) and set(environments) == expected and
            type(report.get('episodes')) is int and
            report['episodes'] == EVALUATION_EPISODE_QUOTA and
            all(isinstance(entry, dict) and type(entry.get('episodes')) is int and
                entry['episodes'] >= 1 for entry in environments.values()) and
            sum(entry['episodes'] for entry in environments.values()) == len(trace),
            'pilot report/trace episode mismatch')
    first = {}
    totals = {str(index): 0 for index in range(count)}
    for row in trace:
        env_id = row.get('env_id')
        require(type(env_id) is int and str(env_id) in expected and
                type(row.get('episode_index')) is int and row['episode_index'] == totals[str(env_id)],
                'pilot trace ID/index mismatch')
        totals[str(env_id)] += 1
        first.setdefault(str(env_id), row)
    require(set(first) == expected and
            all(environments[key]['episodes'] == value for key, value in totals.items()),
            'pilot first-episode coverage incomplete')
    totals_flags = {key: 0 for key in ('success', 'fallen', 'collision', 'out_of_bounds', 'fell_into_gap')}
    support, safe, credited = 0, 0, 0
    failure_x = []
    completed = 0
    for row in first.values():
        require(all(type(row.get(key)) is bool for key in totals_flags) and
                all(type(row.get(key)) is int and row[key] >= 0 for key in
                    ('required_gap_count', 'credited_gap_count')) and
                2 <= row['required_gap_count'] <= 5 and
                row['credited_gap_count'] <= row['required_gap_count'] and
                isinstance(row.get('course_x'), (int, float)) and
                math.isfinite(row['course_x']), 'pilot terminal telemetry missing or invalid')
        events = row.get('reward_events')
        require(isinstance(events, dict) and all(type(events.get(k)) is int and events[k] >= 0
                for k in ('supported_short', 'safe_segment')), 'pilot reward activation missing')
        support += events['supported_short']
        safe += events['safe_segment']
        credited += row['credited_gap_count']
        require(events['safe_segment'] <= row['credited_gap_count'],
                'pilot safe_segment exceeds real credited gaps')
        for key in totals_flags:
            totals_flags[key] += int(row[key])
        clean = not any(row[key] for key in ('fallen', 'collision', 'out_of_bounds', 'fell_into_gap'))
        complete = row['success'] and clean and row['credited_gap_count'] == row['required_gap_count']
        require(not row['success'] or complete, 'pilot success violates hard criteria')
        completed += int(complete)
        if not complete:
            failure_x.append(row['course_x'])
    return {'first_episode_count': count, 'reported_completed_episodes': len(trace),
            'first_episode_completion_rate': completed / count,
            'supported_short_events': support, 'safe_segment_events': safe,
            'credited_gap_count': credited, 'mean_failure_x':
                sum(failure_x) / len(failure_x) if failure_x else None,
            **{key + '_rate': value / count for key, value in totals_flags.items()}}
