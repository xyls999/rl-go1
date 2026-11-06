"""Pinned, manually authorized full entrance teacher train; never a gate.

The pilot is only evidence to attempt training. It does not certify a teacher.
No final holdout seed or evaluation is consumed by this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from preflight_v1322 import digest, require
from diagnostic_v1322 import validate_plan, validate_receipts
from pilot_v1322 import ARMS, summarize

BASE_SHA = '32ba96bf58efa9fc79f78febfd186c35062fbf1b89e9787209909f1e87f3da8f'
PILOT_SHA = 'e861917d1f4784334cd628f957f39cfd6853edefb74688742ab9d63d4ac72912'
PILOT_SPEC_SHA = '4ac57ffe273465c135b2e5abfde534330fa11c11cb3c1d0c7d326f912606dfcb'
SOURCE_PLAN_SHA = 'e16717f32563c9a1a9db72316a6eb85a51b557975080ae7c9066a71494f6e18b'
MODEL_SHA = 'd007b70b690b28da1d23c86daabaa221fb28d72f85e49259652ebdded0ab6450'
LAUNCHER_SHA = '9a1f496c6f1b575f2150aaa28c5ebca42b53dc3e6bf3d1080a297ad471d88a55'
PILOT_ID = 'go1-v1322-supportedshort-pilot-a02'


def artifacts(workspace: Path, root: Path, spec: dict) -> dict[str, Path]:
    run_id = spec['run_id']
    require(run_id == 'go1-v1322-real-entrance-s01-p1-d1', 'unapproved full train ID')
    return {
        'checkpoint_dir': workspace / 'checkpoints/stage1' / run_id,
        'runs_dir': workspace / 'runs/stage1' / run_id,
        'stage_report': workspace / 'reports' / f'stage_{run_id}.json',
        'log': root / 'reports' / f'{run_id}_full_train.log',
        'receipt': root / 'reports' / f'{run_id}_full_train.receipt',
    }


def validate_outputs(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        require(not any(parent.is_symlink() for parent in path.parents),
                'symlinked training output parent: ' + str(path))
        require(not path.exists() and not path.is_symlink(),
                'training output already exists: ' + key)


def require_pilot_improvement(record: dict) -> None:
    require(record.get('status') == 'pilot-diagnostic-only' and record.get('mode') == 'pilot'
            and record.get('restored') is True and record.get('final_holdout') is False
            and record.get('full_training_started') is False
            and record.get('automatic_full_training_authorized') is False,
            'pilot cannot authorize a final gate or prove teacher success')
    arms = record.get('arms', {})
    require(set(arms) == set(ARMS), 'paired pilot evidence missing an arm')
    control = arms['control']['evaluation_first_episode']
    treatment = arms['treatment']['evaluation_first_episode']
    require(control['first_episode_count'] == treatment['first_episode_count'] == 64,
            'pilot did not cover 64 paired first episodes')
    delta = treatment['first_episode_completion_rate'] - control['first_episode_completion_rate']
    require(delta > 0 and math.isclose(record['pilot_completion_delta'], delta, abs_tol=1e-12),
            'pilot first-slot completion did not improve')
    require(record['safety_regression_observed'] is False and
            all(treatment[key] <= control[key] for key in
                ('fallen_rate', 'collision_rate', 'fell_into_gap_rate', 'out_of_bounds_rate')),
            'pilot hard-failure category worsened')
    require(treatment['supported_short_events'] > 0 and treatment['safe_segment_events'] > 0
            and treatment['safe_segment_events'] == treatment['credited_gap_count'],
            'treatment reward events do not match real gap credit')


def _verified_file(path: Path, sha: str) -> None:
    require(path.is_file() and not path.is_symlink() and digest(path) == sha,
            'pinned training provenance absent or changed: ' + str(path))


def verify_pilot_artifacts(root: Path, workspace: Path, record: dict, pilot_spec: dict) -> None:
    require(record.get('pilot_spec_sha256') == PILOT_SPEC_SHA and
            record.get('manifest_sha256') == BASE_SHA and
            record.get('initial_checkpoint_sha256') == MODEL_SHA and
            record.get('final_holdout') is False, 'pilot root provenance changed')
    require(record['plan_sha256'] == pilot_spec['plan_sha256'] and
            record['control_toml_sha256'] == pilot_spec['control_toml_sha256'] and
            record['treatment_toml_sha256'] == pilot_spec['treatment_toml_sha256'],
            'pilot geometry/reward comparison provenance changed')
    _verified_file(root / 'reports' / f'{PILOT_ID}_plan.json', record['plan_sha256'])
    _verified_file(root / 'reports' / f'{PILOT_ID}_control.toml', record['control_toml_sha256'])
    for arm in ARMS:
        entry = record['arms'][arm]
        run_id = pilot_spec['run_ids'][arm]
        require(entry['run_id'] == run_id and
                entry['supported_short_weight'] == (0.0 if arm == 'control' else 0.8) and
                entry['reward_toml_sha256'] == record[f'{arm}_toml_sha256'],
                'pilot arm reward or run identity changed')
        paths = {
            'stage_report_sha256': workspace / 'reports' / f'stage_{run_id}.json',
            'training_log_sha256': root / 'reports' / f'{run_id}_train.log',
            'evaluation_log_sha256': root / 'reports' / f'{run_id}_eval.log',
            'evaluation_report_sha256': workspace / 'reports' /
                f'evaluate_stage1_{run_id}-dev-seed2028.json',
            'evaluation_trace_sha256': root / 'reports' / f'{run_id}_dev_trace.jsonl',
        }
        for key, path in paths.items():
            _verified_file(path, entry[key])
        checkpoint = workspace / 'checkpoints/stage1' / run_id / 'model_29.pt'
        require(entry['checkpoint_path'] == str(checkpoint), 'pilot checkpoint path mismatch')
        _verified_file(checkpoint, entry['checkpoint_sha256'])
        stage = json.loads(paths['stage_report_sha256'].read_text(encoding='utf-8'))
        require(stage.get('run') == run_id and stage.get('iterations') == 30 and
                stage.get('task_plan') == str(root / 'reports' / f'{PILOT_ID}_plan.json') and
                entry.get('training_episodes') == stage.get('episodes'),
                'pilot arm training metadata invalid')
        report = json.loads(paths['evaluation_report_sha256'].read_text(encoding='utf-8'))
        require(report.get('checkpoint_sha256') == entry['checkpoint_sha256'] and
                report.get('seed') == 2028, 'pilot dev evaluation provenance invalid')
        trace = [json.loads(line) for line in
                 paths['evaluation_trace_sha256'].read_text(encoding='utf-8').splitlines()]
        require(summarize(report, trace, arm, 64) == entry['evaluation_first_episode'],
                'pilot first-slot results differ from actual report and trace')


def validate_inputs(root: Path, workspace: Path, manifest: dict, config: dict,
                    spec: dict) -> tuple[dict[str, Path], dict]:
    require(digest(root / 'configs/manifest.json') == spec['frozen_manifest_sha256'] == BASE_SHA,
            'source manifest changed; rerun tests and PhysX before training')
    require(digest(root / 'configs/pilot_execution.json') == spec['pilot_spec_sha256'] == PILOT_SPEC_SHA,
            'pilot specification changed')
    require(spec['pilot_receipt_sha256'] == PILOT_SHA and
            spec['source_plan_sha256'] == SOURCE_PLAN_SHA and
            spec['initial_checkpoint_sha256'] == MODEL_SHA and
            spec['run_id'] == manifest['execution']['run_id'] and
            spec['num_envs'] == config['num_envs'] == 512 and
            spec['seed'] == manifest['execution']['train_seed'] == 42 and
            spec['iterations'] == config['phases'][0]['iterations'] == 650 and
            spec['critic_warmup_iterations'] == config['warmup_first'] == 50 and
            spec['final_holdout'] is False and spec['automatic_gate_authorized'] is False and
            spec['automatic_junction_authorized'] is False and
            config['gate_holdout_seed'] is None and manifest['execution']['holdout_seed'] is None,
            'unapproved full train or final holdout configuration')
    require(config['init_checkpoint'] == manifest['execution']['initial_checkpoint'],
            'full train initial checkpoint differs from original V132.1 config')
    source = workspace / config['init_checkpoint']
    _verified_file(source, MODEL_SHA)
    require(spec['original_torchrun_launcher_sha256'] == LAUNCHER_SHA,
            'original V132.1 torchrun launcher pin changed')
    _verified_file(workspace / 'scripts/torchrun_sim.sh', LAUNCHER_SHA)
    gate_plan = workspace / 'reports/lp_gate_plan_go1-v1321-real-entrance-s01-p1-d1.json'
    require(validate_plan({'plan': gate_plan}, config) == SOURCE_PLAN_SHA,
            'full train original gate-plan provenance changed')
    plan = workspace / 'reports/lp_plan_go1-v1321-real-entrance-s01-p1-d1.json'
    _verified_file(plan, SOURCE_PLAN_SHA)
    receipts = validate_receipts(root, workspace, manifest)
    diagnosis = root / 'reports/v1322_go1-v1322-real-entrance-s01-p1-d1-dev-v1321-seed2028-a01.receipt'
    _verified_file(diagnosis, 'eec7976bf2b42a73d1e7610385accd9835f292be171e40a6706da2058ac9d198')
    diagnostic_record = json.loads(diagnosis.read_text(encoding='utf-8'))
    require(diagnostic_record.get('status') == 'diagnostic' and
            diagnostic_record.get('mode') == 'diagnose' and
            diagnostic_record.get('restored') is True and
            diagnostic_record.get('final_holdout') is False and
            diagnostic_record.get('episodes') == 512 and diagnostic_record.get('seed') == 2028 and
            diagnostic_record.get('manifest_sha256') == BASE_SHA and
            diagnostic_record.get('checkpoint_sha256') ==
            manifest['provenance']['v1321_failed_teacher_checkpoint']['sha256'],
            'old-checkpoint development diagnosis missing')
    diagnostic_run = diagnostic_record['run_id']
    require(diagnostic_run == 'go1-v1322-real-entrance-s01-p1-d1-dev-v1321-seed2028-a01',
            'old-checkpoint development run identity changed')
    for name, path in (
        ('report_sha256', workspace / 'reports' / f'evaluate_stage1_gate-{diagnostic_run}.json'),
        ('trace_sha256', workspace / 'reports' / f'v1322_{diagnostic_run}_trace.jsonl'),
        ('log_sha256', root / 'reports' / f'v1322_{diagnostic_run}.log'),
    ):
        _verified_file(path, diagnostic_record[name])
    pilot_receipt = root / 'reports' / f'{PILOT_ID}.receipt'
    _verified_file(pilot_receipt, PILOT_SHA)
    pilot_spec = json.loads((root / 'configs/pilot_execution.json').read_text(encoding='utf-8'))
    record = json.loads(pilot_receipt.read_text(encoding='utf-8'))
    verify_pilot_artifacts(root, workspace, record, pilot_spec)
    require_pilot_improvement(record)
    paths = artifacts(workspace, root, spec)
    validate_outputs(paths)
    return paths, {'checkpoint': source, 'plan': plan, 'pilot_receipt_sha256': PILOT_SHA,
                   'diagnosis_receipt_sha256': digest(diagnosis), **receipts}


def command(workspace: Path, config: dict, spec: dict, checkpoint: Path) -> list[str]:
    require(spec['run_id'] == 'go1-v1322-real-entrance-s01-p1-d1' and
            spec['num_envs'] == 512 and spec['iterations'] == 650 and
            spec['critic_warmup_iterations'] == 50 and spec['seed'] == 42,
            'training command contract changed')
    return ['bash', str(workspace / 'scripts/torchrun_sim.sh'), 'train',
            '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--device', 'cuda:0', '--num-envs', '512', '--iterations', '650',
            '--critic-warmup-iterations', '50', '--seed', '42',
            '--episode-seconds', str(config['episode_seconds']),
            '--course-difficulty-min', '0.0', '--course-difficulty-max', str(config['difficulty_max']),
            '--speed-min', str(config['speed_min']), '--speed-max', str(config['speed_max']),
            '--initialize-from', str(checkpoint), '--run-name', spec['run_id']]


def finished_checkpoint(paths: dict[str, Path]) -> Path:
    checkpoint = paths['checkpoint_dir'] / 'model_649.pt'
    require(checkpoint.is_file() and not checkpoint.is_symlink(),
            'full training missing final model_649 checkpoint')
    return checkpoint


def reward_activation(log: Path) -> dict[str, float]:
    text = log.read_text(encoding='utf-8', errors='replace')
    maxima = {}
    for name in ('supported_short', 'safe_segment'):
        values = [float(value) for value in re.findall(
            rf'Episode_Reward/{name}:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', text)]
        require(values and max(values) > 0, name + ' positive training reward absent')
        maxima[name] = max(values)
    return maxima
