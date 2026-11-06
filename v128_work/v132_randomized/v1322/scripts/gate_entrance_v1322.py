"""One holdout, one first episode per entrance slot; no next-teacher dispatch."""
from __future__ import annotations

import json
import math
from pathlib import Path

from diagnostic_v1322 import validate_plan
from preflight_v1322 import digest, require

MANIFEST_SHA = '32ba96bf58efa9fc79f78febfd186c35062fbf1b89e9787209909f1e87f3da8f'
TRAIN_RECEIPT_SHA = 'b728dee938262b83d9bcb17f8f1e327eaf3adea0eb0d1566fc6f064fa1c9b185'
CHECKPOINT_SHA = '1a120f37c23f621e9f08772b1f6fc8520fabc4fb642b1da6342e08a96c7793f9'
PLAN_SHA = 'e16717f32563c9a1a9db72316a6eb85a51b557975080ae7c9066a71494f6e18b'
AUDIT_SHA = 'daf13784c1496c258fe7d3efac04bd1cd5d45fd3fc9b476d4e099c23a977621e'
SEED = 1812341965
RUN_ID = 'go1-v1322-real-entrance-s01-p1-d1-holdout-seed1812341965-a01'
TRAIN_ID = 'go1-v1322-real-entrance-s01-p1-d1'


def artifacts(workspace: Path, root: Path, spec: dict) -> dict[str, Path]:
    require(spec['gate_run_id'] == RUN_ID, 'unapproved holdout run ID')
    return {'checkpoint': workspace / 'checkpoints/stage1' / TRAIN_ID / 'model_649.pt',
            'plan': workspace / 'reports/lp_gate_plan_go1-v1321-real-entrance-s01-p1-d1.json',
            'report': workspace / 'reports' / f'evaluate_stage1_gate-{RUN_ID}.json',
            'trace': root / 'reports' / f'v1322_{RUN_ID}_trace.jsonl',
            'log': root / 'reports' / f'v1322_{RUN_ID}.log',
            'receipt': root / 'reports' / f'v1322_{RUN_ID}.receipt'}


def validate_outputs(paths: dict[str, Path]) -> None:
    for key in ('report', 'trace', 'log', 'receipt'):
        target = paths[key]
        require(not any(parent.is_symlink() for parent in target.parents) and
                not target.exists() and not target.is_symlink(),
                'holdout output already exists or has symlink parent: ' + key)


def _file(path: Path, expected: str) -> None:
    require(path.is_file() and not path.is_symlink() and digest(path) == expected,
            'holdout provenance file changed or missing: ' + str(path))


def validate_inputs(root: Path, workspace: Path, manifest: dict, config: dict,
                    spec: dict) -> tuple[dict[str, Path], dict]:
    require(digest(root / 'configs/manifest.json') == spec['frozen_manifest_sha256'] == MANIFEST_SHA,
            'frozen source manifest changed')
    require(spec['trained_run_id'] == TRAIN_ID and spec['gate_run_id'] == RUN_ID and
            spec['train_receipt_sha256'] == TRAIN_RECEIPT_SHA and
            spec['checkpoint_sha256'] == CHECKPOINT_SHA and
            spec['source_plan_sha256'] == PLAN_SHA and
            spec['seed_audit_sha256'] == AUDIT_SHA and
            type(spec['holdout_seed']) is int and spec['holdout_seed'] == SEED and
            SEED not in (42, 2026, 2028) and
            spec['num_envs'] == config['num_envs'] == spec['episodes'] == 512 and
            spec['minimum_successes'] == math.ceil(512 * .9) == 461 and
            spec['strict_first_episode'] is True and
            spec['automatic_junction_authorized'] is False and
            config['gate_holdout_seed'] is None and manifest['execution']['holdout_seed'] is None and
            manifest['execution']['train_and_gate_env'].get('GO1_PRIVILEGED_ACTOR') == '1',
            'unapproved seed, gate contract or actor observations')
    audit = root / 'provenance/holdout_seed_1812341965_audit.txt'
    _file(audit, AUDIT_SHA)
    text = audit.read_text(encoding='utf-8')
    require('SEED 1812341965 FILES_SCANNED 4030 BYTES_SCANNED 191124552 HITS []' in text and
            'HOLDOUT_SEED_UNUSED_IN_AUDITED_TEXT' in text and
            '2026-11-02T09:33:54.409407+00:00' in text,
            'pre-gate seed audit missing or inconclusive')
    paths = artifacts(workspace, root, spec)
    validate_outputs(paths)
    require(validate_plan(paths, config) == PLAN_SHA, '512-slot original plan changed')
    train_receipt = root / 'reports' / f'{TRAIN_ID}_full_train.receipt'
    _file(train_receipt, TRAIN_RECEIPT_SHA)
    record = json.loads(train_receipt.read_text(encoding='utf-8'))
    require(record.get('status') == 'trained-unverified' and record.get('mode') == 'train' and
            record.get('restored') is True and record.get('manifest_sha256') == MANIFEST_SHA and
            record.get('run_id') == TRAIN_ID and
            record.get('checkpoint_path') == str(paths['checkpoint']) and
            record.get('checkpoint_sha256') == CHECKPOINT_SHA and
            record.get('source_plan_sha256') == PLAN_SHA and
            record.get('iterations') == 650 and record.get('critic_warmup_iterations') == 50 and
            record.get('actor_updates') == 600 and record.get('environments') == 512 and
            record.get('train_seed') == 42 and record.get('final_holdout') is False and
            record.get('final_gate_passed') is False and
            record.get('automatic_gate_authorized') is False and
            record.get('automatic_junction_authorized') is False and
            all(record.get('reward_maxima', {}).get(name, 0) > 0 for name in
                ('supported_short', 'safe_segment')),
            'completed entrance training provenance mismatch')
    _file(paths['checkpoint'], CHECKPOINT_SHA)
    _file(workspace / 'reports/lp_plan_go1-v1321-real-entrance-s01-p1-d1.json', PLAN_SHA)
    _file(root / 'reports' / f'{TRAIN_ID}_full_train.log', record['training_log_sha256'])
    stage_path = workspace / 'reports' / f'stage_{TRAIN_ID}.json'
    _file(stage_path, record['stage_report_sha256'])
    stage = json.loads(stage_path.read_text(encoding='utf-8'))
    require(stage.get('run') == TRAIN_ID and stage.get('iterations') == 650 and
            stage.get('task_plan') == str(workspace / 'reports/lp_plan_go1-v1321-real-entrance-s01-p1-d1.json') and
            isinstance(stage.get('environments'), dict) and len(stage['environments']) == 512,
            'training stage provenance changed')
    return paths, {'train_receipt_sha256': TRAIN_RECEIPT_SHA, 'seed_audit_sha256': AUDIT_SHA}


def command(workspace: Path, config: dict, paths: dict[str, Path], spec: dict) -> list[str]:
    require(spec['holdout_seed'] == SEED and spec['gate_run_id'] == RUN_ID and
            spec['episodes'] == spec['num_envs'] == 512 and spec['strict_first_episode'] is True,
            'holdout command contract changed')
    return [str(workspace / 'sim.sh'), 'evaluate', '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--num-envs', '512', '--episodes', '512', '--strict-first-episode',
            '--episode-trace', str(paths['trace']), '--episode-seconds', str(config['episode_seconds']),
            '--speed-min', str(config['gate_speed_min']), '--speed-max', str(config['gate_speed_max']),
            '--seed', str(SEED), '--checkpoint', str(paths['checkpoint']),
            '--run-name', f'gate-{RUN_ID}']


def score(report: dict, trace: list[dict], spec: dict) -> dict:
    """Only validated, real-gap-complete first slots count; failure is measured, not an exception."""
    require(report['episodes'] == spec['episodes'] == 512 and
            len(report['environments']) == len(trace) == 512 and
            sum(slot['episodes'] for slot in report['environments'].values()) == 512,
            'holdout episode count differs from exactly 512 first slots')
    successes = 0
    events = {'supported_short': 0, 'safe_segment': 0}
    for row in trace:
        required, credited = row['required_gap_count'], row['credited_gap_count']
        reward = row.get('reward_events')
        require(isinstance(reward, dict) and all(type(reward.get(key)) is int and reward[key] >= 0
                                                 for key in events), 'holdout reward events missing')
        for key in events:
            events[key] += reward[key]
        require(reward['safe_segment'] <= credited,
                'safe_segment paid without credited real gap')
        hard = any(row[key] for key in ('fallen', 'collision', 'out_of_bounds', 'fell_into_gap'))
        require(not row['success'] or (not hard and credited == required and
                                     reward['safe_segment'] == required),
                'successful slot lacks hard safety or all real gaps')
        successes += int(row['success'] and not hard and credited == required)
    require(math.isclose(report['completion_rate'], successes / 512, abs_tol=1e-9),
            'holdout completion rate differs from verified numerator')
    return {'first_episode_successes': successes, 'first_episode_count': 512,
            'first_episode_completion_rate': successes / 512,
            'minimum_successes': 461, 'gate_passed': successes >= spec['minimum_successes'],
            'reward_events': events}


def validate_result(paths: dict[str, Path], spec: dict) -> dict:
    from run_teacher_v1322 import require_strict_gate_report
    from preflight_v1322 import validate_gate

    require(paths['report'].is_file() and not paths['report'].is_symlink() and
            paths['trace'].is_file() and not paths['trace'].is_symlink(),
            'strict holdout report or trace absent')
    report = require_strict_gate_report(paths['report'], paths['trace'], 512,
                                        paths['checkpoint'], SEED, RUN_ID)
    trace = [json.loads(line) for line in paths['trace'].read_text(encoding='utf-8').splitlines()]
    validate_gate(report, trace, 512, CHECKPOINT_SHA)
    return score(report, trace, spec)
