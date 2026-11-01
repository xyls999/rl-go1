"""Pinned development replay of the failed V132.1 entrance checkpoint, never a holdout."""
from __future__ import annotations

import json
import math
from pathlib import Path

from preflight_v1322 import digest, require

FROZEN_MANIFEST_SHA256 = '32ba96bf58efa9fc79f78febfd186c35062fbf1b89e9787209909f1e87f3da8f'
FAILED_CHECKPOINT_SHA256 = '1740e37faec1d9f1f50020faf56690f30ef36712899e532c3db6ec4eecf95e2f'
DEV_SEED = 2028  # Already used by V132.1: categorically not a final holdout.
OLD_GATE_PLAN = 'reports/lp_gate_plan_go1-v1321-real-entrance-s01-p1-d1.json'


def artifacts(workspace: Path, manifest: dict) -> dict:
    provenance = manifest['provenance']['v1321_failed_teacher_checkpoint']
    require(provenance['sha256'] == FAILED_CHECKPOINT_SHA256 and
            provenance['path'] == 'checkpoints/stage1/go1-v1321-real-entrance-s01-p1-d1/model_649.pt',
            'failed checkpoint provenance drift')
    run_id = f"{manifest['execution']['run_id']}-dev-v1321-seed2028-a01"
    require(run_id.startswith('go1-v1322-') and '/' not in run_id and '..' not in run_id,
            'unsafe development run ID')
    return {'run_id': run_id, 'checkpoint': workspace / provenance['path'],
            'plan': workspace / OLD_GATE_PLAN,
            'report': workspace / 'reports' / f'evaluate_stage1_gate-{run_id}.json',
            'trace': workspace / 'reports' / f'v1322_{run_id}_trace.jsonl',
            'log': workspace / 'v128_work/v132_randomized/v1322/reports' / f'v1322_{run_id}.log',
            'receipt': workspace / 'v128_work/v132_randomized/v1322/reports' / f'v1322_{run_id}.receipt'}


def validate_outputs(paths: dict) -> None:
    for key in ('report', 'trace', 'log', 'receipt'):
        target = paths[key]
        require(not target.exists() and not target.is_symlink(), f'development {key} already exists: {target}')


def validate_plan(paths: dict, config: dict) -> str:
    plan_path = paths['plan']
    require(plan_path.is_file() and not plan_path.is_symlink(), 'V132.1 entrance gate plan missing')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    count = config['num_envs']
    families, difficulties = plan.get('families'), plan.get('difficulty')
    require(isinstance(families, list) and families == [config['gate_family']] * count and
            isinstance(difficulties, list) and len(difficulties) == count and
            all(type(value) in (int, float) and math.isfinite(value) and
                0 <= value <= config['difficulty_max'] for value in difficulties),
            'V132.1 gate plan does not cover 512 real entrance tiles')
    require(plan.get('speed_min') == config['speed_min'] and
            plan.get('speed_max') == config['speed_max'], 'V132.1 gate speed band differs')
    instances = plan.get('instances')
    if instances is not None:
        require(len(instances) == count and
                all(isinstance(row, list) and len(row) == 3 and row[0] == config['gate_family']
                    for row in instances), 'V132.1 gate instances differ from entrance family')
    return digest(plan_path)


def validate_receipts(root: Path, workspace: Path, manifest: dict) -> dict[str, str]:
    """A03 receipts and their evidence bind to the unchanged base source manifest."""
    from production_v1322 import FAMILIES, attempt_artifacts

    manifest_path = root / 'configs/manifest.json'
    require(digest(manifest_path) == FROZEN_MANIFEST_SHA256,
            'frozen a03 source manifest changed; rerun tests and PhysX before diagnosis')
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    require(names['test_receipt'] == 'v1322_test_a03.receipt' and
            names['smoke_receipt'] == 'v1322_physx_a03.receipt', 'a03 receipt lineage changed')
    receipt_paths = {name: root / 'reports' / names[name] for name in ('test_receipt', 'smoke_receipt')}
    for name, path in receipt_paths.items():
        require(path.is_file() and not path.is_symlink(),
                ('test receipt' if name == 'test_receipt' else 'PhysX receipt') + ' missing')
    test = json.loads(receipt_paths['test_receipt'].read_text(encoding='utf-8'))
    smoke = json.loads(receipt_paths['smoke_receipt'].read_text(encoding='utf-8'))
    common = {'status': 'passed', 'restored': True,
              'manifest_sha256': FROZEN_MANIFEST_SHA256,
              'reward_sha256': manifest['reward_config']['sha256']}
    require(all(test.get(key) == value for key, value in common.items()) and
            test.get('mode') == 'test' and
            test.get('log_sha256') == digest(root / 'reports' / names['test_log']) and
            test.get('focus_sha256') == digest(root / 'reports' / names['focus_log']),
            'test receipt or its logs invalid')
    require(all(smoke.get(key) == value for key, value in common.items()) and
            smoke.get('mode') == 'smoke' and
            set(smoke.get('families', {})) == {name for name, _ in FAMILIES},
            'PhysX receipt missing families or source identity')
    for family, _ in FAMILIES:
        row = smoke['families'][family]
        log = root / 'reports' / f"{names['smoke_prefix']}{family}.log"
        report = workspace / 'reports' / (f"check_v1322-{manifest['execution']['run_id']}-"
                                          f"a{manifest['execution']['test_attempt']:02d}-{family}.json")
        require(log.is_file() and report.is_file() and
                row.get('log_sha256') == digest(log) and
                row.get('report_sha256') == digest(report),
                'PhysX evidence mismatch: ' + family)
    return {name + '_sha256': digest(path) for name, path in receipt_paths.items()}


def validate_diagnosis_inputs(root: Path, workspace: Path, manifest: dict, config: dict) -> tuple[dict, dict]:
    require(config['gate_holdout_seed'] is None and manifest['execution']['holdout_seed'] is None,
            'development seed must not be configured as final holdout')
    paths = artifacts(workspace, manifest)
    validate_outputs(paths)
    checkpoint = paths['checkpoint']
    require(checkpoint.is_file() and not checkpoint.is_symlink() and
            digest(checkpoint) == FAILED_CHECKPOINT_SHA256, 'V132.1 failed checkpoint SHA mismatch')
    plan_sha256 = validate_plan(paths, config)
    receipts = validate_receipts(root, workspace, manifest)
    return paths, {'plan_sha256': plan_sha256, **receipts}


def command(workspace: Path, config: dict, paths: dict) -> list[str]:
    return [str(workspace / 'sim.sh'), 'evaluate', '--stage', 'stage1', '--terrain', 'lp_acrl',
            '--headless', '--num-envs', '512', '--episodes', '512', '--strict-first-episode',
            '--episode-trace', str(paths['trace']), '--episode-seconds', str(config['episode_seconds']),
            '--speed-min', str(config['gate_speed_min']), '--speed-max', str(config['gate_speed_max']),
            '--seed', str(DEV_SEED), '--checkpoint', str(paths['checkpoint']),
            '--run-name', f"gate-{paths['run_id']}"]
