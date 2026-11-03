#!/usr/bin/env python3
"""Read-only V132.3 source, lock, GPU and output preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE_SHA = 'b77cb4099ca6a68d1ffd574f1b8dfe505b68de352cc9dc302772b33e1851bc9e'
OLD_REWARD_SHA = 'cfde98e63504329ee1e3d13e0f19adf9207700a8ef1288703f824190680358ca'
BASELINE_REWARD_CONFIG_SHA = '7b46c3a7eb3b423de82eecccdfe7c4750e50cfbaeb94fb1a14e149b6025a834e'
MODEL_SHA = '1a120f37c23f621e9f08772b1f6fc8520fabc4fb642b1da6342e08a96c7793f9'
PLAN_SHA = 'e16717f32563c9a1a9db72316a6eb85a51b557975080ae7c9066a71494f6e18b'
ALLOWED_MODES = ('test', 'smoke', 'pilot')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_candidate_sources(root: Path, manifest: dict) -> None:
    seen: set[str] = set()
    for row in manifest['sources']:
        name, candidate = row['name'], root / row['candidate']
        require(name not in seen and row['candidate'] == row['live'] and
                row['live'].startswith('source/go1_hier_sim/go1_hier_sim/') and
                '..' not in Path(row['live']).parts,
                'unsafe/duplicate candidate source')
        seen.add(name)
        require(candidate.is_file() and not candidate.is_symlink() and
                digest(candidate) == row['candidate_sha256'],
                'candidate source hash drift: ' + str(candidate))
    require(seen == {'cli.py', 'runtime.py', 'geometry.py', 'config.py', 'reward_config.py',
                     'strict_gate_v1322.py', 'entrance_support_v1322.py',
                     'edge_contact_v1323.py', 'gap_score_v1321.py',
                     'landing_stability_v131.py'}, 'candidate imports incomplete')


def validate_pinned_inputs(root: Path, workspace: Path, manifest: dict, config: dict) -> None:
    from pilot_v1323 import MODEL_SHA as PINNED_MODEL, SOURCE_PLAN_SHA
    require(digest(root / 'configs/manifest.json') == BASE_SHA, 'V132.3 manifest SHA drift')
    require(digest(root / 'configs/teacher_entrance.json') == manifest['configuration_sha256'],
            'pilot configuration SHA drift')
    require(digest(root / manifest['reward_config']['path']) == manifest['reward_config']['sha256'],
            'V132.3 reward SHA drift')
    old_reward = workspace / 'v128_work/v132_randomized/v1322/configs/rewards_stage1_gaps_v131_landing_stability.toml'
    require(old_reward.is_file() and not old_reward.is_symlink() and
            digest(old_reward) == OLD_REWARD_SHA, 'frozen V132.2 reward source drift')
    from pilot_v1323 import control_toml, validate_against_v1322, validate_tomls
    treatment = (root / manifest['reward_config']['path']).read_bytes()
    spec = json.loads((root / 'configs/pilot_execution.json').read_text(encoding='utf-8'))
    control = control_toml(treatment)
    validate_tomls(control, treatment, spec, manifest)
    validate_against_v1322(control, old_reward.read_bytes())
    require(digest(root / 'provenance/v1321_gate_plan.json') == SOURCE_PLAN_SHA == PLAN_SHA,
            'source 512-plan SHA drift')
    require(digest(root / 'provenance/reward_config.baseline.py') == BASELINE_REWARD_CONFIG_SHA,
            'live reward loader provenance changed')
    require(manifest['baseline']['reward_config.py'] == BASELINE_REWARD_CONFIG_SHA and
            manifest['baseline']['edge_contact_v1323.py'] is None,
            'remote reward loader/edge helper baseline changed')
    require(config['training_roster'] == 'single:combined_part_entrance' and
            config['gate_family'] == 'combined_part_entrance' and
            config['num_envs'] == 512 and config['episode_seconds'] == 200 and
            config['speed_min'] == config['gate_speed_min'] == .35 and
            config['speed_max'] == config['gate_speed_max'] == .8 and
            config['gate_holdout_seed'] is None and
            config['init_checkpoint'] == manifest['execution']['initial_checkpoint'] ==
            'checkpoints/stage1/go1-v1322-real-entrance-s01-p1-d1/model_649.pt' and
            manifest['execution']['checkpoint_sha256'] == MODEL_SHA == PINNED_MODEL and
            manifest['execution']['holdout_seed'] is None and
            manifest['execution']['test_attempt'] == 1,
            'unapproved checkpoint, speed or development-only pilot settings')
    require(manifest['execution']['train_and_gate_env']['GO1_PRIVILEGED_ACTOR'] == '1' and
            manifest['execution']['train_and_gate_env']['GO1_REWARD_CONFIG_PATH'] ==
            'v128_work/v132_randomized/v1323/configs/rewards_stage1_gaps_v131_landing_stability.toml',
            'observation/reward parity changed')
    for relative in ('provenance/v1321_teacher_entrance.json', 'provenance/v1321_manifest.json',
                     'provenance/remote_live_audit_v1322.txt'):
        original = workspace / 'v128_work/v132_randomized/v1322' / relative
        require(digest(root / relative) == digest(original), 'V132.2 provenance snapshot changed')
    validate_candidate_sources(root, manifest)


def gpu_usage_mib() -> list[int]:
    raw = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True,
                         text=True, check=True).stdout
    lines = [line.strip().split(',') for line in raw.splitlines() if line.strip()]
    require(len(lines) == 4, 'GPU count != 4')
    return [int(parts[1].strip()) for parts in lines]


def preflight(workspace: Path, root: Path, mode: str) -> None:
    from production_v1323 import (attempt_artifacts, check_targets, ensure_idle, targets)
    from pilot_v1323 import artifacts, validate_inputs, validate_outputs
    require(mode in ALLOWED_MODES and root == workspace / 'v128_work/v132_randomized/v1323',
            'unsupported remote mode/root')
    manifest = json.loads((root / 'configs/manifest.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'configs/teacher_entrance.json').read_text(encoding='utf-8'))
    validate_pinned_inputs(root, workspace, manifest, config)
    require(all(0 <= item <= 256 for item in gpu_usage_mib()), 'GPU busy')
    ensure_idle()
    check_targets(targets(root, workspace, manifest, mode), workspace)
    names = attempt_artifacts(manifest['execution']['test_attempt'])
    if mode == 'pilot':
        validate_inputs(root, workspace, manifest, config,
                        json.loads((root / 'configs/pilot_execution.json').read_text(encoding='utf-8')))
    else:
        outputs = ([names['test_log'], names['focus_log'], names['test_receipt']] if mode == 'test'
                   else [names['smoke_receipt']] +
                   [f"{names['smoke_prefix']}{family}.log" for family in ('entrance', 'junction', 'long', 'tire')] +
                   [f'v1323_smoke_plan_a01_{family}.json' for family in ('entrance', 'junction', 'long', 'tire')])
        for filename in outputs:
            p = root / 'reports' / filename
            require(not p.exists() and not p.is_symlink(), 'mode output already exists: ' + str(p))
        if mode == 'smoke':
            receipt = root / 'reports' / names['test_receipt']
            require(receipt.is_file() and not receipt.is_symlink(), 'full tests receipt absent')
            record = json.loads(receipt.read_text(encoding='utf-8'))
            require(record.get('status') == 'passed' and record.get('restored') is True and
                    record.get('manifest_sha256') == BASE_SHA and
                    record.get('reward_sha256') == manifest['reward_config']['sha256'] and
                    record.get('log_sha256') == digest(root / 'reports' / names['test_log']) and
                    record.get('focus_sha256') == digest(root / 'reports' / names['focus_log']),
                    'full tests evidence drift')
            for family in ('entrance', 'junction', 'long', 'tire'):
                target = workspace / 'reports' / f"check_v1323-{manifest['execution']['run_id']}-a01-{family}.json"
                require(not target.exists() and not target.is_symlink(),
                        'smoke report already exists: ' + str(target))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    workspace = Path('/config/Desktop/sim')
    try:
        arg = sys.argv[1]
        require(arg in ('preflight-test', 'preflight-smoke', 'preflight-pilot'),
                'only guarded test/smoke/pilot modes exist')
        import fcntl
        lock = workspace / 'reports/v7_sim.lock'
        require(lock.is_file() and not lock.is_symlink() and os.fstat(9).st_ino == lock.stat().st_ino,
                'exclusive simulator lock fd missing')
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight(workspace, root, arg.removeprefix('preflight-'))
    except (IndexError, KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'PREFLIGHT_BLOCKED: {error}', file=sys.stderr)
        return 1
    print(f'PREFLIGHT_PASS mode={arg.removeprefix("preflight-")} read_only=true final_gate_authorized=false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
