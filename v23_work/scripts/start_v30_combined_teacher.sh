#!/usr/bin/env bash
set -euo pipefail
cd /config/Desktop/sim
/root/env_isaaclab/bin/python - <<PY2
import glob, hashlib, subprocess, json
from pathlib import Path
import torch
root = Path('/config/Desktop/sim')
sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
expected = {
 'scripts/run_student_v30.py': 'ec2620483323f80ca494c60fbfc89dcac3fb1e236dece90598c8e80d74031d06',
 'configs/curriculum_phases_combined_v30.json': 'c0df50ac6aa8d10366535cf9603133e5dd6d4248b160daa45d8f74f3c55f5962',
 'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v30.py': '3d28aec8f87e6fd8b84f587cd43fd12c3e22b9a5c74cee969fb6f2545037114a',
 'source/go1_hier_sim/go1_hier_sim/cli.py': 'b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '08d609da744a769c93f801d2e9f546ffb9d7da2164611d3ad818599cbf6b6a9f',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt': '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
}
for path, digest in expected.items():
    assert sha(path) == digest, f'drift: {path}'
assert (root / 'reports/v29b_student_distill.exit').read_text().startswith('exit=0 ')
assert (root / 'reports/v30_smoke4x16.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_combined_v30_state.json', 'lp_combined_v30_acrl_state.json',
             'lp_combined_v30_results.txt', 'reports/v30_combined_teacher.exit',
             'reports/lp_plan_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v30-combined-teacher*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v30-combined-teacher*'))
print('V30_COMBINED_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_BRAKE_FAMILIES=combined GO1_V23_TIRE_CAP=.60 GO1_V23_TIRE_PEN=.5 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND GO1_V29_TEACHER_DISTILL GO1_V29_TEACHERS GO1_V29_EPS GO1_V29_LAMBDA
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V30_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=single:combined INIT=v26 BRAKE=combined\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v30.py \
  --base-id 2026-10-19_go1-v30-combined-teacher \
  --config configs/curriculum_phases_combined_v30.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v30_combined_teacher.exit
exit "$status"