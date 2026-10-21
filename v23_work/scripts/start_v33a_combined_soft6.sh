#!/usr/bin/env bash
set -euo pipefail
cd /config/Desktop/sim
/root/env_isaaclab/bin/python - <<PY2
import glob, hashlib, subprocess
from pathlib import Path
import torch
root = Path('/config/Desktop/sim')
sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
expected = {
 'scripts/run_student_v32.py': 'd71d0be131861cb4160075413faa56fc5f8ef74e7319c07ca8527cc23555098b',
 'configs/curriculum_phases_combined_soft_v33a.json': '86934e2af9849d7c074dd4aa411e0f9ce55fd54ae6dc2b2bcfb0dcfc52fa61fd',
 'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py': 'fd63e677448f09cdf929661d0b782a89b05ab929bdf7dacd45e7c834b78eda34',
 'source/go1_hier_sim/go1_hier_sim/cli.py': 'e7136aa719e19821d153788d3c909747d55dcb97990205a5b2ec67a39232d1ed',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '89d54edf74f677a5a560841990ab9438657a7e759c5f11c0c6d4dca698dd5c9c',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py': '3f9cb342f53159cb5bdb1960afb0a7413bc3c19baebee7b71be9d00ffe376a15',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt': '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
}
for path, digest in expected.items():
    assert sha(path) == digest, f'drift: {path}'
assert (root / 'reports/v30b_combined_teacher.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_combined_v33a_state.json', 'lp_combined_v33a_acrl_state.json',
             'lp_combined_v33a_results.txt', 'reports/v33a_combined_soft6.exit',
             'reports/lp_plan_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v33a-combined-soft6*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v33a-combined-soft6*'))
print('V32A_COMBINED_SOFT_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND GO1_V29_TEACHER_DISTILL GO1_V29_TEACHERS GO1_V29_EPS GO1_V29_LAMBDA GO1_V23_TIRE_BRAKE GO1_V23_TIRE_BRAKE_FAMILIES GO1_V23_TIRE_CAP GO1_V23_TIRE_PEN
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V32A_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=single:combined_soft6 GATE=combined INIT=v26\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v32.py \
  --base-id 2026-10-19_go1-v33a-combined-soft6 \
  --config configs/curriculum_phases_combined_soft_v33a.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v33a_combined_soft6.exit
exit "$status"
