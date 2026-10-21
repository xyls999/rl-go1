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
 'scripts/run_student_v32.py': '3a30cada4349083b6ec7a70a15ce186325781bf3292a5ff2d1a325ec915b9243',
 'configs/curriculum_phases_long_v46.json': '0c269bc5be15239208aaa8bfab53ccd585d743fd518ee51d35a2b320a5c364d0',
 'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py': 'fd63e677448f09cdf929661d0b782a89b05ab929bdf7dacd45e7c834b78eda34',
 'source/go1_hier_sim/go1_hier_sim/envs/config.py': '23cc8d96efc687ad00a6b5224ae58547e9f90883cf702d44528dc3c2722e8ae9',
 'source/go1_hier_sim/go1_hier_sim/cli.py': 'e7136aa719e19821d153788d3c909747d55dcb97990205a5b2ec67a39232d1ed',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '0ea790c5ebff0cef3273955eccd63515d305b93ea18244c19e6c8dcc6755958a',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py': 'eb072e16e4da03276c98e0bd3cc6df55e890c0ccc334591d16f8f2693fc7bf25',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'f02efc7201113f9d89bef8a01e9cb1c25a077182d92f7dc54bb9ea71c322f2b2',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-19_go1-v38-narrow-credit-s01-p1-d1/model_799.pt': 'd6bcb443dab5bdeb3ff4b842489daa3b0bb4941c488f0f177ac2a60cebd755f1',
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
for path in ('lp_combined_v46_state.json', 'lp_combined_v46_acrl_state.json',
             'lp_combined_v46_results.txt', 'reports/v46_combined.exit',
             'reports/lp_plan_2026-10-19_go1-v46-long-continuation-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-19_go1-v46-long-continuation-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v46-long-continuation*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v46-long-continuation*'))
print('V32A_COMBINED_SOFT_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
unset GO1_V36_SOFT_TERMINATION
export GO1_V36_SOFT_TERMINATION=1 GO1_V8_LONG_GAP_THRESHOLD=0.05 GO1_V19_MIN_WIDTH=0.05
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND GO1_V29_TEACHER_DISTILL GO1_V29_TEACHERS GO1_V29_EPS GO1_V29_LAMBDA GO1_V23_TIRE_BRAKE GO1_V23_TIRE_BRAKE_FAMILIES GO1_V23_TIRE_CAP GO1_V23_TIRE_PEN
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V32A_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=single:combined GATE=combined INIT=v38\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v32.py \
  --base-id 2026-10-19_go1-v46-long-continuation \
  --config configs/curriculum_phases_long_v46.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v46_combined.exit
exit "$status"
