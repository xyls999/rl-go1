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
 'scripts/run_student_v27.py': 'e4eeb9a8fa1be55340f8d7e041aa7abdec63f3c6bc6d1d2e5c2efac57689e08b',
 'configs/curriculum_phases_eleven_mixture_v28.json': '30a4404a086e65744fb7e94091ecdf625b68b9acaee8952c0724c8e6bee92815',
 'source/go1_hier_sim/go1_hier_sim/envs/eleven_roster_v27.py': 'da61d49c23abf2080ec8c6b453eeb582a229a86d2553d72857c13de8e14530cc',
 'source/go1_hier_sim/go1_hier_sim/cli.py': '4f9f4af26b20ee3d838333b1b1dd55368bcbd21eda9a4b66be3aa55e9f4a135d',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '50ed5a7c31f0b20da0d61b961cae7303b1e2b6052e2a65b3db8d4082c198ff1b',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-18_go1-v27-eleven-mixture-s01-p1-d1/model_899.pt': '7b5f7bf450017e81cd9e520f8d8bef1fd58c3ba0eec19ab7bd44a6a990093c43',
 'reports/v27_combined_trace512.jsonl': 'baffeec6e85fecfd2d192ff370af0d1ef86fabfb9df834549ae3bdba64338d3f',
}
for path, digest in expected.items():
    assert sha(path) == digest, f'drift: {path}'
assert (root / 'reports/v27_eleven_mixture.exit').read_text().startswith('exit=0 ')
assert (root / 'reports/v28_smoke4x16.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_eleven_mixture_v28_state.json', 'lp_eleven_mixture_v28_acrl_state.json',
             'lp_eleven_mixture_v28_results.txt', 'reports/v28_eleven_mixture.exit',
             'reports/lp_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v28-eleven-mixture*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v28-eleven-mixture*'))
print('V28_ELEVEN_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_BRAKE_FAMILIES=tire_field,combined GO1_V23_TIRE_CAP=.60 GO1_V23_TIRE_PEN=.5 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V28_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=eleven_with_combined INIT=v27 BRAKE=tire+combined\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v27.py \
  --base-id 2026-10-18_go1-v28-eleven-mixture \
  --config configs/curriculum_phases_eleven_mixture_v28.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v28_eleven_mixture.exit
exit "$status"