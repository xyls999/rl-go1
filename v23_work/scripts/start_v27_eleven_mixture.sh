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
 'configs/curriculum_phases_eleven_mixture_v27.json': '8dacf14b1994fd1af6b86a4c940af8c2e3b19d035569375269e33abbf1ee7d2d',
 'source/go1_hier_sim/go1_hier_sim/envs/eleven_roster_v27.py': 'da61d49c23abf2080ec8c6b453eeb582a229a86d2553d72857c13de8e14530cc',
 'source/go1_hier_sim/go1_hier_sim/cli.py': '4f9f4af26b20ee3d838333b1b1dd55368bcbd21eda9a4b66be3aa55e9f4a135d',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': 'c93ab53b0f8ac6c49358e80657c9a21f00c82b8ab397fe333e89a68d79efac7e',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '3e6a2b73f91bc92b4731a58b7a32db6b5a5001b78989c68e0f9aea893c54db0f',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt': '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
 'reports/v26_final11_trace512.jsonl': '5da04ecda118796872f40d91e116f5b24499d05685af54355e51e8eea9c8b7dc',
 'reports/v26_ten_gate_trace512.jsonl': '71f1b5dd057a174efd4674281146f8d00cfe3b7eb4ce3f8e7acd3f6a28f3638c',
}
for path, digest in expected.items():
    assert sha(path) == digest, f'drift: {path}'
assert (root / 'reports/v26_triple_refine.exit').read_text().startswith('exit=0 ')
assert (root / 'reports/v27_smoke4x16.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_eleven_mixture_v27_state.json', 'lp_eleven_mixture_v27_acrl_state.json',
             'lp_eleven_mixture_v27_results.txt', 'reports/v27_eleven_mixture.exit',
             'reports/lp_plan_2026-10-18_go1-v27-eleven-mixture-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-18_go1-v27-eleven-mixture-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v27-eleven-mixture*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v27-eleven-mixture*'))
print('V27_ELEVEN_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_CAP=.60 GO1_V23_TIRE_PEN=.5 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V27_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=eleven_with_combined INIT=v26 BRAKE=cap.60/pen.50\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v27.py \
  --base-id 2026-10-18_go1-v27-eleven-mixture \
  --config configs/curriculum_phases_eleven_mixture_v27.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v27_eleven_mixture.exit
exit "$status"
