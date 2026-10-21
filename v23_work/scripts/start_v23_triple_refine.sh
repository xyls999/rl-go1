#!/usr/bin/env bash
set -euo pipefail
cd /config/Desktop/sim
/root/env_isaaclab/bin/python - <<'PY'
import glob, hashlib, subprocess
from pathlib import Path
import torch
root = Path('/config/Desktop/sim')
sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
expected = {
 'scripts/run_student_v23.py': 'c0220a2535ec0a36bf882d85a8919c763ad804bc4a0ffd02740a9c120669ce62',
 'configs/curriculum_phases_triple_refine_v23.json': '8e568b26377faae6007075a6e9c0d1632fd379d46e45d9d315582507f0de97c3',
 'source/go1_hier_sim/go1_hier_sim/cli.py': '4f9f4af26b20ee3d838333b1b1dd55368bcbd21eda9a4b66be3aa55e9f4a135d',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '8f3138d88e8e587621ee750d98a12ba1e01d30ba82a27dfb8f1cce96e9142c61',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '3e6a2b73f91bc92b4731a58b7a32db6b5a5001b78989c68e0f9aea893c54db0f',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a',
 'source/go1_hier_sim/go1_hier_sim/envs/three_roster_v22.py': '9d176affd4b233f188e52d8419d76ceeb93d8213297a38a5a3aead40fecd8704',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-17_go1-v22-triple-refine-s01-p1-d1/model_899.pt': '09a5357dca3964d33db84b96453d0f387f8924bf78664f3cf302e6eed352d76a',
 'reports/v22_ten_gate_trace512.jsonl': '84766794301f04dea417da18f6eca7547745e583af73027ac5348a95f57f3eb3',
 'reports/evaluate_stage1_2026-10-17_23-32-52_v22-ten-gate512.json': '510451cf03c59d1f920c3fdaf957edb21847806f60d000a42546c456d6198248',
}
for path, digest in expected.items():
    assert sha(path) == digest, f'drift: {path}'
assert (root / 'reports/v22_triple_refine.exit').read_text().startswith('exit=0 ')
assert (root / 'reports/v23_smoke4x16.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_triple_refine_v23_state.json', 'lp_triple_refine_v23_acrl_state.json',
             'lp_triple_refine_v23_results.txt', 'reports/v23_triple_refine.exit',
             'reports/lp_plan_2026-10-17_go1-v23-triple-refine-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-17_go1-v23-triple-refine-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v23-triple-refine*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v23-triple-refine*'))
print('V23_TRIPLE_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V23_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=triple_refinement INIT=v22 BRAKE=on\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v23.py \
  --base-id 2026-10-17_go1-v23-triple-refine \
  --config configs/curriculum_phases_triple_refine_v23.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v23_triple_refine.exit
exit "$status"
