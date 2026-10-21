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
 'scripts/run_student_v32.py': '0ee4a24d07ae76e6e576a4e8e5e4ba7a851960ed1c4690452c51513de02e5f5f',
 'configs/curriculum_phases_multires_v55.json': 'a679bb7b7a8ab7d2e96c5fb72ebd1506c08a91128d5a228a5f50ea12cf958a39',
 'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py': '4c3a0fee36b3259d4f7f7f380d70f1fc53c4cf911e5316990217712face850db',
 'source/go1_hier_sim/go1_hier_sim/envs/config.py': 'a7920ced5ede09b69b4db403f335130869c91cc45087f44378e0d69e2278b351',
 'source/go1_hier_sim/go1_hier_sim/cli.py': 'e7136aa719e19821d153788d3c909747d55dcb97990205a5b2ec67a39232d1ed',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '8ed99bce3b83990c08a89351786d66fc7de3bcfbecd82152cf0a6a3388e86cdb',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py': '6500267b26ad1d14401e0b8084e45a8845a038aab2de3d1c401203908a9b21e5',
 'source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py': 'f02efc7201113f9d89bef8a01e9cb1c25a077182d92f7dc54bb9ea71c322f2b2',
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
for path in ('lp_combined_v55_state.json', 'lp_combined_v55_acrl_state.json',
             'lp_combined_v55_results.txt', 'reports/v55_combined.exit',
             'reports/lp_plan_2026-10-19_go1-v55-multires-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-19_go1-v55-multires-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v55-multires*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v55-multires*'))
print('V32A_COMBINED_SOFT_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
unset GO1_V36_SOFT_TERMINATION
export GO1_V36_SOFT_TERMINATION=1 GO1_V8_LONG_GAP_THRESHOLD=0.05 GO1_V19_MIN_WIDTH=0.05
export GO1_PRIVILEGED_ACTOR=1 GO1_MULTIRES_SCAN=1
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND GO1_V29_TEACHER_DISTILL GO1_V29_TEACHERS GO1_V29_EPS GO1_V29_LAMBDA GO1_V23_TIRE_BRAKE GO1_V23_TIRE_BRAKE_FAMILIES GO1_V23_TIRE_CAP GO1_V23_TIRE_PEN
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V32A_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 ROSTER=single:combined GATE=combined INIT=v26\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v32.py \
  --base-id 2026-10-19_go1-v55-multires \
  --config configs/curriculum_phases_multires_v55.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v55_combined.exit
exit "$status"
