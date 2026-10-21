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
 'scripts/run_student_v27.py': 'e4eeb9a8fa1be55340f8d7e041aa7abdec63f3c6bc6d1d2e5c2efac57689e08b',
 'configs/curriculum_phases_student_distill_v29b.json': '2b089a2949a922dd9a89752c1b5c93a479832bfd85a5384021235761b40a09f1',
 'configs/teachers_v29.json': '7690fdcf34938e9cc36dc7120e769bac1b188fdcda61467f144c2c18e5fea2eb',
 'source/go1_hier_sim/go1_hier_sim/cli.py': 'b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9',
 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': '08d609da744a769c93f801d2e9f546ffb9d7da2164611d3ad818599cbf6b6a9f',
 'source/go1_hier_sim/go1_hier_sim/envs/teacher_distill_v29.py': '685d945af453b334af6f9e5a989820b2f6f3869fcb50fa888a93e76e2b2e4dbd',
 'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',
 'source/go1_hier_sim/go1_hier_sim/training/on_policy.py': 'f1ea4784bf861dff18c09bcf5d426aa3063e356914b6f8504ee43b380f8ed6ec',
 'lp_phased_stop.flag': 'fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367',
 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt': '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
 'checkpoints/stage1/2026-10-16_go1-v15-tire-teacher-s01-p1-d1/model_899.pt': '4f18d4bcd62608a4707ac309af65277b715b0a55431b5ee20965a56bc23f08c7',
 'checkpoints/stage1/2026-10-17_15-32-19_2026-10-17_go1-v19p4-long-teacher-s01-p1-d1/model_899.pt': '75968c8885cfc842d1edd4f46ba15f3076d4912d100dd02a51840e99f4adbdc3',
}
for path, digest in expected.items():
    if digest is not None:
        assert sha(path) == digest, f'drift: {path}'
tmap = json.loads((root / 'configs/teachers_v29.json').read_text())
assert set(tmap) == {'general', 'tire', 'long'}
assert (root / 'reports/v29_probe_warmup.exit').read_text().startswith('exit=0 ')
assert (root / 'reports/v28_eleven_mixture.exit').read_text().startswith('exit=0 ')
assert torch.cuda.device_count() == 4
free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                      capture_output=True, text=True, check=True).stdout
assert min(int(x) for x in free.splitlines()) >= 12000, free
listing = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True, text=True, check=True).stdout
assert not [line for line in listing.splitlines()
            if line.strip().split()[0] in ('python', 'python3', 'torchrun', 'pt_elastic')
            and 'scripts/sim.py' in line], 'another Isaac job exists'
for path in ('lp_student_distill_v29b_state.json', 'lp_student_distill_v29b_acrl_state.json',
             'lp_student_distill_v29b_results.txt', 'reports/v29b_student_distill.exit',
             'reports/lp_plan_2026-10-19_go1-v29b-student-distill-s01-p1-d1.json',
             'reports/lp_gate_plan_2026-10-19_go1-v29b-student-distill-s01-p1-d1.json'):
    assert not (root / path).exists(), f'output exists: {path}'
assert not glob.glob(str(root / 'v35_*go1-v29b-student-distill*'))
assert not glob.glob(str(root / 'checkpoints/stage1/*go1-v29b-student-distill*'))
print('V29B_DISTILL_PREFLIGHT_OK', free.strip().replace('\n', ' / '), flush=True)
PY2
if [[ ${1:-} == --preflight-only ]]; then exit 0; fi
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_BRAKE_FAMILIES=tire_field,combined GO1_V23_TIRE_CAP=.60 GO1_V23_TIRE_PEN=.5 GO1_V29_TEACHER_DISTILL=1 GO1_V29_TEACHERS="$PWD/configs/teachers_v29.json" GO1_V29_EPS=.5 GO1_V29_LAMBDA=.5 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
printf 'V29B_START_UTC=%s PID=%s RANKS=4 PER_RANK=512 TEACHERS=general,tire,long INIT=v26\n' "$(date -u +%FT%TZ)" "$$"
set +e
/root/env_isaaclab/bin/python -u scripts/run_student_v27.py \
  --base-id 2026-10-19_go1-v29b-student-distill \
  --config configs/curriculum_phases_student_distill_v29b.json --max-stages 1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v29b_student_distill.exit
exit "$status"