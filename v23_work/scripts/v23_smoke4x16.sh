#!/usr/bin/env bash
set -euo pipefail
cd /config/Desktop/sim
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_TASK_PLAN="$PWD/reports/v22_smoke_plan16.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V23_TIRE_BRAKE=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V9_DDP_BACKEND
export GO1_DR_MASS_UP=0.5 GO1_DR_PUSH_VEL=0 GO1_DR_MOTOR_PCT=0.02 GO1_DR_COM_SCALE=0.25
export GO1_OBS_DELAY_STEPS=0 GO1_OBS_DELAY_RANDOM=0 GO1_OBS_NOISE_STD=0 GO1_OBS_BIAS_RAD=0 GO1_OBS_SCAN_DROPOUT=0
export GO1_DR_FRICTION_STATIC_MIN=0.5 GO1_DR_FRICTION_STATIC_MAX=1.3 GO1_DR_FRICTION_DYNAMIC_MIN=0.4 GO1_DR_FRICTION_DYNAMIC_MAX=1.1
log=reports/v23_smoke4x16.log
if [ -e reports/v23_smoke4x16.exit ] && head -1 reports/v23_smoke4x16.exit | grep -q '^exit=0 '; then
  echo 'smoke already passed' >&2; exit 3
fi
rm -f "$log" reports/v23_smoke4x16.exit
set +e
exec 9>reports/v7_sim.lock
flock -n 9 || { echo ISAAC_LOCK_BUSY >&2; exit 70; }
bash scripts/torchrun_sim.sh train --stage stage1 --terrain lp_acrl --headless \
  --num-envs 16 --iterations 2 --episode-seconds 133 \
  --course-difficulty-min 0 --course-difficulty-max 0.6 \
  --speed-min 0.35 --speed-max 0.8 \
  --initialize-from checkpoints/stage1/2026-10-17_go1-v22-triple-refine-s01-p1-d1/model_899.pt \
  --critic-warmup-iterations 0 --run-name 2026-10-17_go1-v23-smoke4x16 > "$log" 2>&1
status=$?
set -e
printf 'exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)" > reports/v23_smoke4x16.exit
exit "$status"
