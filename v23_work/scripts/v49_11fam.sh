#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_TASK_PLAN="$PWD/reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=4
export GO1_DR_MASS_UP=0.5 GO1_DR_PUSH_VEL=0 GO1_DR_MOTOR_PCT=0.02 GO1_DR_COM_SCALE=0.25
export GO1_OBS_DELAY_STEPS=0 GO1_OBS_DELAY_RANDOM=0 GO1_OBS_NOISE_STD=0 GO1_OBS_BIAS_RAD=0 GO1_OBS_SCAN_DROPOUT=0
export GO1_DR_FRICTION_STATIC_MIN=0.5 GO1_DR_FRICTION_STATIC_MAX=1.3 GO1_DR_FRICTION_DYNAMIC_MIN=0.4 GO1_DR_FRICTION_DYNAMIC_MAX=1.1
exec 9>reports/v7_sim.lock
flock 9
rm -f reports/v49_11fam_trace.jsonl
./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless --num-envs 512 --episodes 512 \
  --episode-seconds 133 --speed-min 0.35 --speed-max 0.8 --seed 2026 \
  --checkpoint checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt \
  --episode-trace reports/v49_11fam_trace.jsonl \
  --run-name v49-v26-11family > reports/v49_11fam.log 2>&1
echo "exit=$? $(date -u +%FT%TZ)" > reports/v49_11fam.exit
grep -ao '"completion_rate": [0-9.]*' reports/v49_11fam.log | head -1 | tee -a reports/v49_11fam.exit
