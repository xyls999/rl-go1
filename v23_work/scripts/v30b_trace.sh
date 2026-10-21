#!/usr/bin/env bash
set -euo pipefail
cd /config/Desktop/sim
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_TASK_PLAN="$PWD/reports/lp_gate_plan_2026-10-19_go1-v30b-combined-teacher-s01-p1-d1.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1
unset GO1_V18_PRELAUNCH_REWARD GO1_V14_SHORT_REWARD GO1_V9_GAP_CONTACT GO1_V10_OBSERVABLE_LEAP GO1_V19_PRELAUNCH_REWARD GO1_V19_PRELAUNCH_SCALE GO1_V23_TIRE_BRAKE GO1_V23_TIRE_CAP GO1_V23_TIRE_PEN GO1_V23_TIRE_BRAKE_FAMILIES GO1_V29_TEACHER_DISTILL GO1_V29_TEACHERS GO1_V29_EPS GO1_V29_LAMBDA GO1_DDP_BACKEND GO1_V9_DDP_BACKEND
export GO1_DR_MASS_UP=0.5 GO1_DR_PUSH_VEL=0 GO1_DR_MOTOR_PCT=0.02 GO1_DR_COM_SCALE=0.25
export GO1_OBS_DELAY_STEPS=0 GO1_OBS_DELAY_RANDOM=0 GO1_OBS_NOISE_STD=0 GO1_OBS_BIAS_RAD=0 GO1_OBS_SCAN_DROPOUT=0
export GO1_DR_FRICTION_STATIC_MIN=0.5 GO1_DR_FRICTION_STATIC_MAX=1.3 GO1_DR_FRICTION_DYNAMIC_MIN=0.4 GO1_DR_FRICTION_DYNAMIC_MAX=1.1
trace=reports/v30b_combined_trace512.jsonl
log=reports/v30b_combined_trace.log
test ! -e "$trace" && test ! -e "$log"
ckpt=$(ls checkpoints/stage1/2026-10-19_go1-v30b-combined-teacher-s01-p1-d1/model_899.pt)
flock -n reports/v7_sim.lock ./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless \
  --num-envs 512 --episodes 512 --episode-seconds 133 --speed-min 0.35 --speed-max 0.8 \
  --seed 2026 --checkpoint "$ckpt" --episode-trace "$trace" --run-name v30b-combined-trace > "$log" 2>&1
grep -a REPORT: "$log" | tail -1
sha256sum "$trace" "$ckpt"