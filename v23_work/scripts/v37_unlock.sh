#!/usr/bin/env bash
cd /config/Desktop/sim
for p in $(ps -eo pid,args | grep -E 'env_isaaclab/bin/python.*sim\.py|bin/torchrun' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 6
rm -f /config/.nvidia-omniverse/logs/omni.telemetry.transmitter.lock
rm -rf /config/.cache/ov/DerivedDataCache/app_instance_lock0
rm -f /config/.cache/ov/DerivedDataCache.lock /config/.cache/ov/_cache.lock
rm -rf /tmp/carb.* 2>/dev/null
rm -rf /dev/shm/* 2>/dev/null
echo CLEANED
export CUDA_VISIBLE_DEVICES=1
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_TASK_PLAN="$PWD/reports/v37_plan_soft9_32.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=1
export GO1_DR_MASS_UP=0.5 GO1_DR_PUSH_VEL=0 GO1_DR_MOTOR_PCT=0.02 GO1_DR_COM_SCALE=0.25
export GO1_OBS_DELAY_STEPS=0 GO1_OBS_DELAY_RANDOM=0 GO1_OBS_NOISE_STD=0 GO1_OBS_BIAS_RAD=0 GO1_OBS_SCAN_DROPOUT=0
export GO1_DR_FRICTION_STATIC_MIN=0.5 GO1_DR_FRICTION_STATIC_MAX=1.3 GO1_DR_FRICTION_DYNAMIC_MIN=0.4 GO1_DR_FRICTION_DYNAMIC_MAX=1.1
exec 8>reports/v37_lock.lock
flock -n 8 || { echo LOCK_BUSY; exit 70; }
timeout 480 ./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless --num-envs 32 --episodes 32 \
  --episode-seconds 133 --speed-min 0.35 --speed-max 0.8 --seed 2026 \
  --checkpoint checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt \
  --run-name v37-unlock-test > reports/v37_unlock.log 2>&1
echo "UNLOCK_RC=$? $(date -u +%FT%TZ)"
grep -ao '"completion_rate": [0-9.]*' reports/v37_unlock.log | head -1
grep -ao '"fallen_rate": [0-9.]*' reports/v37_unlock.log | head -1
tail -3 reports/v37_unlock.log | cut -c1-140
