#!/usr/bin/env bash
cd /config/Desktop/sim
pkill -f v37_watch 2>/dev/null
pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
sleep 5
rm -rf /dev/shm/* 2>/dev/null
export CUDA_VISIBLE_DEVICES=1
export OMNI_KIT_LOG_LEVEL=verbose
export GO1_REWARD_CONFIG_PATH="$PWD/configs/rewards_stage1_gaps_v11_unified.toml"
export GO1_PPO_CONFIG_PATH="$PWD/configs/ppo_gaps_long.json"
export GO1_TASK_PLAN="$PWD/reports/v37_plan_soft9_32.json"
export GO1_POLICY_VARIANT=map_attn_v3 GO1_V8_TERRAIN_CONTROL=1 GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1 GO1_DDP_BACKEND=gloo NPROC_PER_NODE=1
timeout 240 ./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless --num-envs 4 --episodes 4 \
  --episode-seconds 20 --speed-min 0.35 --speed-max 0.8 --seed 2026 \
  --checkpoint checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt \
  --run-name v37-verbose > reports/v37_verbose.log 2>&1
echo "VERBOSE_RC=$?"
tail -6 reports/v37_verbose.log | cut -c1-160
