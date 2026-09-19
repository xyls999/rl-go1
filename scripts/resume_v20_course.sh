#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace}"
export DISPLAY="${DISPLAY:-:1}"
export NPROC_PER_NODE=2
export GO1_REWARD_CONFIG_PATH="${workspace}/configs/rewards_stage1_v20_paper.toml"
export GO1_PPO_CONFIG_PATH="${workspace}/configs/ppo_v20_paper.json"

base_id="${GO1_V20_BASE_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)_go1-stage1-v20-course}"
initial="${GO1_V20_INIT:-checkpoints/stage1/2026-09-17_07-21-00_go1-stage1-v20-paper-p1-flat/model_900.pt}"

latest_checkpoint() {
    find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1
}

run_phase() {
    local run_id="$1" iterations="$2" warmup="$3" dmin="$4" dmax="$5" smin="$6" smax="$7" init="$8"
    GO1_RUN_ID="${run_id}" ./scripts/torchrun_sim.sh train \
      --stage stage1 --terrain course --headless --num-envs 512 \
      --iterations "${iterations}" --episode-seconds 100 \
      --course-difficulty-min "${dmin}" --course-difficulty-max "${dmax}" \
      --speed-min "${smin}" --speed-max "${smax}" \
      --initialize-from "${init}" --critic-warmup-iterations "${warmup}" \
      --run-name "${run_id}"
    latest_checkpoint "checkpoints/stage1/${run_id}"
}

p2="${base_id}-p2-course-easy"
initial="$(run_phase "${p2}" 2400 50 0.00 0.25 0.25 0.38 "${initial}" | tail -n 1)"
test -s "${initial}"

p3="${base_id}-p3-course-medium"
initial="$(run_phase "${p3}" 3200 50 0.25 0.60 0.30 0.42 "${initial}" | tail -n 1)"
test -s "${initial}"

p4="${base_id}-p4-course-hard"
initial="$(run_phase "${p4}" 4200 50 0.60 1.00 0.30 0.42 "${initial}" | tail -n 1)"
test -s "${initial}"

echo "FINAL_CHECKPOINT=${initial}"
