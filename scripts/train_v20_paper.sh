#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace}"
export DISPLAY="${DISPLAY:-:1}"
export NPROC_PER_NODE=2
export GO1_REWARD_CONFIG_PATH="${workspace}/configs/rewards_stage1_v20_paper.toml"
export GO1_PPO_CONFIG_PATH="${workspace}/configs/ppo_v20_paper.json"

base_id="${GO1_V20_BASE_ID:-$(date +%Y-%m-%d_%H-%M-%S)_go1-stage1-v20-paper}"

latest_checkpoint() {
    find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1
}

if [[ -n "${GO1_V20_INIT:-}" ]]; then
    initial="${GO1_V20_INIT}"
else
    initial="$(latest_checkpoint checkpoints/stage1/2026-09-17_03-40-00_go1-stage1-v19-curriculum-p2-medium)"
    initial="${initial:-checkpoints/stage1/2026-09-16_15-25-26_go1-stage1-v17-soft-phase-trot-10000/model_9999.pt}"
fi

run_phase() {
    local run_id="$1" terrain="$2" iterations="$3" warmup="$4" episode_seconds="$5"
    local difficulty_min="$6" difficulty_max="$7" speed_min="$8" speed_max="$9" init="${10}"
    echo "[$(date -Is)] START ${run_id} terrain=${terrain} difficulty=${difficulty_min}-${difficulty_max} speed=${speed_min}-${speed_max}"
    args=(--stage stage1 --terrain "${terrain}" --headless --num-envs 1024
          --iterations "${iterations}" --episode-seconds "${episode_seconds}"
          --speed-min "${speed_min}" --speed-max "${speed_max}"
          --initialize-from "${init}" --critic-warmup-iterations "${warmup}"
          --run-name "${run_id}")
    if [[ "${terrain}" == "course" ]]; then
        args+=(--course-difficulty-min "${difficulty_min}" --course-difficulty-max "${difficulty_max}")
    fi
    GO1_RUN_ID="${run_id}" ./scripts/torchrun_sim.sh train "${args[@]}"
    latest_checkpoint "checkpoints/stage1/${run_id}"
}

p1="${base_id}-p1-flat"
initial="$(run_phase "${p1}" flat 1000 250 20 0.00 0.00 0.25 0.40 "${initial}" | tail -n 1)"
test -s "${initial}"

p2="${base_id}-p2-course-easy"
initial="$(run_phase "${p2}" course 1800 150 35 0.00 0.20 0.35 0.50 "${initial}" | tail -n 1)"
test -s "${initial}"

p3="${base_id}-p3-course-medium"
initial="$(run_phase "${p3}" course 2400 150 35 0.20 0.50 0.35 0.55 "${initial}" | tail -n 1)"
test -s "${initial}"

p4="${base_id}-p4-course-hard"
initial="$(run_phase "${p4}" course 3200 150 35 0.45 0.75 0.40 0.60 "${initial}" | tail -n 1)"
test -s "${initial}"

echo "[$(date -Is)] V20 PAPER-STYLE CURRICULUM COMPLETE"
echo "FINAL_CHECKPOINT=${initial}"
