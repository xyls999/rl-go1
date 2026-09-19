#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace}"
export DISPLAY="${DISPLAY:-:1}"
export NPROC_PER_NODE=2

base_id="${GO1_V19_BASE_ID:-$(date +%Y-%m-%d_%H-%M-%S)_go1-stage1-v19-curriculum}"
v17="checkpoints/stage1/2026-09-16_15-25-26_go1-stage1-v17-soft-phase-trot-10000/model_9999.pt"

latest_checkpoint() {
    local directory="$1"
    find "${directory}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' \
        | sort -V | tail -n 1
}

run_phase() {
    local run_id="$1" iterations="$2" warmup="$3" difficulty_min="$4"
    local difficulty_max="$5" speed_min="$6" speed_max="$7" initial="$8"
    echo "[$(date -Is)] START ${run_id} init=${initial} difficulty=${difficulty_min}-${difficulty_max} speed=${speed_min}-${speed_max}"
    GO1_RUN_ID="${run_id}" ./scripts/torchrun_sim.sh train \
        --stage stage1 --terrain course --headless --num-envs 1024 \
        --iterations "${iterations}" --episode-seconds 30 \
        --course-difficulty-min "${difficulty_min}" \
        --course-difficulty-max "${difficulty_max}" \
        --speed-min "${speed_min}" --speed-max "${speed_max}" \
        --initialize-from "${initial}" --critic-warmup-iterations "${warmup}" \
        --run-name "${run_id}"
    echo "[$(date -Is)] FINISH ${run_id}"
}

phase1="${base_id}-p1-easy"
phase2="${base_id}-p2-medium"
phase3="${base_id}-p3-complex"

# Phase 1: retain the V17 gait and learn the now-signed task objective on a
# long lead-in, gentle grades and nearly regular 3--4 cm stairs.
run_phase "${phase1}" 2000 300 0.00 0.20 0.25 0.45 "${v17}"
p1_checkpoint="$(latest_checkpoint "checkpoints/stage1/${phase1}")"
test -n "${p1_checkpoint}"

# Phase 2: moderate roughness, grade and stair variation.
run_phase "${phase2}" 3000 100 0.20 0.50 0.30 0.55 "${p1_checkpoint}"
p2_checkpoint="$(latest_checkpoint "checkpoints/stage1/${phase2}")"
test -n "${p2_checkpoint}"

# Phase 3: the target mixed course, still bounded below V18's untraversable
# initial distribution.
run_phase "${phase3}" 5000 100 0.45 0.75 0.30 0.60 "${p2_checkpoint}"

echo "[$(date -Is)] V19 CURRICULUM COMPLETE"
echo "FINAL_CHECKPOINT=$(latest_checkpoint "checkpoints/stage1/${phase3}")"
