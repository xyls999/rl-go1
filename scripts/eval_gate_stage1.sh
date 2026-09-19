#!/usr/bin/env bash
# Freeze gate for the Stage-1 locomotion policy.
#
# The layer may only be frozen once it traverses the randomised course at every
# speed x difficulty cell of the acceptance matrix. This script measures that
# matrix on one GPU and prints one row per cell; it never writes a checkpoint.
#
# The environment count must match the training batch. GPU PhysX contact
# resolution depends on the number of simulated environments, so a 1- or
# 4-environment probe reports a materially easier course than the 256
# environments the policy is trained with (measured: 0.82-0.85 difficulty
# completes at 1 and 4 environments and fails at 16 and 256).
#
# One cell per invocation, selected by <cell-index>, so a long matrix can be
# driven from short-lived remote sessions: run it with setsid and a line
# buffered log, or call it once per cell. The full matrix is 16 cells.
#
# usage: eval_gate_stage1.sh <checkpoint> <episodes> <speeds-csv> <difficulties-csv> [device] [episode-seconds] [env-count] [max-speed]
set -euo pipefail

ckpt="$1"
episodes="${2:-256}"
speeds="${3:-0.3,0.5,0.8,1.0}"
difficulties="${4:-0.25,0.5,0.75,1.0}"
device="${5:-cuda:1}"
episode_seconds="${6:-90}"
env_count="${7:-256}"
max_speed="${8:-1.0}"

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace}"
export GO1_REWARD_CONFIG_PATH="${workspace}/configs/rewards_stage1_v20_paper.toml"

echo "checkpoint: ${ckpt}"
echo "episodes/cell: ${episodes}  envs/cell: ${env_count}  device: ${device}  episode-seconds: ${episode_seconds}"
printf '%-8s %-8s %-14s %-10s %-10s %-10s %-10s %s\n' speed difficulty completion fallen oob collision velocity_rmse verdict
min_completion=0.90
max_collision=0.10
max_fallen=0.05
cells_file="$(mktemp /tmp/gate_cells_XXXXXX.jsonl)"
: > "${cells_file}"

IFS=',' read -r -a speed_list <<< "${speeds}"
IFS=',' read -r -a difficulty_list <<< "${difficulties}"
for difficulty in "${difficulty_list[@]}"; do
  for speed in "${speed_list[@]}"; do
    half=$(python3 -c "print(max(0.0, ${speed} - 0.1))")
    upper=$(python3 -c "print(min(${max_speed}, ${speed} + 0.1))")
    ./sim.sh evaluate --stage stage1 --terrain course --headless --device "${device}" \
      --num-envs "${env_count}" --episodes "${episodes}" --episode-seconds "${episode_seconds}" \
      --course-difficulty-min "${difficulty}" --course-difficulty-max "${difficulty}" \
      --speed-min "${half}" --speed-max "${upper}" \
      --checkpoint "${ckpt}" > /tmp/gate_cell.log 2>&1 || true
    # Newest report via python, not `ls | head`: under `set -o pipefail` the
    # SIGPIPE from head makes ls exit 141 and aborts the whole gate.
    json=$(python3 -c "import glob,os;print(max(glob.glob('reports/evaluate_stage1_*.json'), key=os.path.getmtime))")
    # -u keeps each row out of the block buffer, so a killed session cannot
    # swallow already measured cells.
    python3 -u - "${json}" "${speed}" "${difficulty}" "${cells_file}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
completion = report.get('completion_rate', float('nan'))
fallen = report.get('fallen_rate', float('nan'))
collision = report.get('collision_rate', float('nan'))
out_of_bounds = report.get('out_of_bounds_rate', float('nan'))
velocity = report.get('velocity_xy_rmse', float('nan'))
# Acceptance agreed with the user on 2026-09-19: 95%/fallen=0 was unattainable
# at 256 episodes (one fall is already 0.39%) and the hardest cell (14-16 cm
# stairs at 1 m/s) is exactly what the second layer exists to avoid by
# commanding a lower speed.
ok = completion >= 0.90 and collision <= 0.10 and fallen <= 0.05
print('%-8s %-8s %-14.4f %-10.4f %-10.4f %-10.4f %-10.4f %s' % (
    sys.argv[2], sys.argv[3], completion, fallen, out_of_bounds, collision,
    velocity, 'PASS' if ok else 'FAIL'))
with open(sys.argv[4], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({
        "speed": float(sys.argv[2]), "difficulty": float(sys.argv[3]),
        "completion_rate": completion, "fallen_rate": fallen,
        "out_of_bounds_rate": out_of_bounds, "collision_rate": collision,
        "velocity_xy_rmse": velocity,
    }) + "\n")
PY
  done
done
echo "acceptance: every cell needs completion >= 0.90, collision <= 0.10, fallen <= 0.05 (agreed 2026-09-19)"
echo "lateral drift is checked separately with a single --debug-motion run (target < 1.5 m)"

# One certifiable record for the whole matrix: certify_checkpoint refuses a
# Stage-1 gate record unless every measured cell holds.
record="${workspace}/reports/gate_stage1_$(date -u +%Y%m%d_%H%M%S).json"
PYTHONPATH="${workspace}/source/go1_hier_sim" "${VIRTUAL_ENV:-/root/env_isaaclab}/bin/python" \
  "${workspace}/scripts/gate_record.py" \
  --checkpoint "${ckpt}" --cells "${cells_file}" --output "${record}" \
  --envs-per-cell "${env_count}" --episode-seconds "${episode_seconds}" \
  --max-speed "${max_speed}" \
  --min-completion "${min_completion}" --max-collision "${max_collision}" --max-fallen "${max_fallen}"
echo "certify with: PYTHONPATH=source/go1_hier_sim /root/env_isaaclab/bin/python -c \"from go1_hier_sim.checkpoints import certify_checkpoint; print(certify_checkpoint('${ckpt}', '${record}'))\""
