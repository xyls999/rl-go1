#!/usr/bin/env bash
# Second-layer (speed policy) acceptance matrix for the frozen schema-5 stack.
#
# Stage 2 has no commanded speed band: the speed policy emits [vx, vy, wz] itself
# and `make_config` only applies `--speed-min/--speed-max` to Stage-1 locomotion
# (Stage 2 is fixed at 0 <= vx <= 0.8, which is also why a larger value is
# rejected outright). The only acceptance dimension for this layer is therefore
# terrain difficulty. A cell passes on the same terms as Stage 1: completion
# >= 0.90, collision <= 0.10, fallen <= 0.05 at 256 environments, the batch size
# the policies are trained with. Usage:
#   scripts/eval_gate_stage2.sh <locomotion-ckpt> <navigation-ckpt> <episodes> \
#       <difficulties-csv> [device] [episode-seconds] [env-count]
set -uo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace}"

locomotion="${1:?locomotion checkpoint is required}"
navigation="${2:?navigation checkpoint is required}"
episodes="${3:-256}"
difficulties="${4:?difficulties csv is required, e.g. 0.5,0.75,0.9}"
device="${5:-cuda:0}"
episode_seconds="${6:-150}"
env_count="${7:-256}"

export DISPLAY="${DISPLAY:-:1}"
export GO1_REWARD_CONFIG_PATH="${workspace}/configs/rewards_stage1_v20_paper.toml"
export GO1_PPO_CONFIG_PATH="${workspace}/configs/ppo_v20_paper.json"

test -s "${locomotion}"
test -s "${navigation}"

run_cell() {
    local difficulty="$1"
    echo "=== CELL difficulty=${difficulty} envs=${env_count} $(date -Is)"
    ./sim.sh evaluate --stage stage2 --terrain course --headless --device "${device}" \
        --num-envs "${env_count}" --episodes "${episodes}" --episode-seconds "${episode_seconds}" \
        --course-difficulty-min "${difficulty}" --course-difficulty-max "${difficulty}" \
        --locomotion-checkpoint "${locomotion}" --allow-uncertified-locomotion \
        --checkpoint "${navigation}" 2>&1 |
        grep -aE '"completion_rate"|"collision_rate"|"fallen_rate"|"mean_command_delta_l1"|"command_step_large_fraction"|REPORT:|Traceback|cooking failure|Error'
}

IFS=',' read -ra difficulty_list <<< "${difficulties}"
for difficulty in "${difficulty_list[@]}"; do
    run_cell "${difficulty}"
done
echo "[$(date -Is)] STAGE2_GATE_DONE"
