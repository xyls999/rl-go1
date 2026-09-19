#!/usr/bin/env bash
set -euo pipefail

# Launch the same simulator entry point as ../sim.sh, but under torchrun so
# RSL-RL can synchronize PPO updates across all local GPU workers.
sim_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/config/Desktop/IsaacLab}"
export VIRTUAL_ENV="${SIM_VIRTUAL_ENV:-/root/env_isaaclab}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${sim_workspace}/source/go1_hier_sim${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

runtime_cudnn="${sim_workspace}/.runtime-libs/nvidia/cudnn/lib"
if [[ ! -f "${runtime_cudnn}/libcudnn_graph.so.9" ]]; then
    echo "Missing the project-local compatible cuDNN runtime." >&2
    exit 1
fi
export LD_LIBRARY_PATH="${runtime_cudnn}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${VIRTUAL_ENV}/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE:-2}" \
    "${sim_workspace}/scripts/sim.py" "$@"
