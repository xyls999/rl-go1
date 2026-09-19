#!/usr/bin/env bash
set -euo pipefail
sim_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/config/Desktop/IsaacLab}"
export VIRTUAL_ENV="${SIM_VIRTUAL_ENV:-/root/env_isaaclab}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${sim_workspace}/source/go1_hier_sim${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
runtime_cudnn="${sim_workspace}/.runtime-libs/nvidia/cudnn/lib"
if [[ -n "${DISPLAY:-}" ]]; then
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
fi
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM=xterm
fi
if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "Missing simulation Python: ${VIRTUAL_ENV}/bin/python" >&2
    exit 1
fi
if [[ "${1:-}" == "test" ]]; then
    shift
    exec "${VIRTUAL_ENV}/bin/python" -B -m pytest "${sim_workspace}/tests" "$@"
fi
if [[ ! -f "${runtime_cudnn}/libcudnn_graph.so.9" ]]; then
    echo "Missing the project-local compatible cuDNN runtime." >&2
    echo "Run: ${sim_workspace}/scripts/bootstrap_runtime.sh" >&2
    exit 1
fi
export LD_LIBRARY_PATH="${runtime_cudnn}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${sim_workspace}/scripts/sim.py" "$@"
