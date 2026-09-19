#!/usr/bin/env bash
set -euo pipefail

sim_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
virtual_env="${SIM_VIRTUAL_ENV:-/root/env_isaaclab}"
target="${sim_workspace}/.runtime-libs"

if [[ ! -x "${virtual_env}/bin/pip" ]]; then
    echo "Missing simulation pip: ${virtual_env}/bin/pip" >&2
    exit 1
fi

"${virtual_env}/bin/pip" install --no-deps --upgrade --target "${target}" \
    'nvidia-cudnn-cu12==9.7.1.26'
test -f "${target}/nvidia/cudnn/lib/libcudnn_graph.so.9"
echo "Installed the project-local cuDNN runtime in ${target}"
