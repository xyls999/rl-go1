#!/usr/bin/env bash
set -euo pipefail
candidate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace="${GO1_SIM_WORKSPACE:-/config/Desktop/sim}"
workspace="$(cd -- "${workspace}" && pwd -P)"
[[ "${candidate_root}" == "${workspace}/v128_work/v132_randomized/v1333" ]] || { echo "V1333 candidate root mismatch" >&2; exit 2; }
[[ "${1:-}" == "train" && "${V1333_TRAIN_CHILD:-}" == "1" ]] || { echo "V1333 wrapper accepts only runner-authorized training" >&2; exit 2; }
expected_runner="${candidate_root}/scripts/train_long_gap_warmstart_v1333_a05.py"
parent_cmdline="$(tr '\0' ' ' < "/proc/${PPID}/cmdline" 2>/dev/null || true)"
[[ "${parent_cmdline}" == *"${expected_runner} train"* ]] || { echo "V1333 training wrapper must be invoked by the A05 one-shot runner" >&2; exit 2; }
expected_python="$(readlink -f /root/env_isaaclab/bin/python)"
parent_exe="$(readlink -f "/proc/${PPID}/exe" 2>/dev/null || true)"
[[ -n "${expected_python}" && "${parent_exe}" == "${expected_python}" ]] || { echo "V1333 training wrapper parent is not the pinned A05 Python interpreter" >&2; exit 2; }
parent_environment="$(tr '\0' '\n' < "/proc/${PPID}/environ" 2>/dev/null || true)"
if ! grep -Fxq 'V1333_LONG_GAP_TRAIN_AUTHORIZED=one-shot' <<<"${parent_environment}"; then
  echo "V1333 training wrapper parent lacks the one-shot authorization token" >&2
  exit 2
fi
lock="${workspace}/reports/v7_sim.lock"
[[ "$(readlink -f /proc/$$/fd/9 2>/dev/null || true)" == "$(readlink -f "${lock}")" ]] || { echo "V1333 training subprocess does not hold the required simulator lock" >&2; exit 73; }
exec 8>>"${lock}"
if flock -n -E 75 8; then
  flock -u 8
  echo "V1333 simulator lock descriptor 9 is open but the flock is not held" >&2
  exit 73
else
  flock_status=$?
  [[ "${flock_status}" -eq 75 ]] || exit "${flock_status}"
fi
export ISAACLAB_PATH="${ISAACLAB_PATH:-/config/Desktop/IsaacLab}"
export VIRTUAL_ENV="${SIM_VIRTUAL_ENV:-/root/env_isaaclab}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export GO1_SIM_WORKSPACE="${workspace}"
export GO1_V1333_BASE_SOURCE="${workspace}/source/go1_hier_sim"
[[ -d "${GO1_V1333_BASE_SOURCE}" && ! -L "${GO1_V1333_BASE_SOURCE}" ]] || { echo "Active source fallback missing or symlinked" >&2; exit 1; }
export PYTHONPATH="${candidate_root}/source/go1_hier_sim:${GO1_V1333_BASE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset GO1_STAGE1_DIRECTION_ONLY
runtime_cudnn="${workspace}/.runtime-libs/nvidia/cudnn/lib"
[[ -f "${runtime_cudnn}/libcudnn_graph.so.9" ]] || { echo "Project cuDNN runtime missing" >&2; exit 1; }
[[ -f "${ISAACLAB_PATH}/isaaclab.sh" && -f "${workspace}/scripts/sim.py" ]] || { echo "Pinned IsaacLab or sim.py missing" >&2; exit 1; }
export LD_LIBRARY_PATH="${runtime_cudnn}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${workspace}/scripts/sim.py" "$@"
