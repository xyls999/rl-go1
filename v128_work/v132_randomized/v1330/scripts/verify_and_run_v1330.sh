#!/usr/bin/env bash
#!/usr/bin/env bash
# Exclusive-lock guarded V1330 fresh-seed audit and exactly one training run.
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-}"
[[ "$#" == 1 ]] || { echo 'BLOCKED: exactly one V1330 mode is required' >&2; exit 85; }
case "$mode" in
  preflight-seed|preflight-train)
    [[ "${PREFLIGHT_ONLY:-}" == 1 ]] || { echo 'PREFLIGHT_ONLY=1 required' >&2; exit 85; }
    [[ -z "${V1330_RUN_AUTHORIZED:-}" ]] || {
      echo 'BLOCKED: preflight cannot carry V1330 run authorization' >&2; exit 85;
    }
    ;;
  audit-train-seed)
    [[ "${PREFLIGHT_ONLY:-0}" != 1 ]] || { echo 'BLOCKED: seed audit cannot use PREFLIGHT_ONLY' >&2; exit 85; }
    [[ "${V1330_RUN_AUTHORIZED:-}" == seed-audit-train-only ]] || {
      echo 'BLOCKED: explicit V1330 seed-audit authorization absent' >&2; exit 85;
    }
    ;;
  train)
    [[ "${PREFLIGHT_ONLY:-0}" != 1 ]] || { echo 'BLOCKED: training cannot use PREFLIGHT_ONLY' >&2; exit 85; }
    [[ "${V1330_RUN_AUTHORIZED:-}" == one-train-only ]] || {
      echo 'BLOCKED: explicit V1330 one-train authorization absent' >&2; exit 85;
    }
    ;;
  *)
    echo 'BLOCKED: V1330 implements only seed audit and one training run; no evaluation or gate' >&2
    exit 85
    ;;
esac
[[ "${V1330_MOCK:-0}" != 1 ]] || { echo 'BLOCKED: V1330 mock mode is not implemented' >&2; exit 85; }
command -v flock >/dev/null || { echo 'PREFLIGHT_BLOCKED: flock unavailable' >&2; exit 86; }
workspace=/config/Desktop/sim
[[ -d "$workspace/reports" && ! -L "$workspace/reports" ]] || {
  echo 'PREFLIGHT_BLOCKED: remote reports directory missing/symlinked' >&2; exit 86;
}
[[ -f "$workspace/reports/v7_sim.lock" && ! -L "$workspace/reports/v7_sim.lock" ]] || {
  echo 'PREFLIGHT_BLOCKED: pre-existing regular simulator lock required' >&2; exit 86;
}
exec 9>>"$workspace/reports/v7_sim.lock"
flock -n 9 || { echo 'PREFLIGHT_BLOCKED: simulator lock busy' >&2; exit 86; }
export PYTHONDONTWRITEBYTECODE=1
exec /root/env_isaaclab/bin/python "$root/scripts/production_v1330.py" "$mode"
