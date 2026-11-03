#!/usr/bin/env bash
# V132.3 test/PhysX/development pilot only; no final gate or teacher queue.
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-}"
case "$mode" in
  preflight-test|preflight-smoke|preflight-pilot)
    [[ "${PREFLIGHT_ONLY:-}" == 1 ]] || { echo 'PREFLIGHT_ONLY=1 required' >&2; exit 85; }
    ;;
  test|smoke|pilot)
    authorization=test-smoke-only
    [[ "$mode" != pilot ]] || authorization=paired-pilot-only
    [[ "${V1323_RUN_AUTHORIZED:-}" == "$authorization" && "${PREFLIGHT_ONLY:-0}" != 1 ]] || {
      echo "BLOCKED: $mode needs explicit authorization" >&2; exit 85;
    }
    ;;
  *) echo 'BLOCKED: only guarded V132.3 test, smoke and pilot are implemented' >&2; exit 85 ;;
esac
command -v flock >/dev/null || { echo 'PREFLIGHT_BLOCKED: exclusive flock unavailable' >&2; exit 86; }
workspace=/config/Desktop/sim
[[ -d "$workspace/reports" && ! -L "$workspace/reports" ]] || {
  echo 'PREFLIGHT_BLOCKED: remote reports directory missing or symlinked' >&2; exit 86;
}
[[ -f "$workspace/reports/v7_sim.lock" && ! -L "$workspace/reports/v7_sim.lock" ]] || {
  echo 'PREFLIGHT_BLOCKED: preexisting regular simulator lock required' >&2; exit 86;
}
exec 9>>"$workspace/reports/v7_sim.lock"
flock -n 9 || { echo 'PREFLIGHT_BLOCKED: simulator lock busy' >&2; exit 86; }
if [[ "$mode" == preflight-* ]]; then
  exec /root/env_isaaclab/bin/python "$root/scripts/preflight_v1323.py" "$mode"
else
  exec /root/env_isaaclab/bin/python "$root/scripts/production_v1323.py" "$mode"
fi
