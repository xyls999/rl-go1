#!/usr/bin/env bash
# Only explicitly authorized test and four-family PhysX modes can overlay live
# sources. Training and evaluation modes do not exist in this script.
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-}"
if [[ "$mode" == mock-restore || "$mode" == mock-signal ]]; then
  [[ "${V1322_MOCK:-}" == 1 && "${PREFLIGHT_ONLY:-1}" == 1 ]] || { echo 'MOCK_DENIED: mock-only' >&2; exit 84; }
  [[ -n "${V1322_MOCK_WORKSPACE:-}" && -d "$V1322_MOCK_WORKSPACE" ]] || { echo 'MOCK_DENIED: sandbox missing' >&2; exit 84; }
  workspace="$(cd "$V1322_MOCK_WORKSPACE" && pwd -P)"
  [[ "$workspace" != /config && "$workspace" != /config/* && "$workspace" != "$root" && "$workspace" != "$root"/* ]] || { echo 'MOCK_DENIED: production/experiment path' >&2; exit 84; }
  [[ -d "$workspace/reports" && -d "$workspace/source/go1_hier_sim/go1_hier_sim/envs" ]] || { echo 'MOCK_DENIED: invalid sandbox' >&2; exit 84; }
  command -v realpath >/dev/null || { echo 'MOCK_DENIED: realpath unavailable' >&2; exit 84; }
  # A source/reports ancestor must never redirect a mock copy into live files.
  mock_safe_path() {
    local relative="$1" path="$workspace" component resolved
    local -a components
    IFS=/ read -r -a components <<< "$relative"
    for component in "${components[@]}"; do
      [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
      path="$path/$component"
      [[ ! -L "$path" ]] || return 1
    done
    resolved="$(realpath -m -- "$path")" || return 1
    [[ "$resolved" == "$workspace/"* ]]
  }
  mock_safe_path reports || { echo 'MOCK_DENIED: symlinked/escaping reports path' >&2; exit 84; }
  files=(source/go1_hier_sim/go1_hier_sim/cli.py
         source/go1_hier_sim/go1_hier_sim/envs/runtime.py
         source/go1_hier_sim/go1_hier_sim/envs/geometry.py
         source/go1_hier_sim/go1_hier_sim/envs/config.py
         source/go1_hier_sim/go1_hier_sim/strict_gate_v1322.py
         source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py
         source/go1_hier_sim/go1_hier_sim/envs/gap_score_v1321.py
         source/go1_hier_sim/go1_hier_sim/envs/landing_stability_v131.py)
  for rel in "${files[@]}"; do
    mock_safe_path "$rel" || { echo 'MOCK_DENIED: symlinked target/parent or escaping path' >&2; exit 84; }
  done
  lock="$workspace/reports/.v1322_mock.lock"
  mkdir "$lock" 2>/dev/null || { echo 'MOCK_DENIED: lock busy' >&2; exit 86; }
  backup="$(mktemp -d "$workspace/reports/v1322-backup.XXXXXX")"
  declare -A original_hash=() existed=()
  # No live byte changes until every existing target is safely backed up.
  trap 'rm -rf "$backup"; rmdir "$lock"' EXIT
  for rel in "${files[@]}"; do
    if [[ -e "$workspace/$rel" ]]; then
      existed[$rel]=yes
      original_hash[$rel]="$(sha256sum "$workspace/$rel" | cut -d' ' -f1)"
      mkdir -p "$(dirname "$backup/$rel")"
      cp "$workspace/$rel" "$backup/$rel"
      [[ "$(sha256sum "$backup/$rel" | cut -d' ' -f1)" == "${original_hash[$rel]}" ]]
    fi
  done
  restore_mock() {
    local rc="$?" rel actual restore_failed=0
    trap - EXIT INT TERM
    for rel in "${files[@]}"; do
      if [[ "${existed[$rel]:-}" == yes ]]; then
        if ! cp "$backup/$rel" "$workspace/$rel"; then
          restore_failed=1
          continue
        fi
        actual="$(sha256sum "$workspace/$rel" | cut -d' ' -f1)"
        [[ "$actual" == "${original_hash[$rel]}" ]] || restore_failed=1
      else
        rm -f "$workspace/$rel" || restore_failed=1
        [[ ! -e "$workspace/$rel" ]] || restore_failed=1
      fi
    done
    if [[ "$restore_failed" == 0 ]]; then
      rm -rf "$backup" || restore_failed=1
    fi
    rmdir "$lock" || restore_failed=1
    if [[ "$restore_failed" != 0 ]]; then
      echo "MOCK_RESTORE_FAILED: backups retained at $backup" >&2
      exit 90
    fi
    echo "MOCK_RESTORED rc=$rc"
    exit "$rc"
  }
  trap restore_mock EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for rel in "${files[@]}"; do
    mkdir -p "$(dirname "$workspace/$rel")"
    cp "$root/$rel" "$workspace/$rel"
  done
  [[ "$mode" != mock-signal ]] || kill -TERM "$$"
  # Fault injection is available only inside the explicitly guarded mock sandbox.
  if [[ "${V1322_MOCK_FAIL_RESTORE:-}" == 1 ]]; then
    rm -f "$backup/source/go1_hier_sim/go1_hier_sim/cli.py"
  fi
  false # deliberate failure exercises the EXIT trap (never a real run)
fi
if [[ "$mode" != preflight && "$mode" != preflight-test && "$mode" != preflight-smoke && "$mode" != test && "$mode" != smoke ]] || [[ "${V1322_MOCK:-}" == 1 ]]; then
  echo 'BLOCKED: training and final gate are not implemented in this runner' >&2
  exit 85
fi
if [[ "$mode" == preflight || "$mode" == preflight-test || "$mode" == preflight-smoke ]]; then
  [[ "${PREFLIGHT_ONLY:-}" == 1 ]] || { echo 'PREFLIGHT_ONLY=1 required' >&2; exit 85; }
else
  [[ "${V1322_RUN_AUTHORIZED:-}" == 'test-smoke-only' && "${PREFLIGHT_ONLY:-0}" != 1 ]] || {
    echo 'BLOCKED: test/smoke needs explicit authorization' >&2; exit 85;
  }
fi
command -v flock >/dev/null || { echo 'PREFLIGHT_BLOCKED: exclusive flock unavailable' >&2; exit 86; }
workspace=/config/Desktop/sim
[[ -d "$workspace/reports" && ! -L "$workspace/reports" ]] || { echo 'PREFLIGHT_BLOCKED: remote reports directory missing or symlinked' >&2; exit 86; }
[[ -f "$workspace/reports/v7_sim.lock" && ! -L "$workspace/reports/v7_sim.lock" ]] || { echo 'PREFLIGHT_BLOCKED: preexisting regular simulator lock required' >&2; exit 86; }
exec 9>>"$workspace/reports/v7_sim.lock"
flock -n 9 || { echo 'PREFLIGHT_BLOCKED: simulator lock busy' >&2; exit 86; }
if [[ "$mode" == preflight || "$mode" == preflight-test || "$mode" == preflight-smoke ]]; then
  /root/env_isaaclab/bin/python "$root/scripts/preflight_v1322.py" "$mode"
else
  exec /root/env_isaaclab/bin/python "$root/scripts/production_v1322.py" "$mode"
fi
