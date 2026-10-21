#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 40); do
  if [ -f reports/v23_triple_refine.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v23_triple_refine.exit
    exit 0
  fi
  sleep 58
done
printf 'STILL_RUNNING %s ' "$(date -u +%FT%TZ)"
grep -aoE 'Learning iteration [0-9]+/[0-9]+' v35_phased_2026-10-17_go1-v23-triple-refine-s01-p1-d1.log 2>/dev/null | tail -1
[ -f v35_gate_2026-10-17_go1-v23-triple-refine-s01-p1-d1.log ] && echo GATE_RUNNING
