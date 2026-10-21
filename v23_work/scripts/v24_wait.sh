#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 55); do
  if [ -f reports/v24_triple_refine.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v24_triple_refine.exit
    exit 0
  fi
  sleep 110
done
printf 'STILL %s ' "$(date -u +%FT%TZ)"
grep -aoE 'Learning iteration [0-9]+/[0-9]+' v35_phased_2026-10-18_go1-v24-triple-refine-s01-p1-d1.log 2>/dev/null | tail -1