#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 18); do
  if [ -f reports/v26_triple_refine.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v26_triple_refine.exit
    grep -a GATE reports/v26_driver.log | tail -2
    grep -a ep, lp_triple_refine_v26_results.txt | tail -3
    exit 0
  fi
  sleep 110
done
printf 'STILL %s ' "$(date -u +%FT%TZ)"
grep -aoc Learning v35_phased_2026-10-18_go1-v26-triple-refine-s01-p1-d1.log 2>/dev/null || true