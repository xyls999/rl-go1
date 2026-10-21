#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 16); do
  if [ -f reports/v30_combined_teacher.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v30_combined_teacher.exit
    grep -a GATE reports/v30_driver.log | tail -2
    grep -a ep, lp_combined_v30_results.txt | tail -2
    exit 0
  fi
  sleep 110
done
printf 'STILL %s ' "$(date -u +%FT%TZ)"
grep -aoc Learning v35_phased_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.log 2>/dev/null || true