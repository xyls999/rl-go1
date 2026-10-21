#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 16); do
  if [ -f reports/v29_student_distill.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v29_student_distill.exit
    grep -a V29_DISTILL v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log | tail -2
    grep -a GATE reports/v29_driver.log | tail -2
    grep -a ep, lp_student_distill_v29_results.txt | tail -12
    exit 0
  fi
  sleep 110
done
printf 'STILL %s ' "$(date -u +%FT%TZ)"
grep -aoc Learning v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log 2>/dev/null || true