#!/usr/bin/env bash
set -u
cd /config/Desktop/sim
for k in $(seq 1 17); do
  if [ -f reports/v27_eleven_mixture.exit ]; then
    printf 'DONE %s ' "$(date -u +%FT%TZ)"
    cat reports/v27_eleven_mixture.exit
    grep -a GATE reports/v27_driver.log | tail -2
    grep -a ep, lp_eleven_mixture_v27_results.txt | tail -12
    exit 0
  fi
  sleep 110
done
printf 'STILL %s ' "$(date -u +%FT%TZ)"
grep -aoc Learning v35_phased_2026-10-18_go1-v27-eleven-mixture-s01-p1-d1.log 2>/dev/null || true