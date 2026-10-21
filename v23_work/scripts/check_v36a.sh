#!/usr/bin/env bash
cd /config/Desktop/sim
grep -a "Learning iteration" v35_phased_2026-10-20*go1-v36a*log v35_phased_2026-10-19*go1-v36a*log 2>/dev/null | tail -1 | sed 's/.*iteration/ITER/'
grep -aE "MEAN episode reward" v35_phased_*go1-v36a*log 2>/dev/null | tail -2 | sed 's/^ *//' | cut -c1-90
tail -2 lp_combined_v36a_results.txt 2>/dev/null | cut -c1-150
cat reports/v36a_combined_soft8.exit 2>/dev/null
ps aux | grep -ac "[s]im.py"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1
date -u +%FT%TZ