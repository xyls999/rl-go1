#!/usr/bin/env bash
cd /config/Desktop/sim
log=v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log
echo ITER: $(grep -a 'Learning iteration' "$log" | tail -1 | tr -s ' ')
tail -4 lp_combined_v33a_results.txt
echo RUNNING: $(ps aux | grep -c '[s]im.py')
ls -la reports/v33a_combined_soft.exit 2>/dev/null || echo 'no exit yet'
