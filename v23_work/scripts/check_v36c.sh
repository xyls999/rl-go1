#!/usr/bin/env bash
cd /config/Desktop/sim
grep -a "Learning iteration" v35_phased_2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1.log 2>/dev/null | tail -1
grep -a "GATE result" lp_combined_v36c_results.txt 2>/dev/null
cat reports/v36c_hard_finetune2.exit 2>/dev/null
ps aux | grep -ac "[s]im.py"
date -u +%FT%TZ