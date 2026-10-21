#!/usr/bin/env bash
cd /config/Desktop/sim
grep -a "Learning iteration" v35_phased_2026-10-19_go1-v36b-hard-finetune-s01-p1-d1.log 2>/dev/null | tail -1
grep -a "GATE result" lp_combined_v36b_results.txt 2>/dev/null | tail -1
cat reports/v36b_hard_finetune.exit 2>/dev/null
ps aux | grep -ac "[s]im.py"
date -u +%FT%TZ