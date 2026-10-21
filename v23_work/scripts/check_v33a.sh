#!/usr/bin/env bash
cd /config/Desktop/sim
echo ITER: $(grep -a "Learning iteration" v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log 2>/dev/null | tail -1 | tr -s " " | cut -c1-38)
grep -aE "MEAN episode reward|mean_episode_return" v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log 2>/dev/null | tail -3 | cut -c1-110
grep -a GATE lp_combined_v33a_results.txt 2>/dev/null | tail -2 | cut -c1-160
cat reports/v33a_combined_soft6.exit 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1
ps aux | grep -ac "[s]im.py"
date -u +%FT%TZ