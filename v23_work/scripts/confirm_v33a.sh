#!/usr/bin/env bash
cd /config/Desktop/sim
sleep 150
echo ITER: $(grep -a "Learning iteration" v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log 2>/dev/null | tail -1 | tr -s " " | cut -c1-40)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -2
head -2 reports/v33a_run.log 2>/dev/null | cut -c1-130
tail -1 lp_combined_v33a_results.txt 2>/dev/null | cut -c1-130