#!/usr/bin/env bash
cd /config/Desktop/sim
pkill -f run_student_v32 || true
pkill -f sim.py || true
sleep 8
ps aux | grep -c "[s]im.py"
rm -f lp_combined_v33a_state.json lp_combined_v33a_acrl_state.json lp_combined_v33a_results.txt reports/v33a_combined_soft6.exit reports/lp_plan_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.json v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log
rm -rf checkpoints/stage1/*go1-v33a-combined-soft6*
rm -f reports/v33a_run.log
echo CLEANED; date -u +%FT%TZ
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -2