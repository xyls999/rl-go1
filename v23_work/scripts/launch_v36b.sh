#!/usr/bin/env bash
cd /config/Desktop/sim
setsid nohup bash scripts/start_v36b_hard_finetune.sh > reports/v36b_run.log 2>&1 &
sleep 50
grep -aE "PREFLIGHT|START|FAMILIES|PLAN|Error" reports/v36b_run.log | head -4
ps aux | grep -ac "[s]im.py"