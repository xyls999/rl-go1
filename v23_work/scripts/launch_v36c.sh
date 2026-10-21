#!/usr/bin/env bash
cd /config/Desktop/sim
setsid nohup bash scripts/start_v36c_hard_finetune2.sh > reports/v36c_run.log 2>&1 &
sleep 50
grep -aE "START|ROSTER|Error" reports/v36c_run.log | head -3
ps aux | grep -ac "[s]im.py"