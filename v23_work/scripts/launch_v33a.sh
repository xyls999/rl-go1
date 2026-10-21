#!/usr/bin/env bash
cd /config/Desktop/sim
nohup bash scripts/start_v33a_combined_soft6.sh > reports/v33a_run.log 2>&1 &
sleep 60
grep -aE "PREFLIGHT|START_UTC|PLAN|FAMILIES|LOCK|drift|Error" reports/v33a_run.log | head -6
ps aux | grep -c "[s]im.py"