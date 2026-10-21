#!/usr/bin/env bash
cd /config/Desktop/sim
head -1 reports/v32a_smoke.exit
nohup bash scripts/start_v32a_combined_soft.sh > reports/v32a_run.log 2>&1 &
sleep 90
grep -aE "PREFLIGHT|V32A_START|PLAN|FAMILIES|ISAAC_LOCK|drift|Error" reports/v32a_run.log | head -8
ps aux | grep -c [s]im.py