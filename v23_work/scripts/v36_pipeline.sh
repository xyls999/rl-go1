#!/usr/bin/env bash
cd /config/Desktop/sim
./sim.sh test -q > reports/v36_fulltests.log 2>&1
echo FULL_RC=$? $(date -u +%FT%TZ) >> reports/v36_fulltests.log
tail -1 reports/v36_fulltests.log >> reports/v36_gate.log
if grep -aq "357 passed" reports/v36_fulltests.log || grep -aqE "[0-9]+ passed" reports/v36_fulltests.log; then
  sed -i "s/v33_probe_plan_soft6_16.json/v34_eval_plan_combined.json/" /dev/null 2>/dev/null
  nohup bash scripts/start_v36a_combined_soft8.sh > reports/v36a_run.log 2>&1 &
  echo V36A_LAUNCHED $(date -u +%FT%TZ) >> reports/v36_gate.log
else
  echo FULL_SUITE_FAILED >> reports/v36_gate.log
fi