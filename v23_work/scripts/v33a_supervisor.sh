#!/usr/bin/env bash
# v33a supervisor: clear stuck Isaac procs, probe until healthy, then launch long run
cd /config/Desktop/sim
log=reports/v33a_supervisor.log
echo "SUP_START $(date -u +%FT%TZ)" >> "$log"
for i in $(seq 1 72); do
  pkill -f v34_probe.sh 2>/dev/null
  pkill -f torchrun_sim 2>/dev/null
  pkill -f "scripts/sim.py" 2>/dev/null
  sleep 10
  out=$(bash scripts/v33a_healthprobe.sh 2>&1 | tail -2)
  echo "PROBE $i $(date -u +%FT%TZ) :: $out" >> "$log"
  if echo "$out" | grep -aq "PROBE_EXIT=0"; then
    pkill -f torchrun_sim 2>/dev/null; sleep 5
    echo "HEALTHY -> LAUNCH V33A $(date -u +%FT%TZ)" >> "$log"
    nohup bash scripts/start_v33a_combined_soft6.sh > reports/v33a_run.log 2>&1 &
    echo "LAUNCHED pid=$!" >> "$log"
    exit 0
  fi
  pkill -f torchrun_sim 2>/dev/null
  pkill -f "scripts/sim.py" 2>/dev/null
  sleep 280
done
echo "SUPERVISOR GAVE UP after 72 probes $(date -u +%FT%TZ)" >> "$log"