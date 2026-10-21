#!/usr/bin/env bash
cd /config/Desktop/sim
ps -eo pid,etimes,args | grep "[s]im.py" | head -4 | cut -c1-110
pkill -9 -f "scripts/sim.py"; pkill -9 -f torchrun_sim; pkill -9 -f v34_probe; sleep 6
echo AFTER_KILL: $(ps aux | grep -ac "[s]im.py")
setsid nohup bash scripts/v33a_supervisor.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo SUP: $(ps aux | grep -ac "[v]33a_supervisor")
tail -2 reports/v33a_supervisor.log 2>/dev/null