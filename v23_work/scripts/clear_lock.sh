#!/usr/bin/env bash
cd /config/Desktop/sim
ps -eo pid,etimes,args | grep -aE "torchrun_sim|sim.py|healthprobe" | grep -av grep | head -6
pkill -f v33a_healthprobe || true
pkill -f torchrun_sim || true
pkill -f sim.py || true
sleep 6
ps aux | grep -ac "[s]im.py"
fuser reports/v7_sim.lock 2>/dev/null || echo LOCK_FREE