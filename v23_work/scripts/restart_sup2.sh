#!/usr/bin/env bash
cd /config/Desktop/sim
setsid nohup bash scripts/v33a_supervisor.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo SUP: $(ps aux | grep -ac "[v]33a_supervisor")
tail -1 reports/v33a_supervisor.log