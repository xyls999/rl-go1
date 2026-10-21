#!/usr/bin/env bash
cd /config/Desktop/sim
pkill -f v37_retry 2>/dev/null
pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
pkill -9 -f 'bin/torchrun' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
sleep 5
echo '--- shm before'; ls /dev/shm | head -5; df -h /dev/shm | tail -1
rm -rf /dev/shm/* 2>/dev/null
rm -rf /tmp/omni* /tmp/kit* /tmp/.kit* 2>/dev/null
rm -rf "$HOME/.nvidia-omniverse/logs"/* 2>/dev/null
rm -rf "$HOME/.cache/ov/kit" 2>/dev/null
echo '--- cleaned'
ls /dev/shm | wc -l
setsid nohup bash scripts/v37_probe_soft9.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo PROBE_PID_STARTED
ps -eo pid,args | grep -c '[s]im.py'
