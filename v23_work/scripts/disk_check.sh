#!/usr/bin/env bash
echo '--- disk'
df -h / /tmp /config /root /dev/shm 2>/dev/null
echo '--- inodes'
df -i / /config 2>/dev/null
echo '--- dmesg tail'
dmesg 2>/dev/null | tail -12
echo '--- nvidia procs'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo '--- sim proc cpu'
ps -eo pid,etimes,time,stat,args --sort=-time | grep '[s]im.py' | head -2 | cut -c1-90
