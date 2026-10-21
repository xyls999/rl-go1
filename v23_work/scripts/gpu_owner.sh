#!/usr/bin/env bash
echo '--- fuser on nvidia devices'
for d in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl; do
  [ -e "$d" ] && echo "$d: $(sudo -n fuser -v "$d" 2>&1 | tr '\n' ' ')"
done
echo '--- ps of those pids'
sudo -n fuser /dev/nvidiactl 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | while read p; do
  ps -o pid,ppid,etimes,stat,args -p "$p" 2>/dev/null | tail -1 | cut -c1-150
done
echo '--- nvidia-smi apps again'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo '--- gpu0'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
