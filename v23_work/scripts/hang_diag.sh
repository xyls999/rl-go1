#!/usr/bin/env bash
cd /config/Desktop/sim
echo '=== kit log tail (last attempt)'
f=$(ls -t /root/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log | head -1)
echo "file=$f lines=$(wc -l < $f)"
tail -6 "$f" | cut -c1-150
echo
echo '=== gpu health'
nvidia-smi -q 2>/dev/null | grep -iE 'GPU Lost|Xid|Uncorrectable|Row Remap|Pending|Retired|Remapped' | head -10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,pstate --format=csv,noheader
echo
echo '=== D-state processes'
ps -eo pid,stat,wchan:25,args | awk '$2 ~ /D/' | cut -c1-130 | head -8
echo
echo '=== nvidia device holders'
for d in /dev/nvidia0 /dev/nvidia1; do sudo -n fuser -v $d 2>&1 | tail -2; done
echo
echo '=== container limits'
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
free -g | head -2
echo
echo '=== recent dmesg via journal (may be empty)'
dmesg 2>/dev/null | tail -5
