#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- Xid / NVRM in dmesg (last 20)'
dmesg 2>/dev/null | grep -aE 'Xid|NVRM' | tail -20
echo '--- gpu err counters'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader
echo '--- kubelet/docker? memory'
free -g | head -2
echo '--- nfs mount'
mount | grep -a '/config' | head -3
