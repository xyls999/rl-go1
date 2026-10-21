#!/usr/bin/env bash
for pid in 50493 8467; do
  echo "--- $pid"
  ps -o pid,ppid,etimes,stat,args -p $pid 2>/dev/null | cut -c1-160
done
echo '--- all kit/python'
ps -eo pid,ppid,etimes,stat,args | grep -E '[k]it|[p]ython|[t]orch' | grep -vE 'grep|v37_' | cut -c1-140 | head -12
echo '--- drive'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
