#!/usr/bin/env bash
cd /config/Desktop/sim
ps -eo pid,etimes,args | grep "[s]im.py" | head -2 | cut -c1-80
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -2
wc -c reports/v37_probe_soft9.log
date -u +%FT%TZ