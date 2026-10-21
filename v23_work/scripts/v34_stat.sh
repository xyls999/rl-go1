#!/usr/bin/env bash
cd /config/Desktop/sim
wc -l reports/v34_probe_flat.log
tail -3 reports/v34_probe_flat.log | cut -c1-150
ps -eo pid,etimes,time,args | grep "[s]im.py" | head -2 | cut -c1-100
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1
date -u +%FT%TZ