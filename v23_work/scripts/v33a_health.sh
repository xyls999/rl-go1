#!/usr/bin/env bash
cd /config/Desktop/sim
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -eo pid,etimes,time,args --sort=-time | grep "[s]im.py" | head -3
ls -la v35_phased_2026-10-19_go1-v33a-combined-soft6-s01-p1-d1.log
date -u +%FT%TZ