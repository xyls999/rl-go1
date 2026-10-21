#!/usr/bin/env bash
cd /config/Desktop/sim
date -u +%FT%TZ
ls -la --time-style=full-iso reports/v37_probe_soft9.log | cut -c30-110
ps -eo pid,etimes,args | grep '[s]im.py' | cut -c1-120
ps -eo pid,etimes,args | grep '[i]saaclab.sh' | cut -c1-100
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
tail -3 reports/v37_probe_soft9.log | cut -c1-140
