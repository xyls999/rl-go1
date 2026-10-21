#!/usr/bin/env bash
cd /config/Desktop/sim
echo --- supervisor log
cat reports/v33a_supervisor.log 2>/dev/null | tail -8
echo --- procs
ps aux | grep -a "[s]im.py" | wc -l
ps aux | grep -ac "[v]33a_supervisor"
echo --- v33a artifacts
ls -la reports/v33a_run.log lp_combined_v33a_results.txt reports/v33a_combined_soft6.exit 2>/dev/null
tail -2 reports/v33a_run.log 2>/dev/null | cut -c1-140
echo --- gpu/health
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -4
uptime -s; date -u +%FT%TZ