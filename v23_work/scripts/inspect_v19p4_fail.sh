#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "Error\|Traceback\|raise\|assert" reports/v33_probe_soft4_v19p4.log 2>/dev/null | head -6
n=$(grep -an "Traceback" reports/v33_probe_soft4_v19p4.log 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "$n" ]; then sed -n "${n},$((n+12))p" reports/v33_probe_soft4_v19p4.log; else tail -15 reports/v33_probe_soft4_v19p4.log; fi