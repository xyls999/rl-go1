#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "Traceback" reports/v33a_health.log | head -2
n=$(grep -an "Traceback" reports/v33a_health.log | head -1 | cut -d: -f1)
if [ -n "$n" ]; then sed -n "${n},$((n+18))p" reports/v33a_health.log | cut -c1-160; fi