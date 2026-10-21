#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE "Error|error|Traceback|raise|assert|Value|Key" reports/v32a_smoke.log | head -20
echo ----
n=$(grep -an "Traceback" reports/v32a_smoke.log | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n)),$((n+25))p" reports/v32a_smoke.log
