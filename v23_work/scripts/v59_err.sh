#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -an "Traceback" reports/v59_smoke.log | head -1 | cut -d: -f1)
echo "line=$n"
[ -n "$n" ] && sed -n "${n},$((n+14))p" reports/v59_smoke.log | cut -c1-160