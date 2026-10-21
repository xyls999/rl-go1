#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "Traceback" reports/v29_smoke4x16.log | head -3
n=$(grep -an "Traceback" reports/v29_smoke4x16.log | head -1 | cut -d: -f1)
sed -n "${n},$((n+25))p" reports/v29_smoke4x16.log