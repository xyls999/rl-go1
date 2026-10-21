#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -an "Traceback" reports/v29_probe_warmup.log | head -1 | cut -d: -f1)
if [ -z "$n" ]; then grep -a "Error\|error" reports/v29_probe_warmup.log | tail -5; fi
sed -n "$((n+1)),$((n+30))p" reports/v29_probe_warmup.log