#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -an "Traceback" v35_phased_2026-10-19_go1-v53-ladder-s01-p1-d1.log | head -1 | cut -d: -f1)
echo "line=$n"
[ -n "$n" ] && sed -n "${n},$((n+16))p" v35_phased_2026-10-19_go1-v53-ladder-s01-p1-d1.log | cut -c1-170