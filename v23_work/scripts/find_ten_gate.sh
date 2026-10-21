#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- results files naming v26'
ls lp_*v26* 2>/dev/null
echo '--- any file with per-family stairs lines'
grep -alE "stairs_mixed +0\.[0-9]+ \(" lp_*.txt reports/*.log 2>/dev/null | head -6
echo '--- show one'
f=$(grep -alE "stairs_mixed +0\.[0-9]+ \(" lp_*.txt reports/*.log 2>/dev/null | head -1)
echo "file=$f"
[ -n "$f" ] && grep -aE "GATE result|   [a-z_]+ +0\.[0-9]{3} \(" "$f" | tail -14
