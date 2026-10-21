#!/usr/bin/env bash
cd /config/Desktop/sim
f=$(grep -alE "stairs_narrow_8_mixed +0\.[0-9]{3} \(" lp_*.txt reports/*.log 2>/dev/null | head -1)
echo "file=$f"
if [ -n "$f" ]; then
  awk '/GATE result=/{last=$0} /   [a-z_]+ +0\.[0-9]{3} \(/{lines[NR]=$0} END{}' "$f" >/dev/null
  grep -anE "GATE result|   [a-z_]+ +0\.[0-9]{3} \(" "$f" | tail -22
fi
