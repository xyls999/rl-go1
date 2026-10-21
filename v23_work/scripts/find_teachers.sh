#!/usr/bin/env bash
cd /config/Desktop/sim
ls reports/ | grep -ai teachers
echo '--- v29 launcher'
ls scripts/ | grep -ai v29
echo '--- example json'
f=$(ls reports/*teachers*.json 2>/dev/null | head -1); echo "f=$f"; [ -n "$f" ] && cat "$f"