#!/usr/bin/env bash
cd /config/Desktop/sim
ls -la v35_phased_2026-10-19_go1-v33a* 2>/dev/null
f=$(ls -t v35_phased_2026-10-19_go1-v33a*log 2>/dev/null | head -1)
echo FILE=$f
tail -3 "$f" 2>/dev/null | cut -c1-120