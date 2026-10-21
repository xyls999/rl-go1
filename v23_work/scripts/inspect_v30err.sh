#!/usr/bin/env bash
cd /config/Desktop/sim
ps aux | grep [s]im.py | wc -l
n=$(grep -an "Traceback\|Error\|SystemExit" v35_phased_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.log | head -1 | cut -d: -f1)
echo FIRST_ERR=$n
if [ -n "$n" ]; then sed -n "$((n)),$((n+28))p" v35_phased_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.log; else tail -25 v35_phased_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.log; fi