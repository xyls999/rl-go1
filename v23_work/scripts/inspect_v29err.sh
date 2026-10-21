#!/usr/bin/env bash
cd /config/Desktop/sim
tail -3 reports/v29_driver.log
n=$(grep -an "Traceback" v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log | head -1 | cut -d: -f1)
echo TB_LINE=$n
if [ -n "$n" ]; then sed -n "${n},$((n+22))p" v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log; else tail -30 v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log; fi