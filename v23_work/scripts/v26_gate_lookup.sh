#!/usr/bin/env bash
cd /config/Desktop/sim
ls -t reports/evaluate_stage1_*gate*v26* 2>/dev/null | head -3
grep -a "GATE result" reports/*v26*gate*.txt 2>/dev/null | tail -2