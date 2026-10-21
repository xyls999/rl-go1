#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -an "Traceback" v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log | head -1 | cut -d: -f1)
sed -n "$((n+40)),$((n+60))p" v35_phased_2026-10-19_go1-v29-student-distill-s01-p1-d1.log
echo === warmup freeze mechanism ===
grep -n "requires_grad\|actor_lr\|warmup" source/go1_hier_sim/go1_hier_sim/cli.py | sed -n "1,14p"