#!/usr/bin/env bash
cd /config/Desktop/sim
echo --- v23 family env
grep -an "GO1_V23_TIRE_BRAKE_FAMILIES\|_v23_tire_mask =\|_v23_tire_mask=" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6
n=$(grep -n "_v23_tire_mask = " source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
sed -n "$((n-10)),$((n+6))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo --- TERRAIN_FAMILIES consumers
grep -arln "TERRAIN_FAMILIES" source scripts | head -12
echo --- ten-family roster source
grep -an "TERRAIN_FAMILIES\|families = \|FAMILIES =" scripts/run_student_v30.py | head -8