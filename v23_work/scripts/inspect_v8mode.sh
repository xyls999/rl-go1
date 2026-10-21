#!/usr/bin/env bash
cd /config/Desktop/sim
grep -n "_v8_gap_mode" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6
n=$(grep -n "_v8_gap_mode =" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
sed -n "$((n-8)),$((n+4))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
grep -n "_v8_gap_bounds =" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -3
m=$(grep -n "_v8_gap_bounds =" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
sed -n "$((m-4)),$((m+8))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py