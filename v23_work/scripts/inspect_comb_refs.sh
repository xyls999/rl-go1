#!/usr/bin/env bash
cd /config/Desktop/sim
echo --- runtime combined refs
grep -an "combined" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -20
echo --- toml combined
grep -an "combined" configs/rewards_stage1_gaps_v11_unified.toml | head -8
echo --- geometry families copy
sed -n "1235,1250p" source/go1_hier_sim/go1_hier_sim/envs/geometry.py
echo --- terrain_names in runtime
grep -an "terrain_names" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -5