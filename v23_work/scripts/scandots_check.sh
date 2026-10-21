#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an 'TERRAIN_FEATURE_DIM' source/go1_hier_sim/go1_hier_sim/envs/*.py | head -6
n=$(grep -an 'TERRAIN_FEATURE_DIM *=' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -1 | cut -d: -f1)
echo "geom dim line=$n"
grep -anE 'def .*terrain.*feature|terrain_feature|def _terrain' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -8
echo '--- runtime terrain feature build'
grep -an 'terrain_feature\|_terrain_exact' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8