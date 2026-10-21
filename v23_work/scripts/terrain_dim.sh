#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'TERRAIN_HEIGHT_DIM|TERRAIN_ROWS|TERRAIN_COLS' source/go1_hier_sim/go1_hier_sim/contracts.py | head -8
echo '--- how terrain vector is built'
grep -an 'def height_geometry\|def _height_geometry' source/go1_hier_sim/go1_hier_sim/envs/runtime.py
n=$(grep -an 'def _height_geometry' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -d: -f1)
[ -n "$n" ] && sed -n "${n},$((n+30))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-130