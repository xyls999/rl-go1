#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '535,570p' source/go1_hier_sim/go1_hier_sim/envs/config.py | cut -c1-140
echo '--- runtime scanner refs'
grep -anE 'height_scanner|gap_edge_scanner|_TERRAIN_DIM' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -14