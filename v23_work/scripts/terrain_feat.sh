#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1120,1205p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140
echo '--- contracts'
grep -an 'TERRAIN_FEATURE_DIM\|LOCO_OBS_DIM' source/go1_hier_sim/go1_hier_sim/contracts.py | head -6