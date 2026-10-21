#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'TERRAIN_FEATURE_DIM|LOCO_OBS_DIM|DEPTH' source/go1_hier_sim/go1_hier_sim/contracts.py | head -12
echo '--- scanner cfg'
grep -anE 'height_scanner|pattern_cfg|size=|resolution=' source/go1_hier_sim/go1_hier_sim/envs/config.py | head -12