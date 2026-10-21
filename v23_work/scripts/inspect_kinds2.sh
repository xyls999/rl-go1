#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "172,215p" source/go1_hier_sim/go1_hier_sim/envs/geometry.py
echo ---- ; sed -n "1305,1330p" source/go1_hier_sim/go1_hier_sim/envs/geometry.py
echo ---- registry
grep -arn "TERRAIN_FAMILIES\|FAMILIES =" source/go1_hier_sim/go1_hier_sim/envs/geometry.py source/go1_hier_sim/go1_hier_sim/envs/terrain.py 2>/dev/null | head -6