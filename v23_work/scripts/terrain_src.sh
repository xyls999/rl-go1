#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'terrain = |terrain,|_perception.terrain|_terrain_observable =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -12
echo '---'
sed -n '1178,1210p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140