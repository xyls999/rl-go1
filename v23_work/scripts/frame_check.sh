#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- forward_axis'
grep -an '_forward_axis =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -5
n=$(grep -an '_forward_axis =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n-10)),$((n+6))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo '--- rotation / serpentine in geometry'
grep -an 'rotat\|yaw\|serpentine\|flip' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -12
echo '--- origins assembly'
n2=$(grep -an 'origins' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | grep -a 'def build_scene' | cut -d: -f1)
sed -n '1255,1305p' source/go1_hier_sim/go1_hier_sim/envs/geometry.py
