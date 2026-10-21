#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an 'course_x' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -10
echo '--- context'
n=$(grep -an 'course_x' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n-14)),$((n+10))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo '--- start in tile usage'
grep -an 'START_IN_TILE\|spawn' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -10
