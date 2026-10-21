#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "retry\|no room" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -10
n=$(grep -an "no room" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -1 | cut -d: -f1)
sed -n "$((n-30)),$((n+4))p" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -20
echo ---- build_scene-loop
grep -an "def build_scene" source/go1_hier_sim/go1_hier_sim/envs/geometry.py
m=$(grep -an "for attempt\|max_attempt\|attempts" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -4)
echo "$m"