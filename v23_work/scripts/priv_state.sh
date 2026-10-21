#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- privileged_state'
n=$(grep -an 'def privileged_state' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -d: -f1)
echo "line=$n"
[ -n "$n" ] && sed -n "$((n)),$((n+40))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo '--- teacher obs'
grep -an 'teacher_locomotion_obs\|teacher_navigation_obs' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -5
