#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1200,1240p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo '--- any flag for actor using teacher obs'
grep -an 'teacher_locomotion_obs\|teacher_navigation_obs\|TEACHER_OBS\|teacher_actor' source/go1_hier_sim/go1_hier_sim/envs/runtime.py source/go1_hier_sim/go1_hier_sim/cli.py | head -12
