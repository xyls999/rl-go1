#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an 'up_velocity' source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py | head -8
echo '--- step head'
sed -n '1,40p' source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py
echo '--- conditions'
grep -anE 'min_up|launch|upright|command_speed|contacts' source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py | head -14
