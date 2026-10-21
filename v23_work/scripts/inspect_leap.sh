#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "725,800p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo === where actions get replaced ===
grep -n "_leap_mode\|applied_actions\|actions =\|_action\b" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | sed -n "1,30p"
echo === policy obs build for actor ===
grep -n "def policy_obs\|teacher_locomotion_obs\|def locomotion_obs" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head