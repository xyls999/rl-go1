#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "1,40p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo === jump policy attrs ===
grep -n "_jump_policy\|_base_policy\|_jump_hidden\|_base_hidden" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -12
echo === v23 init anchor ===
sed -n "415,452p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo === reset block ===
sed -n "2368,2380p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py