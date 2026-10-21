#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -n "_v19_prelaunch_state is not None" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
echo AT=$n
sed -n "$((n-40)),$((n+22))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py