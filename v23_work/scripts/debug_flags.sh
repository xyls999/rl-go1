#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an 'GO1_DEBUG' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -12
echo '--- context around 1280-1360'
sed -n '1280,1365p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-130
