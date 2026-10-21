#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an '_v29_gate\|_policy_mask' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -12
echo '--- context'
n=$(grep -an 'def _v29_gate' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -d: -f1)
echo "gate def=$n"
m=$(grep -an '_v29_gate(' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | tail -1 | cut -d: -f1)
echo "call=$m"
sed -n "$((m-12)),$((m+8))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140