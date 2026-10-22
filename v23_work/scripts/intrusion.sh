#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an 'gap_intrusion' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8
n=$(grep -an 'self._gap_intrusion' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
echo "line=$n"
sed -n "$((n-8)),$((n+16))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140