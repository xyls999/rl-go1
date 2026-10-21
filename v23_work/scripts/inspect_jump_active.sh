#!/usr/bin/env bash
cd /config/Desktop/sim
grep -n "_jump_active" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8
n=$(grep -n "self._jump_active =" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
sed -n "$((n-6)),$((n+6))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py