#!/usr/bin/env bash
cd /config/Desktop/sim
n=$(grep -n "def gap_leap_trigger" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
sed -n "$((n)),$((n+55))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py