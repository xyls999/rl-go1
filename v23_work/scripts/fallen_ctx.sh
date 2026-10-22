#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1325,1375p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-150