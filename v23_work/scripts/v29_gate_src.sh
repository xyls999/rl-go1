#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '770,800p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-150