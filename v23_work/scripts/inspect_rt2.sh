#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "75,90p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo ----; sed -n "383,395p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo ----; sed -n "420,436p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo ----; sed -n "493,512p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py