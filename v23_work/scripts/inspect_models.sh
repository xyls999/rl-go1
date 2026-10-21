#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "55,140p" source/go1_hier_sim/go1_hier_sim/models.py
echo === actor critic forward ===
sed -n "131,210p" source/go1_hier_sim/go1_hier_sim/models.py