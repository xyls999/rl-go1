#!/usr/bin/env bash
cd /config/Desktop/sim
echo === gate_action callers ===
grep -rn gate_action source/go1_hier_sim/go1_hier_sim/*.py source/go1_hier_sim/go1_hier_sim/envs/*.py | head
echo === extras consumption in cli train ===
sed -n "495,515p" source/go1_hier_sim/go1_hier_sim/cli.py
echo === hier build with families ===
grep -n 'task_plan\|families=' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8
grep -n 'families' source/go1_hier_sim/go1_hier_sim/cli.py | head -6