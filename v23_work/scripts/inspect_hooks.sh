#!/usr/bin/env bash
cd /config/Desktop/sim
echo === policy_mask hooks ===
grep -n "policy_mask" source/go1_hier_sim/go1_hier_sim/cli.py | head
grep -n "policy_mask" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head
echo === runtime step signature ===
grep -n "def step\|def _step\|super().step" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head
echo === extras channel ===
grep -n "extras" source/go1_hier_sim/go1_hier_sim/cli.py | head
grep -n "extras" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8