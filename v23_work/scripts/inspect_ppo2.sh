#!/usr/bin/env bash
cd /config/Desktop/sim
echo === on_policy.py head ===
sed -n "1,140p" source/go1_hier_sim/go1_hier_sim/training/on_policy.py
echo === distillation runner act hooks ===
grep -nE 'def (collect|update|act|rollout)|teacher|hidden|reset|driver=' source/go1_hier_sim/go1_hier_sim/training/distillation.py | head -40