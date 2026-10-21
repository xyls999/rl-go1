#!/usr/bin/env bash
cd /config/Desktop/sim
ls source/go1_hier_sim/go1_hier_sim/ | head -20
echo '--- policy classes'
grep -anE 'class .*Policy|def build_policy|map_attn_v3' source/go1_hier_sim/go1_hier_sim/*.py | head -14
echo '--- on_policy'
grep -anE 'class |def ' source/go1_hier_sim/go1_hier_sim/on_policy.py 2>/dev/null | head -20