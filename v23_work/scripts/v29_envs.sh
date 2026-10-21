#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'GO1_V29' source/go1_hier_sim/go1_hier_sim/cli.py source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -16
echo '--- teachers json'
cat reports/teachers_v29.json