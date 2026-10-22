#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1880,1925p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140
echo '--- toml gap_intrusion'
grep -an -A4 '^\[gap_intrusion\]' configs/rewards_stage1_gaps_v11_unified.toml