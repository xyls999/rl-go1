#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1,30p' configs/rewards_stage1_gaps_v11_unified.toml
echo '--- speed error'
sed -n '255,270p' configs/rewards_stage1_gaps_v11_unified.toml