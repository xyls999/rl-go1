#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'track_linear_velocity|speed_error|track_angular|forward|base_height' configs/rewards_stage1_gaps_v11_unified.toml | head -20