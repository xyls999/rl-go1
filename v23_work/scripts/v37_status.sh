#!/usr/bin/env bash
cd /config/Desktop/sim
cat reports/v37_probe.nohup 2>/dev/null
grep -ao '"completion_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1
grep -ao '"fallen_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1
grep -ao '"timeout_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1
grep -ao '"achieved_speed_mean": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1
