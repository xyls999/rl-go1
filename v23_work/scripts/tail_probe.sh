#!/usr/bin/env bash
cd /config/Desktop/sim
tail -6 reports/v33a_health.log | cut -c1-160
echo ----
grep -aE "Error|error|Traceback|raise" reports/v33a_health.log | tail -4 | cut -c1-160