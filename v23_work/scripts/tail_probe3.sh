#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n "279,320p" reports/v33a_health.log | cut -c1-170