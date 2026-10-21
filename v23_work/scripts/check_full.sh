#!/usr/bin/env bash
cd /config/Desktop/sim
tail -1 reports/v36_fulltests.log 2>/dev/null
if ! grep -aq " passed" reports/v36_fulltests.log 2>/dev/null; then
  echo "full suite still running or missing"
fi