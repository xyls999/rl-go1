#!/usr/bin/env bash
cd /config/Desktop/sim
./sim.sh test -q -k v36 > reports/v36_ktests.log 2>&1
echo RC=$?
tail -2 reports/v36_ktests.log