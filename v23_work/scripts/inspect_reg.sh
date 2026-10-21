#!/usr/bin/env bash
cd /config/Desktop/sim
grep -arn "TERRAIN_FAMILIES" source/go1_hier_sim/go1_hier_sim/ | head -10
echo ---- roster-file
cat source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v30.py
echo ---- run_student usage
grep -anE "roster|families|combined" scripts/run_student_v30.py | head -25