#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1,60p' source/go1_hier_sim/go1_hier_sim/envs/teacher_distill_v29.py | cut -c1-140
echo '--- teachers json'
cat reports/teachers_v29.json 2>/dev/null | head -40
echo '--- how segments are read'
grep -anE 'segments|family_teacher_keys' source/go1_hier_sim/go1_hier_sim/envs/teacher_distill_v29.py | head -12