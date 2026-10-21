#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '55,120p' source/go1_hier_sim/go1_hier_sim/envs/teacher_distill_v29.py | cut -c1-140
echo '--- runtime mapping'
grep -an 'combined_segment_teacher\|_v29_names\|family_teacher_keys' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6
sed -n '760,800p' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140