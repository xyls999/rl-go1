#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- course_gaps branch start (opening / lead-in)'
n=$(grep -an 'elif kind in ("course_gaps"' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -1 | cut -d: -f1)
[ -z "$n" ] && n=$(grep -an 'course_gaps' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -1 | cut -d: -f1)
echo "anchor=$n"
sed -n '795,845p' source/go1_hier_sim/go1_hier_sim/envs/geometry.py
echo '--- opening definition'
grep -an 'opening' source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -12
