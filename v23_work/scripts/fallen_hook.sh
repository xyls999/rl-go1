#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an '_fallen =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6
n=$(grep -an 'self._fallen =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
echo "line=$n"
sed -n "$((n-12)),$((n+10))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | cut -c1-140
echo '--- write_root usage'
grep -an 'write_root_pose_to_sim\|write_joint_state_to_sim' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6