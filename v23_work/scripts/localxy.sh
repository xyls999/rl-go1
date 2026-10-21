#!/usr/bin/env bash
cd /config/Desktop/sim
ps aux | grep -ac '[s]im.py'
tail -2 reports/v41_trace.log | cut -c1-120
echo '--- local_xy definition'
grep -an 'local_xy =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -5
n=$(grep -an 'local_xy =' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n-6)),$((n+4))p" source/go1_hier_sim/go1_hier_sim/envs/runtime.py
echo '--- course origin'
grep -an 'env_origin\|scene.env_origins' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -6