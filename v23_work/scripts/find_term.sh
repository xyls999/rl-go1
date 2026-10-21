#!/usr/bin/env bash
cd /config/Desktop/sim
grep -rn "fallen" source/go1_hier_sim/go1_hier_sim/envs/termination.py 2>/dev/null | head -6
ls source/go1_hier_sim/go1_hier_sim/envs/ | grep -a term
grep -rn "FallTermination\|fallen.*term\|term.*fallen" source/go1_hier_sim/go1_hier_sim/envs/*.py 2>/dev/null | grep -av ".bak" | head -12