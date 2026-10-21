#!/usr/bin/env bash
cd /config/Desktop/sim
grep -a "geometry.py" scripts/start_v38_narrow_credit.sh | head -2
sha256sum source/go1_hier_sim/go1_hier_sim/envs/geometry.py source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py