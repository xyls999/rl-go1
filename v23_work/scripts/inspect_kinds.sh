#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "combined" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | grep -aE "KIND|FAMIL|assert|raise|in \(|== .combined" | head -12
grep -an "combined" source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v30.py | head -8
grep -anE "VALID|FAMILY|allowed" source/go1_hier_sim/go1_hier_sim/envs/geometry.py | head -8