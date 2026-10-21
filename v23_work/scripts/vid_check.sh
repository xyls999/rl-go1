#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'video|--frames|render' source/go1_hier_sim/go1_hier_sim/cli.py | head -20
echo '--- sim.sh subcommands'
grep -anE '^  [a-z_-]+\)|usage|case ' sim.sh | head -20