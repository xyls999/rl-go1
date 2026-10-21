#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- sim.sh subcommands'
grep -aE 'eval|gate|trace' sim.sh | head -14
echo '--- v28 gate invocation in its launcher'
grep -aE 'sim.py.*(eval|gate|trace)' scripts/start_v28* scripts/start_v26* 2>/dev/null | head -6
echo '--- recent gate reports'
ls -t reports/ | grep -iE 'gate' | head -8
