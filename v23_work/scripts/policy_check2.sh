#!/usr/bin/env bash
cd /config/Desktop/sim
grep -rln 'class HierActorCriticV3' source/go1_hier_sim | head -3
grep -rln 'class CheckedPPO\|def recurrent' source/go1_hier_sim | head -6
echo '--- files'
ls source/go1_hier_sim/go1_hier_sim/hier_v2 2>/dev/null | head