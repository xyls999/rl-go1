#!/usr/bin/env bash
cd /config/Desktop/sim
grep -rn "def.*terminat\|TerminationTermCfg\|termination" source/go1_hier_sim/go1_hier_sim/envs/config.py | grep -av bak | head -10
grep -an "def.*termination\|def _termination\|fallen" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | grep -aiE "def|term_group|group.*fall" | head -12
grep -arn "def fallen\|def _check_fall\|flat_terrain\|base_height" source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head -8