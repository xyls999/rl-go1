#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- actor observation assembly'
grep -an 'def actor_observation\|def critic_observation\|privileged' source/go1_hier_sim/go1_hier_sim/envs/config.py | head -8
n=$(grep -an 'def actor_observation' source/go1_hier_sim/go1_hier_sim/envs/config.py | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n)),$((n+18))p" source/go1_hier_sim/go1_hier_sim/envs/config.py
echo '--- obs groups'
grep -an 'class .*ObservationCfg\|ObservationTermCfg' source/go1_hier_sim/go1_hier_sim/envs/config.py | head -8
echo '--- teacher selection infra'
ls source/go1_hier_sim/go1_hier_sim/envs/ | grep -aE 'teacher|distill|roster'
