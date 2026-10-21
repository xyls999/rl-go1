#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- tests'
grep -aE 'FAILED|assert|Error' reports/v38_tests.log | head -12
echo '--- pin vs actual'
grep -a runtime.py scripts/start_v38_narrow_credit.sh | head -2
sha256sum source/go1_hier_sim/go1_hier_sim/envs/runtime.py