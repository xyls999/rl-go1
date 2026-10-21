#!/usr/bin/env bash
cd /config/Desktop/sim
grep -a "GATE result" lp_*v26*.txt reports/*v26* 2>/dev/null | tail -3
echo '--- per family lines'
grep -aE "^\[.*v26.*\]   [a-z_]+ " lp_*v26*.txt 2>/dev/null | tail -12