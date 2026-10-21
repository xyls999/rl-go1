#!/usr/bin/env bash
cd /config/Desktop/sim
grep -an "Mean" reports/v29_smoke4x16.log | tail -14
echo === distill traces ===
grep -ac "distill" reports/v29_smoke4x16.log || true
grep -a "distill_loss" reports/v29_smoke4x16.log | tail -2 || true