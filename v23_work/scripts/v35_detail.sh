#!/usr/bin/env bash
cd /config/Desktop/sim
for tag in v33a-on-soft6 v26-on-flat v26-on-combined; do
  echo "=== $tag"
  wc -l reports/v35_trace_${tag}.jsonl 2>/dev/null
  grep -aoE '"completion_rate": [0-9.]+|"fallen_rate": [0-9.]+|"episodes": [0-9]+|REPORT: [^ ]+' reports/v35_${tag}.log 2>/dev/null | head -4
  grep -aE "ValueError|Error|not in|Unknown terrain" reports/v35_${tag}.log 2>/dev/null | head -2 | cut -c1-150
done