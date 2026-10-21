#!/usr/bin/env bash
cd /config/Desktop/sim
log=reports/v41_watch.log
echo "V41_WATCH_START $(date -u +%FT%TZ)" >> "$log"
for i in $(seq 1 5); do
  pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
  pkill -9 -f 'bin/torchrun' 2>/dev/null
  sleep 6
  rm -rf /dev/shm/* 2>/dev/null
  rm -f reports/v41_dense_trace.jsonl reports/v41_trace.log
  timeout 600 bash scripts/v41_dense_trace.sh > /dev/null 2>&1
  rc=$?
  if [ -s reports/v41_dense_trace.jsonl ]; then
    echo "TRY $i rc=$rc TRACE_OK $(date -u +%FT%TZ)" >> "$log"
    /root/env_isaaclab/bin/python scripts/v41_trace_fields.py >> "$log" 2>&1
    echo "V41_WATCH_DONE" >> "$log"
    exit 0
  fi
  echo "TRY $i rc=$rc no-trace $(date -u +%FT%TZ)" >> "$log"
done
echo "V41_WATCH_GAVE_UP" >> "$log"
