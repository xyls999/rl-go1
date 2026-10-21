#!/usr/bin/env bash
cd /config/Desktop/sim
log=reports/v42_watch.log
echo "V42_WATCH_START $(date -u +%FT%TZ)" >> "$log"
for i in $(seq 1 8); do
  pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
  pkill -9 -f 'bin/torchrun' 2>/dev/null
  sleep 6
  rm -rf /dev/shm/* 2>/dev/null
  rm -f reports/v42_flat.log reports/v42_flat_trace.jsonl
  timeout 600 bash scripts/v42_flat.sh > /dev/null 2>&1
  rc=$?
  c=$(grep -ao '"completion_rate": [0-9.]*' reports/v42_flat.log 2>/dev/null | head -1)
  f=$(grep -ao '"fallen_rate": [0-9.]*' reports/v42_flat.log 2>/dev/null | head -1)
  t=$(grep -ao '"timeout_rate": [0-9.]*' reports/v42_flat.log 2>/dev/null | head -1)
  echo "TRY $i rc=$rc $c $f $t $(date -u +%FT%TZ)" >> "$log"
  if [ -n "$c" ]; then
    echo "V42_WATCH_DONE" >> "$log"
    exit 0
  fi
done
echo "V42_WATCH_GAVE_UP" >> "$log"
