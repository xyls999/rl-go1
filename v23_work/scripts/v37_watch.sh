#!/usr/bin/env bash
# Persistent retry: soft9 probe on GPU1, every ~9 min, up to 6 hours.
cd /config/Desktop/sim
log=reports/v37_probe_watch.log
echo "WATCH_START $(date -u +%FT%TZ)" >> "$log"
for i in $(seq 1 40); do
  pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
  pkill -9 -f 'bin/torchrun' 2>/dev/null
  sleep 5
  rm -rf /dev/shm/* 2>/dev/null
  rm -f reports/v37_probe_soft9.log
  timeout 540 bash scripts/v37_probe_soft9.sh > /dev/null 2>&1
  rc=$?
  comp=$(grep -ao '"completion_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1)
  fall=$(grep -ao '"fallen_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1)
  echo "TRY $i rc=$rc $comp $fall $(date -u +%FT%TZ)" >> "$log"
  if [ -n "$comp" ]; then
    echo "WATCH_DONE $(date -u +%FT%TZ)" >> "$log"
    exit 0
  fi
done
echo "WATCH_GAVE_UP $(date -u +%FT%TZ)" >> "$log"
