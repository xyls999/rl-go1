#!/usr/bin/env bash
cd /config/Desktop/sim
log=reports/v37_probe_retry.log
echo "RETRY_START $(date -u +%FT%TZ)" >> "$log"
for i in $(seq 1 6); do
  pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
  pkill -9 -f 'bin/torchrun' 2>/dev/null
  sleep 8
  rm -f reports/v37_probe_soft9.log
  timeout 600 bash scripts/v37_probe_soft9.sh > /dev/null 2>&1
  rc=$?
  comp=$(grep -ao '"completion_rate": [0-9.]*' reports/v37_probe_soft9.log 2>/dev/null | head -1)
  echo "TRY $i rc=$rc $comp $(date -u +%FT%TZ)" >> "$log"
  if [ -n "$comp" ]; then
    echo "DONE $(date -u +%FT%TZ)" >> "$log"
    exit 0
  fi
done
echo "GAVE_UP $(date -u +%FT%TZ)" >> "$log"
