#!/usr/bin/env bash
cd /config/Desktop/sim
pkill -f v37_watch 2>/dev/null
pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
sleep 4
echo '--- probe script'
cat scripts/v37_probe_soft9.sh
echo '--- last log'
tail -20 reports/v37_probe_soft9.log | cut -c1-140