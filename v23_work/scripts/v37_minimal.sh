#!/usr/bin/env bash
cd /config/Desktop/sim
for p in $(ps -eo pid,args | grep -E 'env_isaaclab/bin/python.*sim\.py' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 4
export CUDA_VISIBLE_DEVICES=1
timeout 180 /root/env_isaaclab/bin/python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('APP_OK', flush=True)
app.close()
print('CLOSED', flush=True)
" > reports/v37_minimal.log 2>&1
echo "MIN_RC=$?"
tail -3 reports/v37_minimal.log | cut -c1-150
