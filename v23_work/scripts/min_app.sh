#!/usr/bin/env bash
cd /config/Desktop/sim
pkill -f v42_watch 2>/dev/null
pkill -9 -f 'env_isaaclab/bin/python.*sim[.]py' 2>/dev/null
pkill -9 -f 'bin/torchrun' 2>/dev/null
sleep 5
rm -rf /dev/shm/* 2>/dev/null
export CUDA_VISIBLE_DEVICES=1
cat > /tmp/min_app.py <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_OK", flush=True)
app.close()
print("CLOSED", flush=True)
PY
timeout 150 /root/env_isaaclab/bin/python /tmp/min_app.py > reports/min2.log 2>&1
echo "MIN_RC=$?"
grep -aE 'APP_OK|CLOSED|Error|Traceback' reports/min2.log | head -3
tail -2 reports/min2.log | cut -c1-120
