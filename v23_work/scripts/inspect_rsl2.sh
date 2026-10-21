#!/usr/bin/env bash
cd /config/Desktop/sim
python3 - <<'PY'
import inspect
from rsl_rl.algorithms import PPO
src = inspect.getsource(PPO)
i = src.find('privileged_actions')
while i != -1 and i < len(src):
    print('=== context ===')
    print(src[max(0,i-500):i+400])
    i = src.find('privileged_actions', i+1)
    if i > 20000: break
PY