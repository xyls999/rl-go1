#!/usr/bin/env bash
cd /config/Desktop/sim
/usr/bin/python3 --version 2>&1 || echo NO_SYSTEM_PY
/root/env_isaaclab/bin/python -c "import inspect; from rsl_rl.algorithms import PPO; src=inspect.getsource(PPO.update); print(src[:2600])" 2>&1 | head -60