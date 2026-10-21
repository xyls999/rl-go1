#!/usr/bin/env bash
cd /config/Desktop/sim
echo === storage internals ===
python3 - <<'PY'
import inspect, rsl_rl
from rsl_rl.storage import RolloutStorage
print(RolloutStorage.__init__.__doc__ or inspect.getsource(RolloutStorage.__init__))
src = inspect.getsource(RolloutStorage)
import re
for pat in ('def add(', 'def clear(', 'self.observations', 'recurrent_hidden', 'def actor_obs', 'actions', 'def update_statistics'):
    idx = src.find(pat)
    print('---', pat, idx)
PY
echo === rsl_rl version ===
python3 -c "import rsl_rl; print(rsl_rl.__version__ if hasattr(rsl_rl,'__version__') else 'n/a')"
pip show rsl_rl 2>/dev/null | head -2
echo === policy act in OnPolicyRunner.learn ===
python3 - <<'PY'
import inspect
from rsl_rl.runners import OnPolicyRunner
s = inspect.getsource(OnPolicyRunner.learn)
print(s[:1800])
PY