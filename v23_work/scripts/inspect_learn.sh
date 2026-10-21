#!/usr/bin/env bash
cd /config/Desktop/sim
/root/env_isaaclab/bin/python - <<'PY'
import inspect, re
from rsl_rl.runners import OnPolicyRunner
s = inspect.getsource(OnPolicyRunner.learn)
i = s.find('update()')
print(s[i-400:i+900])
PY