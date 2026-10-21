#!/usr/bin/env bash
cd /config/Desktop/sim
ls reports/lp_gate_plan_*v28*.json 2>/dev/null
/root/env_isaaclab/bin/python - <<'PY'
import json, glob
p = sorted(glob.glob('/config/Desktop/sim/reports/lp_gate_plan_*v28*.json'))
if not p:
    print('no v28 gate plan'); raise SystemExit
d = json.load(open(p[0]))
fams = d['families']
from collections import Counter
print('plan', p[0])
print('instances', len(fams), 'families', sorted(Counter(fams).items()))
print('difficulty max', max(d['difficulty']), 'speed levels', sorted(set(d.get('speed_level', []))))
PY