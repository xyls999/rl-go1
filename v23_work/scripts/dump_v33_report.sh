#!/usr/bin/env bash
cd /config/Desktop/sim
/root/env_isaaclab/bin/python - <<PYEOF
import glob, json
path = glob.glob('/config/Desktop/sim/reports/evaluate_stage1_*v33-probe-v19p4-soft4.json')[0]
d = json.load(open(path))
top = {k: v for k, v in d.items() if not isinstance(v, dict)}
print('TOP:', json.dumps(top, indent=1)[:600])
envs = d['environments']
ks = list(envs)[:3]
for k in ks:
    print('ENV', k, json.dumps(envs[k])[:220])
PYEOF