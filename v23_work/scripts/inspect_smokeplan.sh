#!/usr/bin/env bash
cd /config/Desktop/sim
echo '--- geom tests log'
tail -3 reports/v32_geom_tests.log 2>/dev/null
echo '--- smoke plan keys'
/root/env_isaaclab/bin/python - <<'PY'
import json
d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
print(list(d.keys()))
print({k: (v[:4] if isinstance(v, list) else v) for k, v in d.items()})
PY
