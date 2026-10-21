import json
from collections import defaultdict
import numpy as np

plan = json.load(open('/config/Desktop/sim/reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json'))
fams = plan['families']
rows = [json.loads(l) for l in open('/config/Desktop/sim/reports/v49_11fam_trace.jsonl')]
per = defaultdict(lambda: {'sp': [], 'succ': 0, 'n': 0, 'st': []})
for r in rows:
    eid = r.get('env_id')
    if eid is None or eid >= len(fams):
        continue
    fam = fams[eid]
    d = per[fam]
    d['n'] += 1
    d['succ'] += 1 if r.get('success') else 0
    d['sp'].append(float(r.get('achieved_speed', 0)))
    d['st'].append(float(r.get('steps', 0)))
for fam in ('combined', 'gaps_short', 'tire_field', 'gaps_long'):
    if fam not in per:
        continue
    d = per[fam]
    sp = np.array(d['sp'])
    print(f"{fam:12s} n={d['n']:3d} succ={d['succ']:3d} "
          f"speed med={np.median(sp):.2f} p90={np.quantile(sp, .9):.2f} "
          f"steps med={np.median(d['st']):.0f}")
