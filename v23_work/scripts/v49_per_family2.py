import json
from collections import defaultdict

plan = json.load(open('/config/Desktop/sim/reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json'))
fams = plan['families']
d = json.load(open('/config/Desktop/sim/reports/evaluate_stage1_2026-10-21_10-50-47_v49-v26-11family.json'))
envs = d['environments']
per = defaultdict(lambda: [0, 0])
for k, e in envs.items():
    eid = int(k)
    fam = fams[eid] if eid < len(fams) else 'unknown'
    n = int(e['episodes'])
    s = int(e['success'])
    per[fam][0] += n
    per[fam][1] += s
ok = 0
for fam, (n, s) in sorted(per.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0])):
    rate = s / n if n else 0
    ok += rate >= 0.9
    print(f'{fam:26s} n={n:4d} completion={rate:.4f}')
print('families_ok', ok, '/', len(per), 'overall', round(d.get('completion_rate', 0), 4))
