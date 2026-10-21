import json
from collections import defaultdict

plan = json.load(open('/config/Desktop/sim/reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json'))
fams = plan['families']
rows = [json.loads(l) for l in open('/config/Desktop/sim/reports/v49_11fam_trace.jsonl')]
print('trace rows', len(rows), 'plan slots', len(fams))
per = defaultdict(lambda: [0, 0, 0])
for r in rows:
    eid = r.get('env_id')
    fam = fams[eid] if eid is not None and eid < len(fams) else 'unknown'
    per[fam][0] += 1
    per[fam][1] += 1 if r.get('success') else 0
    per[fam][2] += 1 if (r.get('fallen') or r.get('fell_into_gap')) else 0
ok = 0
for fam, (n, s, f) in sorted(per.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0])):
    rate = s / n if n else 0
    ok += rate >= 0.9
    print(f'{fam:26s} n={n:3d} completion={rate:.3f} fallen={f / n:.3f}')
print('families_ok', ok, '/', len(per))
