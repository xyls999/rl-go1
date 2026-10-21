import json
from collections import Counter
from pathlib import Path

plan = json.loads(Path('v23_work/reports/lp_gate_plan_2026-10-18_go1-v26-final-eleven-512.json').read_text())
fam = plan['families']
rows = [json.loads(s) for s in Path('v23_work/reports/v26_final11_trace512.jsonl').read_text().splitlines()]
per = {}
for r in rows:
    f = fam[r['env_id']]
    d = per.setdefault(f, [0, 0, Counter()])
    d[0] += 1
    d[1] += int(r['success'])
    if not r['success']:
        c = 'fallen' if r['fallen'] else ('collision' if r['collision'] else (
            'timeout' if not any([r['out_of_bounds'], r['fell_into_gap']]) else 'other'))
        d[2][c] += 1
ok = 0
for f in sorted(per):
    tot, succ, causes = per[f]
    rate = succ / tot
    ok += rate >= 0.90
    print(f, round(rate, 3), f'{succ}/{tot}', dict(causes))
print('families_ok', ok, '/11 overall', round(sum(r['success'] for r in rows) / len(rows), 4),
      len(rows), 'coll', sum(r['collision'] for r in rows), 'fallen', sum(r['fallen'] for r in rows))
