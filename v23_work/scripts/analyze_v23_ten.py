import json
from collections import Counter
from pathlib import Path

plan = json.loads(Path('v20_work/reports/lp_gate_plan_2026-10-17_go1-v20-ten-student-s01-p1-d1.json').read_text())
fam = plan['families']
rows = [json.loads(s) for s in Path('v23_work/reports/v23_ten_gate_trace512.jsonl').read_text().splitlines()]
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
overall = sum(r['success'] for r in rows) / len(rows)
print('families_ok', ok, '/10 overall', round(overall, 4), len(rows),
      'coll', sum(r['collision'] for r in rows), 'fallen', sum(r['fallen'] for r in rows))
rep = json.loads(Path('v23_work/reports/evaluate_stage1_2026-10-18_08-24-20_v23-ten-gate512.json').read_text())
print('report', {k: round(rep[k], 4) for k in ('completion_rate', 'collision_rate', 'fallen_rate',
      'command_speed_mean', 'achieved_speed_mean', 'velocity_xy_rmse')}, 'episodes', rep['episodes'])
