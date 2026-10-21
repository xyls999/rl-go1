import json
from collections import Counter
from pathlib import Path

rep = json.loads(Path(
    'v23_work/reports/evaluate_stage1_2026-10-18_16-14-48_gate-2026-10-18_go1-v27-eleven-mixture-s01-p1-d1.json'
).read_text())
plan = json.loads(Path('v23_work/reports/lp_gate_plan_2026-10-18_go1-v27-eleven-mixture-s01-p1-d1.json').read_text())
fam = plan['families']
combos = [int(e) for e, st in rep['environments'].items() if fam[int(e)] == 'combined']
stats = [rep['environments'][str(e)] for e in combos]
tot_ep = sum(s['episodes'] for s in stats)
tot_succ = sum(s['success'] for s in stats)
print('combined envs', len(combos), 'episodes', tot_ep, 'success', tot_succ)
fails = [s for s in stats if s['success'] < s['episodes']]
xs = sorted(s['mean_failure_x'] for s in stats if s['mean_failure_x'])
print('mean_failure_x deciles', [round(xs[int(len(xs)*q)], 1) for q in (0, .25, .5, .75, 1.0)])
keys = set()
for s in stats:
    keys |= set(s)
print('env stat keys', sorted(keys))
for k in ('collision', 'out_of_bounds', 'fell_into_gap', 'survival'):
    if k in keys:
        print(k, sum(int(s.get(k, 0) or 0) for s in stats))
gl = [int(e) for e in rep['environments'] if fam[int(e)] == 'gaps_long']
gls = [rep['environments'][str(e)] for e in gl]
glx = sorted(s['mean_failure_x'] for s in gls if s['mean_failure_x'])
print('gaps_long fail-x median', round(glx[len(glx)//2], 1) if glx else None, 'succ',
      sum(s['success'] for s in gls), '/', sum(s['episodes'] for s in gls))
print('report overall', round(rep['completion_rate'], 4), rep.get('passed'))
