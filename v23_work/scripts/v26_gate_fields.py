import json
from collections import defaultdict

p = '/config/Desktop/sim/reports/evaluate_stage1_2026-10-18_14-06-50_gate-2026-10-18_go1-v26-triple-refine-s01-p1-d1.json'
d = json.load(open(p))
print('overall', {k: (round(d[k], 4) if isinstance(d.get(k), float) else d.get(k))
                  for k in ('episodes', 'completion_rate', 'fallen_rate', 'timeout_rate',
                            'command_speed_mean', 'achieved_speed_mean') if k in d})
envs = d.get('environments', {})
print('envs', len(envs))
per = defaultdict(lambda: [0, 0])
for k, e in envs.items():
    fam = e.get('family') or e.get('terrain') or 'unknown'
    per[fam][0] += int(e['episodes'])
    per[fam][1] += int(e['success'])
for fam, (n, s) in sorted(per.items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0])):
    print(f'{fam:26s} n={n:4d} {s / n:.4f}')
