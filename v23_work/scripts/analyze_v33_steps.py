import glob
import json
import numpy as np

for tag in ('v33-probe-v26-soft4', 'v33-probe-v19p4-soft4', 'gate-2026-10-19_go1-v32a-combined-soft-s01-p1-d1'):
    paths = glob.glob(f'/config/Desktop/sim/reports/evaluate_stage1_*{tag}.json')
    if not paths:
        print(tag, 'missing')
        continue
    d = json.load(open(paths[0]))
    rows = list(d['environments'].values())
    steps = np.array([float(e['mean_steps']) for e in rows])
    returns = np.array([float(e['mean_return']) for e in rows])
    print(tag, 'n_env', len(rows), 'steps med', round(float(np.median(steps)), 1),
          'q10/q90', round(float(np.quantile(steps, .1)), 1), round(float(np.quantile(steps, .9)), 1),
          'return med', round(float(np.median(returns)), 2))
    print('   overall:', {k: d[k] for k in d if k in ('completion_rate', 'survival_rate', 'failure_rate', 'episodes')})
