import glob
import json
import numpy as np

paths = sorted(glob.glob('/config/Desktop/sim/reports/evaluate_stage1_*v33-probe*soft4*.json'))
print('reports:', paths)
for path in paths:
    d = json.load(open(path))
    tag = path.split('evaluate_stage1_')[1][:24]
    envs = d.get('environments', {})
    fx, fall_x, tof_x = [], [], []
    for k, e in envs.items():
        x = e.get('mean_failure_x', None)
        comp = e.get('completion') or e.get('completed') or 0
        ok = sum(1 for _ in [0] if comp)
        if x is not None and float(x) > 0:
            fx.append(float(x))
    if fx:
        v = np.array(fx)
        hist, edges = np.histogram(v, bins=[0, 2, 4, 5, 7, 10, 15, 20, 25, 30, 34])
        print(tag, 'fail_x n=', len(v), 'hist[0-2,2-4,4-5,5-7,7-10,10-15,15-20,20-25,25-30,30-34]=',
              hist.tolist(), 'median=', round(float(np.median(v)), 2))
    else:
        # fallback: dump per-env keys
        one = next(iter(envs.values()))
        print(tag, 'env keys:', sorted(one.keys()), 'n_envs', len(envs))
