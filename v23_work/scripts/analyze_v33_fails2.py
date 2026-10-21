import json
import numpy as np

for tag in ('v33-probe-v26-soft4', 'v33-probe-v19p4-soft4'):
    import glob
    path = glob.glob(f'/config/Desktop/sim/reports/evaluate_stage1_*{tag}.json')[0]
    d = json.load(open(path))
    envs = d['environments']
    rows = list(envs.values())
    fx = np.array([float(e['mean_failure_x']) for e in rows if e['mean_failure_x'] and float(e['mean_failure_x']) > 0])
    steps = np.array([float(e['mean_steps']) for e in rows if e['mean_failure_x'] and float(e['mean_failure_x']) > 0])
    gap = sum(int(e['fell_into_gap']) for e in rows)
    col = sum(int(e['collision']) for e in rows)
    oob = sum(int(e['out_of_bounds']) for e in rows)
    eps = sum(int(e['episodes']) for e in rows)
    surv = np.mean([float(e['survival']) for e in rows])
    print(tag, f'episodes={eps} fell_gap={gap} collision={col} oob={oob} surv={surv:.3f}')
    if len(fx):
        hist, _ = np.histogram(fx, bins=[0, 2, 3, 5, 7, 10, 14, 18, 22, 26, 30, 34])
        print('  fail_x hist [0,2,3,5,7,10,14,18,22,26,30,34]:', hist.tolist())
        print('  fail_x quantiles:', [round(float(q), 2) for q in np.quantile(fx, [0, .25, .5, .75, 1])])
        s0 = steps[fx < 3]
        print('  steps@x<3:', (round(float(np.median(s0)), 0), len(s0)) if len(s0) else 'none',
              ' steps@x>10:', (round(float(np.median(steps[fx > 10])), 0), len(steps[fx > 10])) if (fx > 10).any() else 'none')
