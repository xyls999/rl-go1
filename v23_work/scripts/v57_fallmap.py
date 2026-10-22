import json
import numpy as np

rows = [json.loads(l) for l in open('/config/Desktop/sim/reports/v57_soft15_trace.jsonl')]
fx = np.array([r['course_x'] for r in rows])
fall = np.array([r.get('fallen', False) for r in rows])
st = np.array([r['steps'] for r in rows])
sp = np.array([r.get('achieved_speed', 0) for r in rows])
print('n', len(rows), 'fallen', int(fall.sum()), 'success', int(sum(1 for r in rows if r.get('success'))))
print('fall x quantiles', [round(float(q), 2) for q in np.quantile(fx, [0, .1, .25, .5, .75, .9, 1])])
hist, edges = np.histogram(fx, bins=[0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 34])
print('fall x hist', dict(zip([f'{edges[i]:.0f}-{edges[i+1]:.0f}' for i in range(len(hist))], hist.tolist())))
print('steps med', float(np.median(st)), 'speed med', round(float(np.median(sp)), 2))
