import json
import numpy as np

rows = [json.loads(l) for l in open('/config/Desktop/sim/reports/v41_dense_trace.jsonl')]
succ = [r for r in rows if r.get('success')]
print('n', len(rows), 'succ', len(succ))
if succ:
    xs = np.array([r['course_x'] for r in succ])
    st = np.array([r['steps'] for r in succ])
    print('success course_x: min/med/max', round(float(xs.min()), 2),
          round(float(np.median(xs)), 2), round(float(xs.max()), 2))
    print('success steps med', float(np.median(st)))
fail = [r for r in rows if not r.get('success')]
if fail:
    print('fail course_x', sorted(round(float(r['course_x']), 2) for r in fail)[:8])
