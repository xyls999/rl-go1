import json
import numpy as np
rows = [json.loads(l) for l in open("/config/Desktop/sim/reports/v35_trace_v26-on-combined.jsonl")]
fx = np.array([r["course_x"] for r in rows])
steps = np.array([r["steps"] for r in rows])
fall = np.array([r["fallen"] for r in rows])
print("n", len(rows), "fallen", int(fall.sum()))
print("fall_x:", sorted([round(float(x),2) for x in fx[fall]])[:14])
print("fall_steps med:", float(np.median(steps[fall])))
tfx = fx[~fall]
print("nonfall_x med:", round(float(np.median(tfx)),2) if len(tfx) else "none", "n", len(tfx))
import collections
bins = collections.Counter(min(7, int(x//4)) for x in fx[fall])
print("fall_x hist(0-4,4-8,8-12,12-16,16-20,20-24,24-28,28+):", [bins.get(i,0) for i in range(8)])