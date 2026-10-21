import json
import numpy as np
rows=[json.loads(l) for l in open("/config/Desktop/sim/reports/v36c_trace_combined128.jsonl")]
n=len(rows)
fall=[r for r in rows if r.get("fallen") or r.get("fell_into_gap")]
still=[r for r in rows if not (r.get("fallen") or r.get("fell_into_gap")) and not r.get("success")]
succ=[r for r in rows if r.get("success")]
print("n",n,"fall",len(fall),"stall",len(still),"success",len(succ))
if still:
    xs=np.array([r["course_x"] for r in still])
    print("stall final_x quantiles",[round(float(q),1) for q in np.quantile(xs,[0,.1,.25,.5,.75,.9,1])])
    hist,_=np.histogram(xs,bins=[0,4,8,12,16,20,24,28,31,33,40])
    print("stall hist [0,4,8,12,16,20,24,28,31,33,40]:",hist.tolist())
    sp=np.array([r["achieved_speed"] for r in still])
    print("stall achieved_speed med",round(float(np.median(sp)),3))
    st=np.array([r["steps"] for r in still])
    print("stall steps med",float(np.median(st)))
if fall:
    xs=np.array([r["course_x"] for r in fall])
    print("fall final_x",[round(float(q),1) for q in np.quantile(xs,[0,.5,1])])