import json
p='/config/Desktop/sim/reports/evaluate_stage1_2026-10-20_19-39-05_gate-2026-10-19_go1-v46-long-continuation-s01-p1-d1.json'
d=json.load(open(p))
for k in ("episodes","completion_rate","fallen_rate","collision_rate","fell_into_gap_rate","timeout_rate","command_speed_mean","achieved_speed_mean"):
    v=d.get(k)
    print(k, round(v,4) if isinstance(v,float) else v)