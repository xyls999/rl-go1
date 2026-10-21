import glob, json
ps = sorted(glob.glob('/config/Desktop/sim/reports/evaluate_stage1_*gate*v38*.json'))
print(ps)
for p in ps[-1:]:
    d = json.load(open(p))
    for k in ("episodes","completion_rate","fallen_rate","collision_rate","fell_into_gap_rate",
              "timeout_rate","out_of_bounds_rate","command_speed_mean","achieved_speed_mean"):
        if k in d:
            v = d[k]
            print(k, round(v,4) if isinstance(v,float) else v)