import glob, json
for tag in ("v36a","v36b","v36c"):
    ps=glob.glob(f"/config/Desktop/sim/reports/evaluate_stage1_*gate*{tag}*.json")
    if not ps:
        print(tag,"missing"); continue
    d=json.load(open(ps[0]))
    print(tag, {k: (round(d[k],4) if isinstance(d[k],float) else d[k])
                for k in ("completion_rate","success_rate","fallen_rate","collision_rate",
                          "fell_into_gap_rate","timeout_rate","out_of_bounds_rate",
                          "command_speed_mean","achieved_speed_mean","episodes") if k in d})