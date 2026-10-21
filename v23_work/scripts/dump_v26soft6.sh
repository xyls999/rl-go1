import glob, json
for tag in ("v33-probe-v26-soft6", "v33-probe-v26-combined_soft2"):
    paths = glob.glob(f"/config/Desktop/sim/reports/evaluate_stage1_*{tag}.json")
    if not paths:
        print(tag, "missing"); continue
    d = json.load(open(paths[0]))
    print("==", tag)
    for k, v in d.items():
        if not isinstance(v, dict):
            print("  ", k, "=", round(v, 3) if isinstance(v, float) else v)