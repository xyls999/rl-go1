import json
import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

# Map fail distances to tile-local feature blocks using seed 2026 difficulty 0.1 tiles.
rng = np.random.default_rng(2026)
mesh = build_tile("combined", 0.1, rng, [])
blocks = mesh.metadata["course_profile"]["blocks"]
print("tile blocks (local x, spawn=3.0):")
for b in blocks[:14]:
    print(f"  {b['kind']:14s} {b['start']:6.2f} -> {b['end']:6.2f}"
          + (f" width {b['width']:.3f}" if b['kind'] == 'gap' else ''))

hist = {"fallen": [], "timeout": []}
kinds_at = {}
with open("/config/Desktop/sim/reports/v30b_combined_trace512.jsonl", encoding="utf-8") as fh:
    episodes = {}
    for line in fh:
        try:
            row = json.loads(line)
        except Exception:
            continue
        eid = row.get("env") or row.get("episode")
        episodes[eid] = row
    for eid, row in episodes.items():
        reason = str(row.get("termination") or row.get("reason") or row.get("done_reason") or "")
        fx = row.get("fail_x", row.get("failure_x"))
        if fx is None:
            continue
        key = "fallen" if "fall" in reason.lower() else ("timeout" if "time" in reason.lower() else "other")
        hist[key].append(float(fx))
print()
for key, vals in hist.items():
    if not vals:
        continue
    v = np.array(vals)
    print(key, "n=", len(v), "min/med/max:", round(float(v.min()), 2),
          round(float(np.median(v)), 2), round(float(v.max()), 2),
          "deciles:", [round(float(q), 2) for q in np.quantile(v, np.arange(0, 1.01, 0.2))])
