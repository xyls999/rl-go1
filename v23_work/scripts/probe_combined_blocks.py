import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
for d in (0.0, 0.05, 0.1):
    rng = np.random.default_rng(2026)
    mesh = build_tile("combined", d, rng, [])
    blocks = mesh.metadata["course_profile"]["blocks"]
    gaps = blocks[:3]
    print("difficulty", d)
    for i, b in enumerate(gaps):
        print("  gap", i, "kind", b["kind"], "start", round(b["start"], 2),
              "end", round(b["end"], 2), "width", round(b["end"] - b["start"], 3),
              "depth", round(b.get("depth", float("nan")), 3))
    tires = [b for b in blocks if b["kind"] == "tire_field"]
    print("  tire band start/end", round(tires[0]["start"], 2), round(tires[0]["end"], 2) if tires else None)
    others = [b["kind"] for b in blocks[3:] if b["kind"] != "tire_field"]
    print("  other blocks", others[:6])