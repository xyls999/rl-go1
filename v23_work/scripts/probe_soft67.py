import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
for kind in ("combined_soft6", "combined_soft7"):
    m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    g = [b for b in m.metadata["course_profile"]["blocks"] if b["kind"] == "gap"]
    print(kind, "entrance widths:", [round(b["width"], 3) for b in g[:3]],
          "depths:", [round(b["depth"], 3) for b in g[:3]],
          "late:", round(g[3]["width"], 3))
mr = build_tile("combined", 0.1, np.random.default_rng(2026), [])
gr = [b for b in mr.metadata["course_profile"]["blocks"] if b["kind"] == "gap"]
print("real entrance widths:", [round(b["width"], 3) for b in gr[:3]])