import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

m = build_tile("combined_soft15", 0.1, np.random.default_rng(2026), [])
gaps = [b for b in m.metadata["course_profile"]["blocks"] if b["kind"] == "gap"]
print("soft15 gap depths:", [round(b["depth"], 3) for b in gaps])
print("soft15 gap widths:", [round(b["width"], 3) for b in gaps])
v = np.asarray(m.vertices)
print("min z:", round(float(v[:, 2].min()), 3))
mr = build_tile("combined", 0.1, np.random.default_rng(2026), [])
gr = [b for b in mr.metadata["course_profile"]["blocks"] if b["kind"] == "gap"]
print("real depths:", [round(b["depth"], 3) for b in gr])
