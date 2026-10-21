import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

m4 = build_tile("combined_soft4", 0.1, np.random.default_rng(2026), [])
g = [b for b in m4.metadata["course_profile"]["blocks"] if b["kind"] == "gap"][:4]
print("soft4:", [(round(b["depth"], 3), round(b["width"], 3), round(b["drop"], 3)) for b in g])
mr = build_tile("combined", 0.1, np.random.default_rng(2026), [])
gr = [b for b in mr.metadata["course_profile"]["blocks"] if b["kind"] == "gap"][:4]
print("real:", [(round(b["depth"], 3), round(b["width"], 3), round(b["drop"], 3)) for b in gr])
v = np.asarray(m4.vertices)
deep = v[v[:, 2] < -0.30]
print("verts below -0.30 (should be only later real gaps):", round(float(deep[:, 0].min()), 2),
      round(float(deep[:, 0].max()), 2), len(deep))
