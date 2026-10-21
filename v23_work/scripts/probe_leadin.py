import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile, START_IN_TILE

mesh = build_tile("combined", 0.1, np.random.default_rng(2026), [])
blocks = mesh.metadata["course_profile"]["blocks"]
print("all blocks sorted by start:")
for b in blocks:
    print(f"  {b['kind']:14s} {b['start']:6.2f} -> {b['end']:6.2f}")
v = np.asarray(mesh.vertices)
z = v[:, 2]
x = v[:, 0]
lead = (x > 2.0) & (x < 4.6)
print("lead-in vertex z range:", round(float(z[lead].min()), 3), round(float(z[lead].max()), 3))
print("spawn:", START_IN_TILE)
prof = mesh.metadata.get("course_profile", {})
print("profile keys:", list(prof.keys()))
for k in ("bumps", "features", "lead_in"):
    if k in prof:
        print(k, prof[k])
