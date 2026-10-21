import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

m = build_tile("combined", 0.1, np.random.default_rng(2026), [])
for x in (4.25, 4.34, 4.45, 4.75, 5.10, 5.16, 5.5, 6.0, 6.6, 6.7):
    origins = np.array([[x, 8.0, 2.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    locs, idx_ray, idx_tri = m.ray.intersects_location(origins, dirs, multiple_hits=True)
    zs = sorted(round(float(l[2]), 3) for l in locs)
    print(f"x={x:.2f} hits={zs}")
