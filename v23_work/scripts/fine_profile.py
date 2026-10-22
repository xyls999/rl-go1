import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for kind in ("combined", "course", "combined_soft15"):
    m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    xs = np.arange(4.20, 5.30, 0.01)
    origins = np.array([[x, 8.0, 2.0] for x in xs])
    dirs = np.tile([0.0, 0.0, -1.0], (len(xs), 1))
    locs, idx_ray, idx_tri = m.ray.intersects_location(origins, dirs, multiple_hits=True)
    surf = {}
    for loc, ray in zip(locs, idx_ray):
        surf.setdefault(int(ray), []).append(round(float(loc[2]), 3))
    prof = []
    for i, x in enumerate(xs):
        hits = sorted(surf.get(i, []), reverse=True)
        prof.append(hits[0] if hits else None)
    print(f"== {kind}")
    line = " ".join('--' if v is None else f"{v:+.2f}" for v in prof)
    print("   x=4.20..5.29 step .01:", line)
