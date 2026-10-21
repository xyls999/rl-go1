import sys
import numpy as np
import trimesh
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for kind in ("combined", "gaps_short_dense", "combined_soft13"):
    m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    origins = np.array([[x, 8.0, 3.0] for x in np.arange(3.0, 9.0, 0.05)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    locs, idx_ray, idx_tri = m.ray.intersects_location(origins, dirs, multiple_hits=False)
    surf = {}
    for loc, ray in zip(locs, idx_ray):
        surf.setdefault(int(ray), []).append(float(loc[2]))
    zs = np.full(len(origins), np.nan)
    for i in range(len(origins)):
        if i in surf:
            zs[i] = max(surf[i])
    xs = origins[:, 0]
    print(f"== {kind}")
    # print surface profile coarsely
    line = []
    for x in np.arange(3.0, 9.0, 0.25):
        j = int(round((x - 3.0) / 0.05))
        line.append(f"{zs[j]:+.2f}" if j < len(zs) and np.isfinite(zs[j]) else " -- ")
    print("   x=3.0..9.0 (0.25):", " ".join(line))
    holes = xs[~np.isfinite(zs)]
    if len(holes):
        print("   NO-HIT x ranges (first/last):", round(float(holes.min()), 2), round(float(holes.max()), 2),
              "count", len(holes))
