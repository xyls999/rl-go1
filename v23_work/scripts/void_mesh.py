import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for kind in ("combined", "gaps_short_dense"):
    m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    v = np.asarray(m.vertices)
    deep = v[(v[:, 2] < -0.30)]
    print(f"== {kind}: verts<{len(v)} deep(<-0.30)={len(deep)}")
    if len(deep):
        xs = np.sort(np.unique(np.round(deep[:, 0], 2)))
        print("   deep x ranges:", xs[:6], "...", xs[-6:] if len(xs) > 6 else "")
    # upward rays from below at several y
    for y in (7.0, 7.5, 8.0, 8.5, 9.0):
        origins = np.array([[x, y, -2.0] for x in np.arange(4.0, 7.2, 0.1)])
        dirs = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
        locs, idx_ray, _ = m.ray.intersects_location(origins, dirs, multiple_hits=False)
        zmap = {}
        for loc, ray in zip(locs, idx_ray):
            zmap.setdefault(int(ray), []).append(float(loc[2]))
        vals = [min(zmap[i]) if i in zmap else None for i in range(len(origins))]
        prof = " ".join("--" if v is None else f"{v:+.2f}" for v in vals[:16])
        print(f"   y={y}: x=4.0..5.5 from-below surface: {prof}")
