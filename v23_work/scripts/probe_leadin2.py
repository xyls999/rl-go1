import sys
import json
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile, START_IN_TILE

for kind in ("combined", "gaps_short_dense", "gap_course"):
    mesh = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    v = np.asarray(mesh.vertices)
    prof = mesh.metadata["course_profile"]
    spawn_x = START_IN_TILE[0]
    # window strictly before the first blocked feature: spawn .. spawn+1.25m
    for lo, hi, tag in ((spawn_x, spawn_x + 1.25, "pre-entrance"),
                        (spawn_x - 1.0, spawn_x, "behind-spawn")):
        m = (v[:, 0] > lo) & (v[:, 0] < hi)
        if m.any():
            z = v[m, 2]
            print(f"{kind:16s} {tag:12s} n={int(m.sum()):5d} zmin={z.min():+.3f} zmax={z.max():+.3f} "
                  f"zstd={z.std():.3f}")
        else:
            print(f"{kind:16s} {tag:12s} no vertices")
    print("   roughness_amplitude", prof.get("roughness_amplitude"),
          "feature_kinds", prof.get("feature_kinds"))
