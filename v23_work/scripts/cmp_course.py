import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for kind in ("course", "course_gaps", "combined_soft13", "gaps_short_dense"):
    try:
        m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
    except Exception as e:
        print(kind, "BUILD FAIL", e)
        continue
    prof = m.metadata.get("course_profile", {})
    print("==", kind)
    print("  ground_segments:", prof.get("ground_segments"))
    print("  flat_approach_m:", prof.get("flat_approach_m"),
          "finish_start_offset:", prof.get("finish_start_offset"),
          "finish_end_offset:", prof.get("finish_end_offset"))
    v = np.asarray(m.vertices)
    z = v[:, 2]
    xs = v[:, 0]
    for lo, hi in ((0, 1.3), (1.3, 2.2), (2.2, 3.2), (3.2, 4.3)):
        m_ = (xs >= lo) & (xs < hi)
        if m_.any():
            print(f"  x[{lo},{hi}) zmin={z[m_].min():+.3f} zmax={z[m_].max():+.3f} n={int(m_.sum())}")
        else:
            print(f"  x[{lo},{hi}) no verts")
    print("  blocks:", [(b['kind'], round(b['start'], 2), round(b['end'], 2)) for b in prof.get('blocks', [])][:6])
