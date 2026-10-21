import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for kind in ("combined_soft15", "combined"):
    try:
        m = build_tile(kind, 0.1, np.random.default_rng(2026), [])
        print(kind, 'verts', len(m.vertices), 'meta keys', list(m.metadata.keys())[:8])
        prof = m.metadata.get("course_profile")
        print('   profile?', prof is not None)
    except Exception as e:
        print(kind, 'ERROR', type(e).__name__, e)
