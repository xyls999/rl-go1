import numpy as np
import sys
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
rng = np.random.default_rng(7)
ok, bad = [], []
for i in range(61):
    d = i / 100.
    try:
        build_tile("combined", d, rng, [])
        ok.append(d)
    except Exception as e:
        bad.append((d, str(e)[:60]))
print("OK max:", max(ok) if ok else None, "count", len(ok))
print("BAD:", bad[:8])