import numpy as np
import sys
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
rng = np.random.default_rng(2026)
bad = []
total = 0
for rep in range(30):
    for i in range(64):
        d = float(rng.uniform(0.0, 0.6))
        total += 1
        try:
            build_tile("combined", d, rng, [])
        except Exception:
            bad.append(round(d, 3))
print("fail rate", len(bad), "/", total)
if bad:
    print("min bad d", min(bad), "max bad d", max(bad))
    import collections
    print(collections.Counter([round(x * 10) for x in bad]))