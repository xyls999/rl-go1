import numpy as np
import sys
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
found = None
for seed in range(4):
    rng = np.random.default_rng(2000 + seed)
    for i in range(50):
        d = float(rng.uniform(0.0, 0.6))
        try:
            build_tile("combined", d, rng, [])
        except Exception as e:
            found = (seed, i, round(d, 4), str(e)[:80])
            break
    if found:
        break
print("FOUND", found)
if found:
    seed, i, d, _ = found
    rng2 = np.random.default_rng(2000 + seed)
    for j in range(i + 1):
        dj = float(rng2.uniform(0.0, 0.6))
        if j == i:
            try:
                build_tile("combined", dj, rng2, [])
                print("NOT REPRODUCIBLE at exact replay")
            except Exception as e2:
                print("REPRODUCIBLE", round(dj, 4), str(e2)[:60])