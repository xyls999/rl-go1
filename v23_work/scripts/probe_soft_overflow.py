import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

for extra in (0.0, 0.05, 0.10, 0.15):
    fail = 0
    worst = 0.0
    for seed in range(600):
        rng = np.random.default_rng(2026 * 7919 + seed)
        ok = False
        for attempt in range(6):
            stream = rng if attempt == 0 else np.random.default_rng(int(rng.integers(1 << 62)))
            try:
                if extra > 0:
                    build_tile('combined_soft', 0.1, stream, [])
                else:
                    build_tile('combined', 0.1, stream, [])
                ok = True
                break
            except ValueError as e:
                if 'finishing tyres' not in str(e):
                    raise
                worst = max(worst, float(str(e).rsplit(': ', 1)[1]))
        if not ok:
            fail += 1
    print(f"extra={extra}: fail_after_6_retries={fail}/600 worst_cursor={worst:.3f}")
