import sys
import types
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile as ref_build

SRC = open("/config/Desktop/sim/source/go1_hier_sim/go1_hier_sim/envs/geometry.py", encoding="utf-8").read()

for extra in ("0.15", "0.10", "0.05"):
    src = SRC.replace("(0.15 if soft_entrance else 0.0)", f"({extra} if soft_entrance else 0.0)")
    mod = types.ModuleType(f"g_{extra}")
    mod.__dict__["__name__"] = f"g_{extra}"
    exec(compile(src, f"g_{extra}", "exec"), mod.__dict__)
    bt = mod.build_tile
    fail1 = fail6 = 0
    for seed in range(400):
        rng = np.random.default_rng(2026 * 7919 + seed)
        ok = False
        for attempt in range(6):
            stream = rng if attempt == 0 else np.random.default_rng(int(rng.integers(1 << 62)))
            try:
                bt("combined_soft", 0.1, stream, [])
                ok = True
                break
            except ValueError as e:
                if "finishing tyres" not in str(e):
                    raise
        if attempt == 0 and not ok:
            pass
        if not ok:
            fail6 += 1
    # single-attempt rate on first 400 fresh streams
    fail1 = 0
    for seed in range(400):
        try:
            bt("combined_soft", 0.1, np.random.default_rng(2026 * 7919 + seed), [])
        except ValueError:
            fail1 += 1
    print(f"extra={extra}: single-attempt fail={fail1}/400, after-6-retry fail={fail6}/400 tiles")
