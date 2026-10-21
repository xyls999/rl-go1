import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
import go1_hier_sim.envs.geometry as g
print("GAP_WIDTH_MIN", getattr(g, "GAP_WIDTH_MIN", None))
print("GAP_SHORT_WIDTH_MAX", getattr(g, "GAP_SHORT_WIDTH_MAX", None))
print("GAP_JUMP_WIDTH_M", getattr(g, "GAP_JUMP_WIDTH_M", None))
print("GAP_LONG_WIDTH_MIN", getattr(g, "GAP_LONG_WIDTH_MIN", None))
print("GAP_LONG_WIDTH_MAX", getattr(g, "GAP_LONG_WIDTH_MAX", None))
for fam in ("gaps_short", "gaps_long"):
    widths = []
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        for d in (0.0, 0.3, 0.6):
            mesh = g.build_tile(fam, d, rng, [])
            for b in mesh.metadata["course_profile"]["blocks"]:
                if b["kind"] == "gap":
                    widths.append(round(b["end"] - b["start"], 3))
    print(fam, "gap widths min/max", min(widths), max(widths), "n", len(widths))