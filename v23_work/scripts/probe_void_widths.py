import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
for fam in ("combined", "gaps_short", "gaps_long"):
    stats = []
    for seed in range(6):
        rng = np.random.default_rng(2026 + seed)
        mesh = build_tile(fam, 0.1 if fam == "combined" else 0.1, rng, [])
        for b in mesh.metadata["course_profile"]["blocks"]:
            if b["kind"] == "gap" and "void_end" in b:
                stats.append((round(b["void_end"] - b["start"], 3),
                              round(b.get("void_end", 0) and b.get("end", 0) or 0, 2),
                              round(b["end"] - b["void_end"], 2)))
    widths = [s[0] for s in stats]
    pads = [s[2] for s in stats]
    print(fam, "n", len(stats), "void width min/med/max",
          round(min(widths), 3), round(sorted(widths)[len(widths)//2], 3), round(max(widths), 3),
          "| pad max", round(max(pads), 2))