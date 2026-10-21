import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile
def depths(fam):
    out = []
    for seed in range(6):
        rng = np.random.default_rng(2026 + seed)
        mesh = build_tile(fam, 0.1, rng, [])
        for b in mesh.metadata["course_profile"]["blocks"]:
            if b["kind"] == "gap":
                out.append((round(b["void_end"] - b["start"], 3),
                            round(b.get("depth", float("nan")), 3)))
    return out
for fam in ("gaps_short", "gaps_long", "combined"):
    d = depths(fam)
    narrow = [x for x in d if x[0] <= 0.26]
    wide = [x for x in d if x[0] > 0.26]
    print(fam, "total", len(d),
          "narrow<=0.26:", (min((x[1] for x in narrow), default=None), max((x[1] for x in narrow), default=None), len(narrow)),
          "wide:", (min((x[1] for x in wide), default=None), max((x[1] for x in wide), default=None), len(wide)))