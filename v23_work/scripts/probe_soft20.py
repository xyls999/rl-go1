import sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_tile

m = build_tile("combined_soft20", 0.1, np.random.default_rng(2026), [])
gaps = [b for b in m.metadata["course_profile"]["blocks"] if b["kind"] == "gap"]
print("soft20 depths:", [round(b["depth"], 3) for b in gaps])
print("soft20 widths:", [round(b["width"], 3) for b in gaps])
