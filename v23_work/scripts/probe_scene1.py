import json, sys
import numpy as np
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_scene
plan = json.load(open("/config/Desktop/sim/reports/lp_plan_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.json"))
tasks = [(f, d) for f, d in zip(plan["families"], plan["difficulty"])]
print("tasks", len(tasks), "max diff", max(d for _, d in tasks))
import inspect
print(inspect.signature(build_scene))