import json, sys
sys.path.insert(0, "/config/Desktop/sim/source/go1_hier_sim")
from go1_hier_sim.envs.geometry import build_scene
plan = json.load(open("/config/Desktop/sim/reports/lp_plan_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.json"))
tasks = [(f, d) for f, d in zip(plan["families"], plan["difficulty"])]
for seed in (2026, 2027, 2028, 2029):
    try:
        build_scene(512, "lp_acrl", True, seed, 0.0, 0.0, 0.6, tasks=tasks)
        print("SEED", seed, "OK")
    except Exception as e:
        print("SEED", seed, "FAIL", str(e)[:70])