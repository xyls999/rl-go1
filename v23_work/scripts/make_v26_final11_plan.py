import importlib.util, json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    'drv', '/config/Desktop/sim/scripts/run_student_v23.py')
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
from go1_hier_sim.envs.task_curriculum import BandCurriculum, LPSampler, TERRAIN_FAMILIES

assert len(TERRAIN_FAMILIES) == 11 and 'combined' in TERRAIN_FAMILIES
families = tuple(TERRAIN_FAMILIES)
sampler = LPSampler(families, temperature=1.0, min_family_share=0.06, max_family_share=0.10,
                    state_path=Path('/tmp/v26_final11_sampler.json'))
weights = BandCurriculum().instance_weights(sampler.instances, 1, [0, 1, 2, 3, 4])
plan = sampler.stage_plan(512, np.random.default_rng(2026), max_difficulty=.6, speed_min=.35,
                          speed_max=.8, target_level=1, allowed_speed_levels=[0, 1, 2, 3, 4],
                          instance_weights=weights, min_family_share=0.06, max_family_share=0.10)
drv.rebalance_family_plan(plan, families, .6)
gate = drv.make_balanced_gate_plan(plan, families, .6)
assert len(gate['instances']) == 512 and set(gate['families']) == set(families)
assert min(gate['families'].count(f) for f in families) >= 40
out = Path('/config/Desktop/sim/reports/lp_gate_plan_2026-10-18_go1-v26-final-eleven-512.json')
assert not out.exists()
out.write_text(json.dumps(gate, indent=2, allow_nan=False) + '\n')
counts = {f: gate['families'].count(f) for f in families}
print('V26_FINAL11_PLAN', out.name, counts)
