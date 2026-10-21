import importlib.util, json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    'drv', '/config/Desktop/sim/scripts/run_student_v27.py')
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
from go1_hier_sim.envs.task_curriculum import BandCurriculum, LPSampler, TERRAIN_FAMILIES
from go1_hier_sim.envs.eleven_roster_v27 import select_training_roster

families = select_training_roster({'training_roster': 'eleven_with_combined'}, TERRAIN_FAMILIES)
assert len(families) == 11
sampler = LPSampler(families, state_path=Path('/tmp/v27_smoke_sampler_state.json'))
weights = BandCurriculum().instance_weights(sampler.instances, 1, [0, 1, 2, 3, 4])
plan = sampler.stage_plan(16, np.random.default_rng(2026), max_difficulty=.6, speed_min=.35,
                          speed_max=.8, target_level=1, allowed_speed_levels=[0, 1, 2, 3, 4],
                          instance_weights=weights)
drv.rebalance_family_plan(plan, families, .6)
assert len(plan['instances']) == 16 and set(plan['families']) == set(families)
out = Path('/config/Desktop/sim/reports/v27_smoke_plan16.json')
assert not out.exists()
out.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
print('V27_SMOKE_PLAN16', len(plan['instances']), sorted(set(plan['families']))[:3], '...', len(set(plan['families'])), 'families')