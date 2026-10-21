import json
import os

FAMS = ['stairs_up_long', 'stairs_down_long', 'stairs_mixed', 'stairs_narrow_8_mixed',
        'slope_up', 'slope_down', 'slope_mixed', 'tire_field', 'gaps_short', 'gaps_long',
        'combined']
base = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
os.makedirs('/config/Desktop/sim/reports/videos_v26', exist_ok=True)
for fam in FAMS:
    e = dict(base)
    e['families'] = [fam]
    e['difficulty'] = [0.1]
    e['difficulty_level'] = [1]
    e['instances'] = [[fam, 1, 2]]
    e['speed_level'] = [2]
    e['allowed_speed_levels'] = [0, 1, 2, 3, 4]
    json.dump(e, open(f'/config/Desktop/sim/reports/videos_v26/plan_{fam}.json', 'w'), indent=2)
print('plans ok', len(FAMS))
