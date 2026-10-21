import json
from pathlib import Path

d = json.loads(Path('v23_work/configs/curriculum_phases_combined_soft_v33a.json').read_text(encoding='utf-8'))
d['comment'] = ('v33a soft-curriculum stage-1: train on combined_soft6 (opening voids widened to '
                'jumpable 0.29-0.42m at true depth), gate on real combined.')
d['training_roster'] = 'single:combined_soft6'
d['iterations_note'] = '800'
for phase in d['phases']:
    phase['iterations'] = 800
Path('v23_work/configs/curriculum_phases_combined_soft_v33a.json').write_text(
    json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')
print('v33a:', d['training_roster'], 'iters', d['phases'][0]['iterations'])
