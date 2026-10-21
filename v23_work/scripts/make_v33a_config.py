import json
from pathlib import Path

src = Path('v23_work/configs/curriculum_phases_combined_soft_v32a.json').read_text(encoding='utf-8')
d = json.loads(src)
d['comment'] = 'v33a soft-curriculum: train on combined_soft4 (opening voids shallow 0.115m), gate on real combined.'
d['training_roster'] = 'single:combined_soft4'
d['state'] = 'lp_combined_v33a_state.json'
d['sampler_state'] = 'lp_combined_v33a_acrl_state.json'
d['results_file'] = 'lp_combined_v33a_results.txt'
out = Path('v23_work/configs/curriculum_phases_combined_soft_v33a.json')
out.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')
print('v33a config:', d['training_roster'], d['gate_family'], d['init_checkpoint'])
