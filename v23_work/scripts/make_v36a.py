import json
from pathlib import Path
import hashlib

d = json.loads(Path('v23_work/configs/curriculum_phases_combined_soft_v33a.json').read_text(encoding='utf-8'))
d['comment'] = ('v36a P10 soft-dynamics: real combined geometry with fall termination suppressed on '
                'combined_soft8 slots (GO1_V36_SOFT_TERMINATION=1); gate on real combined hard physics.')
d['training_roster'] = 'single:combined_soft8'
d['state'] = 'lp_combined_v36a_state.json'
d['sampler_state'] = 'lp_combined_v36a_acrl_state.json'
d['results_file'] = 'lp_combined_v36a_results.txt'
for phase in d['phases']:
    phase['iterations'] = 800
out = Path('v23_work/configs/curriculum_phases_combined_soft_v36a.json')
out.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')
print('config', hashlib.sha256(out.read_bytes()).hexdigest())

# runner registry: add soft8
p3 = Path('v23_work/scripts/run_student_v32.py')
s3 = p3.read_text(encoding='utf-8')
old = '''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7",
    ))
'''
assert s3.count(old) == 1
s3 = s3.replace(old, '''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8",
    ))
''')
p3.write_text(s3, encoding='utf-8', newline='\n')
import ast
ast.parse(s3)
print('runner', hashlib.sha256(p3.read_bytes()).hexdigest())
