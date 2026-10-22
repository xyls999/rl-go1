import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_pen_v60.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v61 calibrated virtual penetration: 3.5-4.5 cm grooves with the one-off entry penalty '
                'disabled, so walking through costs ~2-3 per step while leaping costs nothing.')
d['training_roster'] = 'single:combined_soft21'
d['state'] = 'lp_combined_v61_state.json'
d['sampler_state'] = 'lp_combined_v61_acrl_state.json'
d['results_file'] = 'lp_combined_v61_results.txt'
for phase in d['phases']:
    phase['iterations'] = 500
cfg = Path('v23_work/configs/curriculum_phases_pen2_v61.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v60_penetration.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_pen_v60.json', 'curriculum_phases_pen2_v61.json')
s = s.replace('49086b01282946c3b62535592830bfc746e297906731135dec3a40d8b63518dc', sh('configs/curriculum_phases_pen2_v61.json'))
s = s.replace('v60', 'v61').replace('penetration', 'penetration2')
s = s.replace('export GO1_V8_LONG_GAP_THRESHOLD=0.05',
              'export GO1_GAP_ENTRY_MULT=0 GO1_V8_LONG_GAP_THRESHOLD=0.05')
s = s.replace('c30c2f6b0875731925a4af2fe00d0a6a40a88a44b76ecde95108f4b918738555',
              sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('4a18748a317a45aee5609af87d8e0e0106508d5242595666db44771fce3fdb99',
              sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('6b1ba53ddfe092016baccb9445bbd113d0be9d9c889085fa5e5577c22bfd498b',
              sh('scripts/run_student_v32.py'))
q = Path('v23_work/scripts/start_v61_penetration2.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
print('flags ok', 'GO1_GAP_ENTRY_MULT=0' in s)
