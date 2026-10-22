import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_credit_v38.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v60 virtual penetration: 7.5 cm grooves on the real combined layout; the existing '
                '-80 gap-intrusion penalty charges every step inside the void footprint, so the only '
                'profitable behaviour is to leap. Gate on the real combined.')
d['training_roster'] = 'single:combined_soft20'
d['gate_family'] = 'combined'
d['state'] = 'lp_combined_v60_state.json'
d['sampler_state'] = 'lp_combined_v60_acrl_state.json'
d['results_file'] = 'lp_combined_v60_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt'
for phase in d['phases']:
    phase['iterations'] = 600
cfg = Path('v23_work/configs/curriculum_phases_pen_v60.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v59_respawn.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_respawn_v59.json', 'curriculum_phases_pen_v60.json')
s = s.replace('9aa2746175a06ea2daf3b4e48896e437a1ef5b67e5bf39402fafe45e0799053c', sh('configs/curriculum_phases_pen_v60.json'))
s = s.replace('v59', 'v60').replace('respawn', 'penetration')
s = s.replace('export GO1_SOFT_RESPAWN=1\n', 'export GO1_V8_LONG_GAP_THRESHOLD=0.05 GO1_V19_MIN_WIDTH=0.05\n')
s = s.replace('3192772896 ', '')  # no-op guard
q = Path('v23_work/scripts/start_v60_penetration.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
print('soft respawn removed:', 'GO1_SOFT_RESPAWN' not in s)
