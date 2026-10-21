import json
from pathlib import Path
import hashlib

d = json.loads(Path('v23_work/configs/curriculum_phases_hard_v36b.json').read_text(encoding='utf-8'))
d['comment'] = 'v36c extended hard-physics fine-tune of v36b (fall rate 6.6%; gap is pace, not obstacles).'
d['state'] = 'lp_combined_v36c_state.json'
d['sampler_state'] = 'lp_combined_v36c_acrl_state.json'
d['results_file'] = 'lp_combined_v36c_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-19_go1-v36b-hard-finetune-s01-p1-d1/model_399.pt'
for phase in d['phases']:
    phase['iterations'] = 800
p = Path('v23_work/configs/curriculum_phases_hard_v36c.json')
p.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

s = Path('v23_work/scripts/start_v36b_hard_finetune.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_hard_v36b.json', 'curriculum_phases_hard_v36c.json')
s = s.replace('v36b', 'v36c').replace('hard-finetune', 'hard-finetune2')
s = s.replace('1c4a3852f7455daf49aae337fc629ef51eabcdd41307dadfb9ab5a9955ebfadb',
              hashlib.sha256(p.read_bytes()).hexdigest())
s = s.replace('checkpoints/stage1/2026-10-19_go1-v36a-combined-soft8-s01-p1-d1/model_799.pt',
              'checkpoints/stage1/2026-10-19_go1-v36b-hard-finetune-s01-p1-d1/model_399.pt')
s = s.replace('2baa9497ca618e7db45a580302e1d2b85a766fc30718701b41210ec6911afa85',
              '472ae1c319d302fd94dc8a607234fa11cb64b1a7b96cce8ecf5a38035a1457b7')
s = s.replace('INIT=v36a', 'INIT=v36b')
q = Path('v23_work/scripts/start_v36c_hard_finetune2.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert 'v36c' in s and '472ae1c3' in s
print('config', hashlib.sha256(p.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
