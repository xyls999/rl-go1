import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_hard_v36c.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v38: soft termination + narrow-void take-off credit on the real combined geometry; '
                'gate on real combined with normal termination.')
d['training_roster'] = 'single:combined_soft8'
d['state'] = 'lp_combined_v38_state.json'
d['sampler_state'] = 'lp_combined_v38_acrl_state.json'
d['results_file'] = 'lp_combined_v38_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt'
for phase in d['phases']:
    phase['iterations'] = 800
cfg = Path('v23_work/configs/curriculum_phases_credit_v38.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v36d_fast_command.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_fast_v36d.json', 'curriculum_phases_credit_v38.json')
s = s.replace('1c7343bfedba37644087af098ae5620327afb92ccc196d75d2765f3f7fcb79dc', sh('configs/curriculum_phases_credit_v38.json'))
s = s.replace('v36d', 'v38').replace('fast-command', 'narrow-credit')
s = s.replace('3f9cb342f53159cb5bdb1960afb0a7413bc3c19baebee7b71be9d00ffe376a15', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('89d54edf74f677a5a560841990ab9438657a7e759c5f11c0c6d4dca698dd5c9c', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a', sh('source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py'))
s = s.replace('checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt',
              'checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt')
s = s.replace('bc8501a445dbfd7fb1d95351d3372a498f5604753427b9ad542e705bde497989',
              'bc8501a445dbfd7fb1d95351d3372a498f5604753427b9ad542e705bde497989')
s = s.replace('INIT=v36c', 'INIT=v36c')
# guarantee the required env flags for v38
marker = 'export GO1_POLICY_VARIANT=map_attn_v3'
assert marker in s
s = s.replace(marker, 'export GO1_V36_SOFT_TERMINATION=1 GO1_V8_LONG_GAP_THRESHOLD=0.05 GO1_V19_MIN_WIDTH=0.05\n' + marker, 1)
q = Path('v23_work/scripts/start_v38_narrow_credit.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert 'GO1_V19_MIN_WIDTH=0.05' in s and 'v38' in s
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
