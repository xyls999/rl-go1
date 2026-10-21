import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_credit_v38.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v47 privileged specialist: exact terrain features in the actor (P10 privileged teacher), '
                'soft termination on the training-only soft8 slot, gate on the real combined.')
d['state'] = 'lp_combined_v47_state.json'
d['sampler_state'] = 'lp_combined_v47_acrl_state.json'
d['results_file'] = 'lp_combined_v47_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt'
for phase in d['phases']:
    phase['iterations'] = 800
cfg = Path('v23_work/configs/curriculum_phases_priv_v47.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v38_narrow_credit.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_credit_v38.json', 'curriculum_phases_priv_v47.json')
s = s.replace('75387293ed56fa223695a45c8a2dea3fc7d5f841cf4c51c0cc8c1c181531712a', sh('configs/curriculum_phases_priv_v47.json'))
s = s.replace('0fffeec907a754668a59aa247f031260bc2fbb0ef754cc905aec325e5b76fc16', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('26ddb3969431b4c1df77cd2e81d557ef3124e28ddd32593469e7e58395454989', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('23cc8d96efc687ad00a6b5224ae58547e9f90883cf702d44528dc3c2722e8ae9', sh('source/go1_hier_sim/go1_hier_sim/envs/config.py'))
s = s.replace('v38', 'v47').replace('narrow-credit', 'privileged')
s = s.replace('bc8501a445dbfd7fb1d95351d3372a498f5604753427b9ad542e705bde497989',
              '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674')
s = s.replace('checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt',
              'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt')
s = s.replace('INIT=v36c', 'INIT=v26')
marker = 'export GO1_POLICY_VARIANT=map_attn_v3'
assert marker in s
s = s.replace(marker, 'export GO1_PRIVILEGED_ACTOR=1\n' + marker, 1)
q = Path('v23_work/scripts/start_v47_privileged.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
print('has priv flag', 'GO1_PRIVILEGED_ACTOR=1' in s)
