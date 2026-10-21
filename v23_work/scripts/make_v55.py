import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_ladder_v53.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v55 multi-resolution privileged teacher: actor sees exact coarse+fine scandots '
                '(0.75x1.15 m at 0.05 m) on the real combined layout with soft termination; '
                'gate on the real combined.')
d['training_roster'] = 'single:combined_soft8'
d['gate_family'] = 'combined'
d['state'] = 'lp_combined_v55_state.json'
d['sampler_state'] = 'lp_combined_v55_acrl_state.json'
d['results_file'] = 'lp_combined_v55_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt'
for phase in d['phases']:
    phase['iterations'] = 900
cfg = Path('v23_work/configs/curriculum_phases_multires_v55.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v53_ladder.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_ladder_v53.json', 'curriculum_phases_multires_v55.json')
s = s.replace('5a83f3b447d384cd8d4f82f96e14bf6b474fcb8312a9556306724a87591b6a0f', sh('configs/curriculum_phases_multires_v55.json'))
s = s.replace('v53', 'v55').replace('ladder', 'multires')
s = s.replace('6500267b26ad1d14401e0b8084e45a8845a038aab2de3d1c401203908a9b21e5', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('939578adec2c0b41c53a7e5a6241ed7d544ccccbe3ce96ef17b8aa51fde654d9', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('c6cbcbbeb711c587b39d29493857582797e64a9c1065e60681f5a4103b750f9a', sh('source/go1_hier_sim/go1_hier_sim/envs/config.py'))
s = s.replace('export GO1_POLICY_VARIANT=map_attn_v3',
              'export GO1_PRIVILEGED_ACTOR=1 GO1_MULTIRES_SCAN=1\nexport GO1_POLICY_VARIANT=map_attn_v3', 1)
q = Path('v23_work/scripts/start_v55_multires.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
