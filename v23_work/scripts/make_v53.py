import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_comp1_v51.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v53 composition ladder mixture: course (0 voids) + 1/2/3-void variants + real combined, '
                'soft termination + narrow credit; gate on the real combined.')
d['training_roster'] = 'ladder:combined'
d['gate_family'] = 'combined'
d['min_family_share'] = 1.0
d['max_family_share'] = 1.0
d['state'] = 'lp_combined_v53_state.json'
d['sampler_state'] = 'lp_combined_v53_acrl_state.json'
d['results_file'] = 'lp_combined_v53_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt'
for phase in d['phases']:
    phase['iterations'] = 1000
cfg = Path('v23_work/configs/curriculum_phases_ladder_v53.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v51_comp_step1.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_comp1_v51.json', 'curriculum_phases_ladder_v53.json')
s = s.replace('cba58ef76ef6afa4a8079e167609b7e41803f46703816ef9051bcda4b1346d2e', sh('configs/curriculum_phases_ladder_v53.json'))
s = s.replace('v51', 'v53').replace('comp-step1', 'ladder')
s = s.replace('88e0f884710b9667b1d3dfe938c945127c7f822e9e7402df2365255813d9ceb4', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('939578adec2c0b41c53a7e5a6241ed7d544ccccbe3ce96ef17b8aa51fde654d9', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('c6cbcbbeb711c587b39d29493857582797e64a9c1065e60681f5a4103b750f9a', sh('source/go1_hier_sim/go1_hier_sim/envs/config.py'))
s = s.replace('4c3a0fee36b3259d4f7f7f380d70f1fc53c4cf911e5316990217712face850db', sh('source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py'))
s = s.replace('3a30cada4349083b6ec7a70a15ce186325781bf3292a5ff2d1a325ec915b9243', sh('scripts/run_student_v32.py'))
s = s.replace('d44e289445aaaee355f3ee0b63bbdbf9a377115425c9ac1522ad5224984256dc', sh('scripts/run_student_v32.py'))
q = Path('v23_work/scripts/start_v53_ladder.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
