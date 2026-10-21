import json
from pathlib import Path
import hashlib

base = json.loads(Path('v23_work/configs/curriculum_phases_credit_v38.json').read_text(encoding='utf-8'))
d = dict(base)
d['comment'] = ('v51 composition curriculum step 1: exactly ONE entrance void (rest of the real combined '
                'route intact), soft termination + narrow credit, gate on the real combined.')
d['training_roster'] = 'single:combined_soft13'
d['state'] = 'lp_combined_v51_state.json'
d['sampler_state'] = 'lp_combined_v51_acrl_state.json'
d['results_file'] = 'lp_combined_v51_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt'
for phase in d['phases']:
    phase['iterations'] = 500
cfg = Path('v23_work/configs/curriculum_phases_comp1_v51.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v48_soft_dynamics.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_softdyn_v48.json', 'curriculum_phases_comp1_v51.json')
s = s.replace('41e39d2e6f3eab4a1e5efb65a820e6a7787fe88260c11281f82f6c777bf89dfc', sh('configs/curriculum_phases_comp1_v51.json'))
s = s.replace('v48', 'v51').replace('soft-dynamics', 'comp-step1')
s = s.replace('0fffeec907a754668a59aa247f031260bc2fbb0ef754cc905aec325e5b76fc16', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('26ddb3969431b4c1df77cd2e81d557ef3124e28ddd32593469e7e58395454989', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('f9e06cb1db6d308680bd69cf8e090b578a1c8284560eb8bb613e9e718eef66d1', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('23cc8d96efc687ad00a6b5224ae58547e9f90883cf702d44528dc3c2722e8ae9', sh('source/go1_hier_sim/go1_hier_sim/envs/config.py'))
s = s.replace('c6cbcbbeb711c587b39d29493857582797e64a9c1065e60681f5a4103b750f9a', sh('source/go1_hier_sim/go1_hier_sim/envs/config.py'))
s = s.replace('6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
              '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674')
q = Path('v23_work/scripts/start_v51_comp_step1.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
