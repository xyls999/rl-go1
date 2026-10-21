from pathlib import Path
import hashlib, json

s = Path('v23_work/scripts/run_student_v27.py').read_text()
assert 'from go1_hier_sim.envs.eleven_roster_v27 import select_training_roster' in s
s = s.replace('from go1_hier_sim.envs.eleven_roster_v27 import select_training_roster',
              'from go1_hier_sim.envs.combined_roster_v30 import select_training_roster')
s = s.replace('Stage-C mixture plan must contain all eleven families including combined',
              'Stage-C teacher plan must contain only the combined family')
s = s.replace('Stage-C mixture gate must cover all eleven families with at least 40 slots each',
              'Stage-C teacher gate must cover combined with at least 40 slots')
p = Path('v23_work/scripts/run_student_v30.py'); p.write_text(s); p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
print('DRIVER30', hashlib.sha256(p.read_bytes()).hexdigest())

c = json.loads(Path('v23_work/configs/curriculum_phases_student_distill_v29.json').read_text())
c['comment'] = 'Stage-C combined specialization teacher: pure combined on real hard-collision physics; distillation reuses it only once competent.'
c['training_roster'] = 'single:combined'
c['min_family_share'] = 1.0; c['max_family_share'] = 1.0
c['state'] = 'lp_combined_v30_state.json'; c['sampler_state'] = 'lp_combined_v30_acrl_state.json'
c['results_file'] = 'lp_combined_v30_results.txt'
q = Path('v23_work/configs/curriculum_phases_combined_v30.json'); q.write_text(json.dumps(c, indent=2) + '\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('CFG30', hashlib.sha256(q.read_bytes()).hexdigest())
print('ROSTER30', hashlib.sha256(Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v30.py').read_bytes()).hexdigest())
for f in ['v23_work/source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v30.py','v23_work/tests/test_v30_combined_teacher_contract.py']:
    z = Path(f); z.write_bytes(z.read_bytes().replace(b'\r\n', b'\n')); print('strip', f)