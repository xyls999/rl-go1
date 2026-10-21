from pathlib import Path
import json, hashlib

c = json.loads(Path('v23_work/configs/curriculum_phases_student_distill_v29.json').read_text())
c['state'] = 'lp_student_distill_v29b_state.json'
c['sampler_state'] = 'lp_student_distill_v29b_acrl_state.json'
c['results_file'] = 'lp_student_distill_v29b_results.txt'
p = Path('v23_work/configs/curriculum_phases_student_distill_v29b.json')
p.write_text(json.dumps(c, indent=2) + '\n')
p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
print('CFG29B', hashlib.sha256(p.read_bytes()).hexdigest())

s = Path('v23_work/scripts/start_v29_student_distill.sh').read_text()
s = s.replace('go1-v29-student-distill', 'go1-v29b-student-distill')
s = s.replace('curriculum_phases_student_distill_v29.json', 'curriculum_phases_student_distill_v29b.json')
s = s.replace('v29_student_distill', 'v29b_student_distill')
s = s.replace('V29_DISTILL_PREFLIGHT', 'V29B_DISTILL_PREFLIGHT')
s = s.replace('V29_START', 'V29B_START')
s = s.replace("'source/go1_hier_sim/go1_hier_sim/training/on_policy.py': None",
              "'source/go1_hier_sim/go1_hier_sim/training/on_policy.py': "
              "'f1ea4784bf861dff18c09bcf5d426aa3063e356914b6f8504ee43b380f8ed6ec'")
s = s.replace("'source/go1_hier_sim/go1_hier_sim/cli.py': None",
              "'source/go1_hier_sim/go1_hier_sim/cli.py': "
              "'b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9'")
s = s.replace('reports/v29_smoke4x16.exit', 'reports/v29_probe_warmup.exit')
s = s.replace('lp_student_distill_v29_', 'lp_student_distill_v29b_')
for token in ('go1-v29b-student-distill', 'f1ea4784bf861dff18c09bcf5d426aa3063e356914b6f8504ee43b380f8ed6ec',
              'b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9',
              'v29_probe_warmup.exit', 'curriculum_phases_student_distill_v29b.json'):
    assert token in s, token
q = Path('v23_work/scripts/start_v29b_student_distill.sh')
q.write_text(s)
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('LAUNCHER29B', hashlib.sha256(q.read_bytes()).hexdigest())
