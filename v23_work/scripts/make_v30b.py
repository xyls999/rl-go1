from pathlib import Path
import json, hashlib

c = json.loads(Path('v23_work/configs/curriculum_phases_combined_v30.json').read_text())
c['state'] = 'lp_combined_v30b_state.json'
c['sampler_state'] = 'lp_combined_v30b_acrl_state.json'
c['results_file'] = 'lp_combined_v30b_results.txt'
p = Path('v23_work/configs/curriculum_phases_combined_v30b.json')
p.write_text(json.dumps(c, indent=2) + '\n')
p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
print('CFG30B', hashlib.sha256(p.read_bytes()).hexdigest())

s = Path('v23_work/scripts/start_v30_combined_teacher.sh').read_text()
s = s.replace('go1-v30-combined-teacher', 'go1-v30b-combined-teacher')
s = s.replace('curriculum_phases_combined_v30.json', 'curriculum_phases_combined_v30b.json')
s = s.replace('v30_combined_teacher', 'v30b_combined_teacher')
s = s.replace('lp_combined_v30_', 'lp_combined_v30b_')
s = s.replace('V30_COMBINED_PREFLIGHT', 'V30B_COMBINED_PREFLIGHT')
s = s.replace('V30_START', 'V30B_START')
s = s.replace("'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',",
              "'source/go1_hier_sim/go1_hier_sim/envs/tire_brake_v23.py': '41a9dec73c4f150f2e5cf1d1484312a72c15220ed14b2e73c6d1c483d65941fe',\n"
              " 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py': '298c24671689421b8861083868abe75e4c7d4565da4af0bab48bc02506f57f13',")
s = s.replace('assert (root / ' + chr(39) + 'reports/v30_smoke4x16.exit' + chr(39) + ').read_text().startswith(' + chr(39) + 'exit=0 ' + chr(39) + ')',
              'assert (root / ' + chr(39) + 'reports/v30_smoke4x16.exit' + chr(39) + ').read_text().startswith(' + chr(39) + 'exit=0 ' + chr(39) + ')\nassert (root / ' + chr(39) + 'reports/v30_combined_teacher.exit' + chr(39) + ').read_text().startswith(' + chr(39) + 'exit=1 ')
for token in ('go1-v30b-combined-teacher', 'curriculum_phases_combined_v30b.json',
              'geometry.py', 'reports/v30_combined_teacher.exit'):
    assert token in s, token
q = Path('v23_work/scripts/start_v30b_combined_teacher.sh')
q.write_text(s)
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('LAUNCHER30B', hashlib.sha256(q.read_bytes()).hexdigest())
