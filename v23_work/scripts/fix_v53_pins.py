from pathlib import Path
import hashlib
import re

q = Path('v23_work/scripts/start_v53_ladder.sh')
s = q.read_text(encoding='utf-8')
targets = {
    'scripts/run_student_v32.py': 'scripts/run_student_v32.py',
    'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py': 'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py',
    'source/go1_hier_sim/go1_hier_sim/envs/geometry.py': 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py',
    'source/go1_hier_sim/go1_hier_sim/envs/runtime.py': 'source/go1_hier_sim/go1_hier_sim/envs/runtime.py',
    'source/go1_hier_sim/go1_hier_sim/envs/config.py': 'source/go1_hier_sim/go1_hier_sim/envs/config.py',
    'configs/curriculum_phases_ladder_v53.json': 'configs/curriculum_phases_ladder_v53.json',
}
for key, path in targets.items():
    new = hashlib.sha256(Path('v23_work/' + path).read_bytes()).hexdigest()
    pat = re.compile(r"'" + re.escape(key) + r"': '([0-9a-f]{64})'")
    m = pat.findall(s)
    print(key, 'pinned', m, '->', new)
    if m:
        s = pat.sub("'" + key + "': '" + new + "'", s)
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
