from pathlib import Path
import hashlib
import re

q = Path('v23_work/scripts/start_v59_respawn.sh')
s = q.read_text(encoding='utf-8')
for key in ('scripts/run_student_v32.py',
            'source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py',
            'source/go1_hier_sim/go1_hier_sim/envs/geometry.py',
            'source/go1_hier_sim/go1_hier_sim/envs/runtime.py',
            'source/go1_hier_sim/go1_hier_sim/envs/config.py',
            'configs/curriculum_phases_respawn_v59.json'):
    new = hashlib.sha256(Path('v23_work/' + key).read_bytes()).hexdigest()
    pat = re.compile(r"'" + re.escape(key) + r"': '([0-9a-f]{64})'")
    m = pat.findall(s)
    if m:
        s = pat.sub("'" + key + "': '" + new + "'", s)
        print('pin', key, m[0][:10], '->', new[:10])
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
