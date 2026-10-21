from pathlib import Path
import hashlib
import re

q = Path('v23_work/scripts/start_v47_privileged.sh')
s = q.read_text(encoding='utf-8')
new = hashlib.sha256(Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/config.py').read_bytes()).hexdigest()
found = re.findall(r"'source/go1_hier_sim/go1_hier_sim/envs/config.py': '([0-9a-f]{64})'", s)
print('pinned', found, 'new', new)
assert len(found) == 1
s = s.replace(found[0], new)
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
