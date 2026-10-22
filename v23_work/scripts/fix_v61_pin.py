from pathlib import Path
import hashlib
import re

q = Path('v23_work/scripts/start_v61_penetration2.sh')
s = q.read_text(encoding='utf-8')
new = hashlib.sha256(Path('v23_work/scripts/run_student_v32.py').read_bytes()).hexdigest()
m = re.findall(r"'scripts/run_student_v32.py': '([0-9a-f]{64})'", s)
print('pin', m, '->', new)
assert len(m) == 1
s = s.replace(m[0], new)
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
