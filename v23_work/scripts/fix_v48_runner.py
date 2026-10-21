from pathlib import Path
import ast, hashlib

p = Path('v23_work/scripts/run_student_v32.py')
s = p.read_text(encoding='utf-8')
old = '''        "combined_soft6", "combined_soft7", "combined_soft8",
    ))'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''        "combined_soft6", "combined_soft7", "combined_soft8",
        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",
    ))''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runner', hashlib.sha256(p.read_bytes()).hexdigest())

q = Path('v23_work/scripts/start_v48_soft_dynamics.sh')
t = q.read_text(encoding='utf-8')
import re
found = re.findall(r"'scripts/run_student_v32.py': '([0-9a-f]{64})'", t)
print('pinned', found)
assert len(found) == 1
t = t.replace(found[0], hashlib.sha256(p.read_bytes()).hexdigest())
q.write_text(t, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
