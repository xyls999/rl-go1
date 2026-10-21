import re
from pathlib import Path

s = Path('v23_work/scripts/v29_smoke4x16.sh').read_text()
s = s.replace('v29_smoke4x16', 'v30_smoke4x16').replace('go1-v29-smoke4x16', 'go1-v30-smoke4x16')
s = s.replace('v27_smoke_plan16.json', 'v30_smoke_plan16.json')
s = re.sub(r'GO1_V29_TEACHER_DISTILL=1 GO1_V29_EPS=\.5 GO1_V29_LAMBDA=\.5 ', '', s)
s = re.sub(r'^export GO1_V29_TEACHERS=.*\n', '', s, flags=re.M)
s = s.replace('GO1_V23_TIRE_BRAKE_FAMILIES=tire_field,combined', 'GO1_V23_TIRE_BRAKE_FAMILIES=combined')
assert 'GO1_V29_TEACHER' not in s and 'v30_smoke_plan16' in s and 'BRAKE_FAMILIES=combined' in s
q = Path('v23_work/scripts/v30_smoke4x16.sh')
q.write_text(s)
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('smoke30 ok')
