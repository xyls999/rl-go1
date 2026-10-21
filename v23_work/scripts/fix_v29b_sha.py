from pathlib import Path
p = Path('v23_work/scripts/start_v29b_student_distill.sh')
s = p.read_text()
old = '64f3bcd5d74f616b0b784f98ce3ff8172ff4849fff20a47d3a69ee04c1eaf0d6'
new = '2b089a2949a922dd9a89752c1b5c93a479832bfd85a5384021235761b40a09f1'
assert old in s
s = s.replace(old, new)
p.write_text(s)
p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
import hashlib
print('LAUNCHER29B2', hashlib.sha256(p.read_bytes()).hexdigest())
