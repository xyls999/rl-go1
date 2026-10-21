#!/usr/bin/env bash
cd /config/Desktop/sim
sed -n '1,60p' scripts/start_v55_multires.sh | grep -anE "assert|already|glob|exists" 
echo '--- run python part with traceback'
/root/env_isaaclab/bin/python - <<'PY'
import hashlib, subprocess
from pathlib import Path
root = Path('/config/Desktop/sim')
sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
expected = {
 'scripts/run_student_v32.py': None,
}
import re
text = (root/'scripts/start_v55_multires.sh').read_text()
for m in re.finditer(r"'([^']+)': '([0-9a-f]{64})'", text):
    p, d = m.group(1), m.group(2)
    try:
        actual = sha(p)
    except Exception as e:
        print('MISSING', p, e); continue
    print(('OK ' if actual == d else 'DRIFT '), p)
PY