from pathlib import Path
import re

s = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_text(encoding='utf-8')
for m in re.finditer('combined_soft18', s):
    i = m.start()
    print('...', repr(s[max(0, i - 100):i + 20]))
i = s.find('elif kind in ("combined", "combined_soft"')
print('--- alias block')
print(s[i:i + 430])
