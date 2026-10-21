from pathlib import Path
s = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_text(encoding='utf-8')
i = s.find('elif kind in ("combined"')
print(s[i - 160:i + 430])
