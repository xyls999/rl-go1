from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '        "gap_course", "gaps_short_dense",\n    )\n    if mode != "mixed"'
assert s.count(old) == 1
s = s.replace(old, '        "gap_course", "gaps_short_dense", "combined_flat",\n    )\n    if mode != "mixed"')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
