from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

pairs = [
    ('"combined_soft13",\n    }:',
     '"combined_soft13", "combined_soft14",\n    }:'),
    ('entrance_count = 1 if kind == "combined_soft13" else 3',
     'entrance_count = 1 if kind == "combined_soft13" else (2 if kind == "combined_soft14" else 3)'),
    ('"combined_soft12", "combined_soft13"):',
     '"combined_soft12", "combined_soft13", "combined_soft14"):'),
    ('"combined_soft12", "combined_soft13"\n                ) or',
     '"combined_soft12", "combined_soft13", "combined_soft14"\n                ) or'),
    ('"combined_soft12", "combined_soft13",\n    )\n    if mode',
     '"combined_soft12", "combined_soft13", "combined_soft14",\n    )\n    if mode'),
]
for old, new in pairs:
    assert s.count(old) == 1, (old[:40], s.count(old))
    s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
