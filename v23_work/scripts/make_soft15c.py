from pathlib import Path
import ast, hashlib

# geometry: tile_kind block for mixed reward segments
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '''                         "combined_soft12", "combined_soft13", "combined_soft14"):'''
n = s.count(old)
print('tile_kind block count', n)
if n == 1:
    s = s.replace(old, '''                         "combined_soft12", "combined_soft13", "combined_soft14",
                         "combined_soft15"):''')
    p.write_text(s, encoding='utf-8', newline='\n')
    ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())

# runtime family sets
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
pairs = [
    ('"combined_soft9", "combined_soft12", "combined_soft13",\n}',
     '"combined_soft9", "combined_soft12", "combined_soft13",\n    "combined_soft15",\n}'),
    ('"combined_soft12", "combined_soft13"})',
     '"combined_soft12", "combined_soft13", "combined_soft15"})'),
]
for old, new in pairs:
    c = s2.count(old)
    print('runtime pair', c, old[:40])
    if c == 1:
        s2 = s2.replace(old, new)
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime', hashlib.sha256(p2.read_bytes()).hexdigest())
