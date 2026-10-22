from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '"combined_soft13", "combined_soft14", "combined_soft15",\n    }:'
assert s.count(old) == 1
s = s.replace(old, '"combined_soft13", "combined_soft14", "combined_soft15",\n'
                   '        "combined_soft17",\n    }:')

old = '''            middle = (lambda x: mixed_tires(x, middle_tire_rows), mixed_grades, mixed_stairs)'''
assert s.count(old) == 1
s = s.replace(old, '''            def flat_middle(x0):
                # Diagnostic: keep the route length but no stairs section.
                return x0 + 2.76

            middle = ((lambda x: mixed_tires(x, middle_tire_rows), mixed_grades,
                       flat_middle if orig_kind == "combined_soft17" else mixed_stairs))''')

for old, new in [
    ('"combined_soft13", "combined_soft14", "combined_soft15"):',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17"):'),
    ('"combined_soft13", "combined_soft14", "combined_soft15"\n                ) or',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17"\n                ) or'),
    ('"combined_soft14", "combined_soft15",\n    )\n    if mode != "mixed"',
     '"combined_soft14", "combined_soft15", "combined_soft17",\n    )\n    if mode != "mixed"'),
    ('"combined_soft13", "combined_soft14", "combined_soft15"):',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17"):'),
]:
    if s.count(old) == 1:
        s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest(), 'soft17 count', s.count('combined_soft17'))
