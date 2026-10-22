from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '"combined_soft13", "combined_soft14", "combined_soft15",\n        "combined_soft17",\n    }:'
assert s.count(old) == 1
s = s.replace(old, '"combined_soft13", "combined_soft14", "combined_soft15",\n'
                   '        "combined_soft17", "combined_soft18", "combined_soft19",\n    }:')

old = '''            shared_riser = float(rng.uniform(0.04, 0.04 + 0.12 * difficulty))'''
assert s.count(old) == 1
s = s.replace(old, '''            shared_riser = float(rng.uniform(0.04, 0.04 + 0.12 * difficulty))
            stair_scale = {"combined_soft18": 0.45, "combined_soft19": 0.70}.get(orig_kind, 1.0)
            shared_riser *= stair_scale''')

for old, new in [
    ('"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17"):',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17",\n'
     '                  "combined_soft18", "combined_soft19"):'),
    ('"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17"\n                ) or',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17",\n'
     '                        "combined_soft18", "combined_soft19"\n                ) or'),
    ('"combined_soft14", "combined_soft15", "combined_soft17",\n    )\n    if mode != "mixed"',
     '"combined_soft14", "combined_soft15", "combined_soft17",\n        "combined_soft18", "combined_soft19",\n    )\n    if mode != "mixed"'),
]:
    c = s.count(old)
    print(c, old[:46].replace('\n', ' '))
    if c == 1:
        s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest(), 'n18', s.count('combined_soft18'), 'n19', s.count('combined_soft19'))
