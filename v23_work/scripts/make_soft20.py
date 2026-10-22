from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

# whitelist + alias + scene list for soft20 (penetration shim: 6 cm grooves)
old = '"combined_soft17", "combined_soft18", "combined_soft19",\n    }:'
assert s.count(old) == 1
s = s.replace(old, '"combined_soft17", "combined_soft18", "combined_soft19",\n        "combined_soft20",\n    }:')

old = '''                  "combined_soft17", "combined_soft18", "combined_soft19"):'''
assert s.count(old) == 1
s = s.replace(old, '''                  "combined_soft17", "combined_soft18", "combined_soft19",
                  "combined_soft20"):''')

old = '''    entrance_depth_floor = 0.10 if kind in ("combined_soft4", "combined_soft15") else None
    all_void_depth_floor = 0.10 if kind == "combined_soft15" else None'''
assert s.count(old) == 1
s = s.replace(old, '''    entrance_depth_floor = (0.10 if kind in ("combined_soft4", "combined_soft15")
                            else 0.06 if kind == "combined_soft20" else None)
    all_void_depth_floor = (0.10 if kind == "combined_soft15"
                            else 0.06 if kind == "combined_soft20" else None)''')

for old, new in [
    ('"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17",\n'
     '                        "combined_soft18", "combined_soft19"\n                ) or',
     '"combined_soft13", "combined_soft14", "combined_soft15", "combined_soft17",\n'
     '                        "combined_soft18", "combined_soft19", "combined_soft20"\n                ) or'),
    ('"combined_soft14", "combined_soft15", "combined_soft17",\n'
     '        "combined_soft18", "combined_soft19",\n    )\n    if mode != "mixed"',
     '"combined_soft14", "combined_soft15", "combined_soft17",\n'
     '        "combined_soft18", "combined_soft19", "combined_soft20",\n    )\n    if mode != "mixed"'),
]:
    c = s.count(old)
    print('pair', c)
    if c == 1:
        s = s.replace(old, new)

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest(), 'n20', s.count('combined_soft20'))
