from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

# entrance pad override per kind
old = '    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None\n'
assert s.count(old) == 1
s = s.replace(old, old + '    entrance_pad = 1.10 if kind == "combined_soft9" else None\n')

old = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),'''
new = '''                cursor = gap_block(cursor,
                                   pad=entrance_pad if entrance_pad is not None
                                   else 0.65 + (0.15 if soft_entrance else 0.0),'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '        "combined_soft6", "combined_soft7", "combined_soft8",\n    }:'
assert s.count(old) == 1
s = s.replace(old, '        "combined_soft6", "combined_soft7", "combined_soft8",\n        "combined_soft9",\n    }:')

old = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8"):'''
assert s.count(old) == 1
s = s.replace(old, '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9"):''')

old = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8"):'''
assert s.count(old) == 1
s = s.replace(old, '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9"):''')

old = '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7", "combined_soft8"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7", "combined_soft8", "combined_soft9"
                ) or "finishing tyres" not in str(error) or attempt == 5:''')

old = '        "combined_soft6", "combined_soft7", "combined_soft8",\n    )\n    if mode != "mixed"'
assert s.count(old) == 1
s = s.replace(old, '        "combined_soft6", "combined_soft7", "combined_soft8", "combined_soft9",\n    )\n    if mode != "mixed"')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry soft9', hashlib.sha256(p.read_bytes()).hexdigest())

# runtime family lists
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
pairs = [
    ('    "combined_soft4", "combined_soft6", "combined_soft7", "combined_soft8",\n}',
     '    "combined_soft4", "combined_soft6", "combined_soft7", "combined_soft8",\n    "combined_soft9",\n}'),
    ('                           "combined_soft4", "combined_soft6", "combined_soft7",\n'
     '                           "combined_soft8"})',
     '                           "combined_soft4", "combined_soft6", "combined_soft7",\n'
     '                           "combined_soft8", "combined_soft9"})'),
    ('''                              "combined_soft7", "combined_soft8") for name in families],''',
     '''                              "combined_soft7", "combined_soft8", "combined_soft9")
                 for name in families],'''),
]
for old, new in pairs:
    assert s2.count(old) == 1, old[:60]
    s2 = s2.replace(old, new)
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime soft9', hashlib.sha256(p2.read_bytes()).hexdigest())
