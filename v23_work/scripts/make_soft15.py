from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '"combined_soft9", "gaps_short_dense", "combined_soft12",\n        "combined_soft13", "combined_soft14",\n    }:'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '"combined_soft9", "gaps_short_dense", "combined_soft12",\n'
                   '        "combined_soft13", "combined_soft14", "combined_soft15",\n    }:')

# virtual support floor for every void of the soft15 family
old = '    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None\n'
assert s.count(old) == 1
s = s.replace(old, '    entrance_depth_floor = 0.10 if kind in ("combined_soft4", "combined_soft15") else None\n'
                   '    all_void_depth_floor = 0.10 if kind == "combined_soft15" else None\n')

# long gaps in the course_gaps branch
old = '''            for _ in range(3):
                cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                   width_hi=GAP_LONG_WIDTH_MAX, rhythm=True)'''
assert s.count(old) == 1
s = s.replace(old, '''            for _ in range(3):
                cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                   width_hi=GAP_LONG_WIDTH_MAX, rhythm=True,
                                   depth_floor=all_void_depth_floor)''')

old = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13", "combined_soft14"):'''
assert s.count(old) == 1
s = s.replace(old, '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13", "combined_soft14", "combined_soft15"):''')

old = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9",
                         "combined_soft12", "combined_soft13", "combined_soft14"):'''
assert s.count(old) == 1
s = s.replace(old, '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9",
                         "combined_soft12", "combined_soft13", "combined_soft14",
                         "combined_soft15"):''')

old = '''                        "combined_soft12", "combined_soft13", "combined_soft14"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, '''                        "combined_soft12", "combined_soft13", "combined_soft14",
                        "combined_soft15"
                ) or "finishing tyres" not in str(error) or attempt == 5:''')

old = '        "combined_soft12", "combined_soft13", "combined_soft14",\n    )\n    if mode != "mixed"'
assert s.count(old) == 1
s = s.replace(old, '        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",\n    )\n    if mode != "mixed"')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())

p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
old = '"combined_soft9", "combined_soft12", "combined_soft13",\n}'
assert s2.count(old) == 1, s2.count(old)
s2 = s2.replace(old, '"combined_soft9", "combined_soft12", "combined_soft13",\n    "combined_soft15",\n}')
old = '''                           "combined_soft8", "combined_soft9",
                           "combined_soft12", "combined_soft13"})'''
assert s2.count(old) == 1, s2.count(old)
s2 = s2.replace(old, '''                           "combined_soft8", "combined_soft9",
                           "combined_soft12", "combined_soft13", "combined_soft15"})''')
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime', hashlib.sha256(p2.read_bytes()).hexdigest())
