from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '"combined_soft9", "gaps_short_dense", "combined_soft12",\n    }:'
assert s.count(old) == 1
s = s.replace(old, '"combined_soft9", "gaps_short_dense", "combined_soft12",\n'
                   '        "combined_soft13",\n    }:')

old = '''            middle_tire_rows = 4 if difficulty <= 0.30 else 3'''
assert s.count(old) == 1
s = s.replace(old, '''            entrance_count = 1 if kind == "combined_soft13" else 3
            middle_tire_rows = 4 if difficulty <= 0.30 else 3''')

old = '''            cursor = 4.30
            for _ in range(3):
                # These voids are walked, so a shorter pad is safe; long-gap
                # takeoff/landing platforms below remain at 1.10 m.'''
assert s.count(old) == 1
s = s.replace(old, '''            cursor = 4.30
            for _ in range(entrance_count):
                # These voids are walked, so a shorter pad is safe; long-gap
                # takeoff/landing platforms below remain at 1.10 m.''')

old = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9", "combined_soft12"):'''
assert s.count(old) == 1
s = s.replace(old, '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13"):''')

old = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9",
                         "combined_soft12"):'''
assert s.count(old) == 1
s = s.replace(old, '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9",
                         "combined_soft12", "combined_soft13"):''')

old = '''                        "combined_soft7", "combined_soft8", "combined_soft9",
                        "combined_soft12"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, '''                        "combined_soft7", "combined_soft8", "combined_soft9",
                        "combined_soft12", "combined_soft13"
                ) or "finishing tyres" not in str(error) or attempt == 5:''')

old = '        "gap_course", "gaps_short_dense", "combined_flat",\n        "combined_soft12",\n    )\n    if mode != "mixed"'
assert s.count(old) == 1
s = s.replace(old, '        "gap_course", "gaps_short_dense", "combined_flat",\n        "combined_soft12", "combined_soft13",\n    )\n    if mode != "mixed"')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
