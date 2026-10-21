from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '    entrance_band = {"combined_soft6": (0.29, 0.42), "combined_soft7": (0.18, 0.26)}.get(kind)\n'
assert s.count(old) == 1
s = s.replace(old, old + '    orig_kind = kind\n')

old = '''            entrance_count = 1 if kind == "combined_soft13" else (2 if kind == "combined_soft14" else 3)
            middle_tire_rows = 4 if difficulty <= 0.30 else 3
            if kind == "combined_soft12":
                # diagnostic: no tyre band right after the entrance voids
                middle_tire_rows = 0'''
assert s.count(old) == 1
s = s.replace(old, '''            entrance_count = (1 if orig_kind == "combined_soft13"
                              else 2 if orig_kind == "combined_soft14" else 3)
            middle_tire_rows = 4 if difficulty <= 0.30 else 3
            if orig_kind == "combined_soft12":
                # diagnostic: no tyre band right after the entrance voids
                middle_tire_rows = 0''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
