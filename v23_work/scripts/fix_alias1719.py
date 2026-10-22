from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '''                  "combined_soft13", "combined_soft14", "combined_soft15"):'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''                  "combined_soft13", "combined_soft14", "combined_soft15",
                  "combined_soft17", "combined_soft18", "combined_soft19"):''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
