from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '''                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13"):'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13", "combined_soft14", "combined_soft15"):''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
