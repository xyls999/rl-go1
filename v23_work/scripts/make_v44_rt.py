from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')
old = '''    "combined_soft4", "combined_soft6", "combined_soft7", "combined_soft8",
    "combined_soft9",
}'''
assert s.count(old) == 1
s = s.replace(old, '''    "combined_soft4", "combined_soft6", "combined_soft7", "combined_soft8",
    "combined_soft9", "combined_soft12", "combined_soft13",
}''')
old = '''                           "combined_soft8", "combined_soft9"})'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''                           "combined_soft8", "combined_soft9",
                           "combined_soft12", "combined_soft13"})''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())
