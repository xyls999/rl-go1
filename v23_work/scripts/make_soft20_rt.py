from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')
pairs = [
    ('    "combined_soft15", "combined_soft17",\n}',
     '    "combined_soft15", "combined_soft17", "combined_soft18", "combined_soft19",\n'
     '    "combined_soft20",\n}'),
    ('                           "combined_soft12", "combined_soft13", "combined_soft15",\n'
     '                           "combined_soft17"})',
     '                           "combined_soft12", "combined_soft13", "combined_soft15",\n'
     '                           "combined_soft17", "combined_soft18", "combined_soft19",\n'
     '                           "combined_soft20"})'),
]
for old, new in pairs:
    c = s.count(old)
    print(c, old[:44].replace('\n', ' '))
    if c == 1:
        s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest(), 'n20', s.count('combined_soft20'))
