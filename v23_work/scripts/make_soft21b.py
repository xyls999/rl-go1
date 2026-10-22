from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

def rep(old, new, tag):
    global s
    n = s.count(old)
    print(f'{tag}: {n}')
    if n == 1:
        s = s.replace(old, new)
    return n

rep('"combined_soft19",\n        "combined_soft20",\n    }:',
    '"combined_soft19",\n        "combined_soft20", "combined_soft21",\n    }:', 'whitelist')

rep('"combined_soft19",\n                  "combined_soft20"):',
    '"combined_soft19",\n                  "combined_soft20", "combined_soft21"):', 'alias')

rep('else 0.06 if kind == "combined_soft20" else None)\n    all_void_depth_floor',
    'else 0.02 if kind == "combined_soft21" else 0.06 if kind == "combined_soft20" else None)\n'
    '    all_void_depth_floor', 'entrance floor')

rep('else 0.06 if kind == "combined_soft20" else None)\n\n',
    'else 0.02 if kind == "combined_soft21" else 0.06 if kind == "combined_soft20" else None)\n\n',
    'all floor')

rep('"combined_soft20"\n                ) or',
    '"combined_soft20", "combined_soft21"\n                ) or', 'retry guard')

rep('"combined_soft20",\n    )\n    if mode != "mixed"',
    '"combined_soft20", "combined_soft21",\n    )\n    if mode != "mixed"', 'scene list')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest(), 'n21', s.count('combined_soft21'))
