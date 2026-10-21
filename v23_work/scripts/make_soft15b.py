from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

steps = []

def rep(old, new, tag):
    global s
    n = s.count(old)
    print(f'{tag}: count={n}')
    if n == 1:
        s = s.replace(old, new)
        steps.append(tag)

rep('"combined_soft13", "combined_soft14",\n    }:',
    '"combined_soft13", "combined_soft14", "combined_soft15",\n    }:', 'whitelist')

rep('    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None\n',
    '    entrance_depth_floor = 0.10 if kind in ("combined_soft4", "combined_soft15") else None\n'
    '    all_void_depth_floor = 0.10 if kind == "combined_soft15" else None\n', 'depth floor vars')

rep('''            for _ in range(3):
                cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                   width_hi=GAP_LONG_WIDTH_MAX, rhythm=True)''',
    '''            for _ in range(3):
                cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                   width_hi=GAP_LONG_WIDTH_MAX, rhythm=True,
                                   depth_floor=all_void_depth_floor)''', 'long gaps floor')

rep('"combined_soft13", "combined_soft14"):', '"combined_soft13", "combined_soft14", "combined_soft15"):', 'alias')
rep('"combined_soft13", "combined_soft14"):', '"combined_soft13", "combined_soft14", "combined_soft15"):', 'alias2')
rep('"combined_soft13", "combined_soft14"\n                ) or',
    '"combined_soft13", "combined_soft14", "combined_soft15"\n                ) or', 'retry guard')
rep('"combined_soft14",\n    )\n    if mode != "mixed"',
    '"combined_soft14", "combined_soft15",\n    )\n    if mode != "mixed"', 'scene whitelist')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('applied', steps)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
