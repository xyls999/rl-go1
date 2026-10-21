from pathlib import Path
s = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_text(encoding='utf-8')
for key in ('elif kind in ("combined", "combined_soft"',
            'if tile_kind in ("combined", "combined_soft"',
            'combined_soft14"\n                ) or',
            '"combined_soft14",\n    )\n    if mode',
            '"combined_soft13", "combined_soft14",\n    }:'):
    i = s.find(key)
    print('KEY', repr(key), 'at', i)
    if i >= 0:
        print('   ', repr(s[i:i + 230]))
