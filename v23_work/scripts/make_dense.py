from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '''"combined_soft8",
        "combined_soft9",
    }:'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''"combined_soft8",
        "combined_soft9", "gaps_short_dense",
    }:''')

old = '''            if kind == "gaps_short":
                cursor = 4.30
                while cursor < FEATURE_STOP - 2.30:
                    cursor = gap_block(cursor, width_lo=GAP_WIDTH_MIN,
                                       width_hi=GAP_SHORT_WIDTH_MAX)'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''            if kind in ("gaps_short", "gaps_short_dense"):
                cursor = 4.30
                dense = kind == "gaps_short_dense"
                while cursor < FEATURE_STOP - 2.30:
                    cursor = gap_block(cursor,
                                       pad=0.65 if dense else GAP_LANDING_PAD_M,
                                       width_lo=GAP_WIDTH_MIN,
                                       width_hi=GAP_SHORT_WIDTH_MAX)''')

old = '''        "combined_soft6", "combined_soft7", "combined_soft8", "combined_soft9",
        "gap_course",
    )
    if mode != "mixed"'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''        "combined_soft6", "combined_soft7", "combined_soft8", "combined_soft9",
        "gap_course", "gaps_short_dense",
    )
    if mode != "mixed"''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
