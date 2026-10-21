from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '    soft_entrance = kind == "combined_soft"\n'
assert s.count(old) == 1
new = ('    soft_entrance = kind == "combined_soft"\n'
       '    entrance_width_scale = {"combined_soft2": 0.25, "combined_soft3": 0.6}.get(kind, 1.0)\n')
s = s.replace(old, new)

old = '''        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft",
    }:'''
new = '''        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft", "combined_soft2", "combined_soft3",
    }:'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    elif kind in ("combined", "combined_soft"):'''
new = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3"):'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=GAP_WIDTH_MIN,
                                   width_hi=GAP_SHORT_WIDTH_MAX)'''
new = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=GAP_WIDTH_MIN * entrance_width_scale,
                                   width_hi=GAP_SHORT_WIDTH_MAX * entrance_width_scale)'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        if tile_kind in ("combined", "combined_soft"):'''
new = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3"):'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''                if tile_kind not in ("combined", "combined_soft") or "finishing tyres" not in str(error) or attempt == 5:'''
new = '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2", "combined_soft3"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft",
    )
    if mode != "mixed"'''
new = '''        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft", "combined_soft2", "combined_soft3",
    )
    if mode != "mixed"'''
assert s.count(old) == 1
s = s.replace(old, new)

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry soft2/3 ok', hashlib.sha256(p.read_bytes()).hexdigest())
