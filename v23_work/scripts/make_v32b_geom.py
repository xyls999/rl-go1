from pathlib import Path

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

# 1) drop the env helper, replace with kind-driven extra
helper = '''def _entrance_pad_extra() -> float:
    # P10-style soft-curriculum knob (v32): widening the recovery pads of the
    # three opening voids teaches the dense triple-gap rhythm on training-only
    # tiles; the default 0 keeps every published terrain byte-identical.
    import os
    return float(os.environ.get("GO1_V32_ENTRANCE_PAD", "0"))


'''
assert helper in s
s = s.replace(helper, '')

old = '''    if kind not in {'''
new = '''    soft_entrance = kind == "combined_soft"
    if kind not in {'''
assert old in s
s = s.replace(old, new, 1)

old = '''        "tire_field", "gaps_short", "gaps_long", "combined",
    }:'''
new = '''        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft",
    }:'''
assert old in s
s = s.replace(old, new, 1)

old = '''    elif kind == "combined":
        # The generic mixed route already packs grades, uneven stairways,
        # rough ground, bumps and leaps across the whole span, so the
        # combination family reuses it instead of a second implementation.
        kind = "course_gaps"'''
new = '''    elif kind in ("combined", "combined_soft"):
        # The generic mixed route already packs grades, uneven stairways,
        # rough ground, bumps and leaps across the whole span, so the
        # combination family reuses it instead of a second implementation.
        # combined_soft (v32 P10 soft-curriculum) is the same route with the
        # three opening recovery pads widened by 0.15 m; it is training-only.
        kind = "course_gaps"'''
assert old in s
s = s.replace(old, new, 1)

old = '''                cursor = gap_block(cursor, pad=0.65 + _entrance_pad_extra(),'''
new = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),'''
assert old in s
s = s.replace(old, new, 1)

old = '''        if tile_kind == "combined":'''
new = '''        if tile_kind in ("combined", "combined_soft"):'''
assert old in s
s = s.replace(old, new, 1)

old = '''    terrain_families = (
        "stairs_up_long", "stairs_down_long", "stairs_mixed",
        "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
        "tire_field", "gaps_short", "gaps_long", "combined",
    )'''
new = '''    terrain_families = (
        "stairs_up_long", "stairs_down_long", "stairs_mixed",
        "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft",
    )'''
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8', newline='\n')
import ast
ast.parse(s)
print('geometry combined_soft ok')
