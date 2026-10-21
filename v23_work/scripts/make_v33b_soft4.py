from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

# gap_block signature: add depth_floor
old = '''        def gap_block(x0, pad=GAP_LANDING_PAD_M, rhythm=False,
                      width_lo=None, width_hi=None):'''
assert s.count(old) == 1
s = s.replace(old, '''        def gap_block(x0, pad=GAP_LANDING_PAD_M, rhythm=False,
                      width_lo=None, width_hi=None, depth_floor=None):''')

old = '            depth = GAP_FLOOR_DEPTH + GAP_DEPTH_SPAN * difficulty + 0.10 * progress\n'
assert s.count(old) == 1
s = s.replace(old, '            depth = ((GAP_FLOOR_DEPTH if depth_floor is None else float(depth_floor))\n'
                   '                   + GAP_DEPTH_SPAN * difficulty + 0.10 * progress)\n')

old = '''    soft_entrance = kind == "combined_soft"
    entrance_width_scale = {"combined_soft2": 0.25, "combined_soft3": 0.6}.get(kind, 1.0)
'''
assert s.count(old) == 1
s = s.replace(old, '''    soft_entrance = kind == "combined_soft"
    entrance_width_scale = {"combined_soft2": 0.25, "combined_soft3": 0.6}.get(kind, 1.0)
    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None
''')

old = '''        "combined_soft", "combined_soft2", "combined_soft3",
    }:'''
new = '''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
    }:'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3"):'''
new = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4"):'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=GAP_WIDTH_MIN * entrance_width_scale,
                                   width_hi=GAP_SHORT_WIDTH_MAX * entrance_width_scale)'''
new = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=GAP_WIDTH_MIN * entrance_width_scale,
                                   width_hi=GAP_SHORT_WIDTH_MAX * entrance_width_scale,
                                   depth_floor=entrance_depth_floor)'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3"):'''
new = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4"):'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2", "combined_soft3"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
new = '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        "combined_soft", "combined_soft2", "combined_soft3",
    )
    if mode != "mixed"'''
new = '''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
    )
    if mode != "mixed"'''
assert s.count(old) == 1
s = s.replace(old, new)

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry soft4', hashlib.sha256(p.read_bytes()).hexdigest())

# runtime family sets
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
reps = [
    ('    "combined", "combined_soft", "combined_soft2", "combined_soft3",\n}',
     '    "combined", "combined_soft", "combined_soft2", "combined_soft3",\n'
     '    "combined_soft4",\n}'),
    ('                           "combined_soft", "combined_soft2", "combined_soft3"})',
     '                           "combined_soft", "combined_soft2", "combined_soft3",\n'
     '                           "combined_soft4"})'),
    ('''                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3") for name in families],''',
     '''                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3", "combined_soft4") for name in families],'''),
    ('                credit_families.update({"combined", "combined_soft", "combined_soft2",\n'
     '                                   "combined_soft3"})',
     '                credit_families.update({"combined", "combined_soft", "combined_soft2",\n'
     '                                   "combined_soft3", "combined_soft4"})'),
]
for old, new in reps:
    assert s2.count(old) == 1, old[:50]
    s2 = s2.replace(old, new)
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime soft4', hashlib.sha256(p2.read_bytes()).hexdigest())

# runner registry
p3 = Path('v23_work/scripts/run_student_v32.py')
s3 = p3.read_text(encoding='utf-8')
old = '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "combined_soft", "combined_soft2", "combined_soft3",
    ))
'''
assert s3.count(old) == 1
s3 = s3.replace(old, '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
    ))
''')
p3.write_text(s3, encoding='utf-8', newline='\n')
ast.parse(s3)
print('runner soft4', hashlib.sha256(p3.read_bytes()).hexdigest())
