from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None\n'
assert s.count(old) == 1
new = ('    entrance_depth_floor = 0.10 if kind == "combined_soft4" else None\n'
       '    entrance_band = {"combined_soft6": (0.29, 0.42), "combined_soft7": (0.18, 0.26)}.get(kind)\n')
s = s.replace(old, new)

old = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=GAP_WIDTH_MIN * entrance_width_scale,
                                   width_hi=GAP_SHORT_WIDTH_MAX * entrance_width_scale,
                                   depth_floor=entrance_depth_floor)'''
new = '''                cursor = gap_block(cursor, pad=0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=(entrance_band[0] if entrance_band
                                             else GAP_WIDTH_MIN * entrance_width_scale),
                                   width_hi=(entrance_band[1] if entrance_band
                                             else GAP_SHORT_WIDTH_MAX * entrance_width_scale),
                                   depth_floor=entrance_depth_floor)'''
assert s.count(old) == 1
s = s.replace(old, new)

for old, new in [
    ('        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",\n    }:',
     '        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",\n'
     '        "combined_soft6", "combined_soft7",\n    }:'),
    ('''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4"):''',
     '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7"):'''),
    ('''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4"):''',
     '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7"):'''),
    ('''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4"
                ) or "finishing tyres" not in str(error) or attempt == 5:''',
     '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''),
    ('''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
    )
    if mode != "mixed"''',
     '''        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7",
    )
    if mode != "mixed"'''),
]:
    assert s.count(old) == 1, old[:50]
    s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry soft6/7', hashlib.sha256(p.read_bytes()).hexdigest())

p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
reps = [
    ('    "combined", "combined_soft", "combined_soft2", "combined_soft3",\n    "combined_soft4",\n}',
     '    "combined", "combined_soft", "combined_soft2", "combined_soft3",\n'
     '    "combined_soft4", "combined_soft6", "combined_soft7",\n}'),
    ('                           "combined_soft", "combined_soft2", "combined_soft3",\n'
     '                           "combined_soft4"})',
     '                           "combined_soft", "combined_soft2", "combined_soft3",\n'
     '                           "combined_soft4", "combined_soft6", "combined_soft7"})'),
    ('''                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3", "combined_soft4") for name in families],''',
     '''                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3", "combined_soft4", "combined_soft6",
                              "combined_soft7") for name in families],'''),
    ('                credit_families.update({"combined", "combined_soft", "combined_soft2",\n'
     '                                   "combined_soft3", "combined_soft4"})',
     '                credit_families.update({"combined", "combined_soft", "combined_soft2",\n'
     '                                   "combined_soft3", "combined_soft4",\n'
     '                                   "combined_soft6", "combined_soft7"})'),
]
for old, new in reps:
    assert s2.count(old) == 1, old[:50]
    s2 = s2.replace(old, new)
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime soft6/7', hashlib.sha256(p2.read_bytes()).hexdigest())

p3 = Path('v23_work/scripts/run_student_v32.py')
s3 = p3.read_text(encoding='utf-8')
old = '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
    ))
'''
assert s3.count(old) == 1
s3 = s3.replace(old, '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7",
    ))
''')
p3.write_text(s3, encoding='utf-8', newline='\n')
ast.parse(s3)
print('runner soft6/7', hashlib.sha256(p3.read_bytes()).hexdigest())
