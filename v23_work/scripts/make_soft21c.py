from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '''    all_void_depth_floor = (0.10 if kind == "combined_soft15"
                            else 0.06 if kind == "combined_soft20" else None)'''
new = '''    all_void_depth_floor = (0.10 if kind == "combined_soft15"
                            else 0.02 if kind == "combined_soft21"
                            else 0.06 if kind == "combined_soft20" else None)'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())

# runtime: entry multiplier override + family lists
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
old = '''        gap_intrusion = (
            intrusion + entering_gap.to(intrusion.dtype) * intrusion_cfg["entry_multiplier"]
        ) * self._gap_family_mask.to(intrusion.dtype)'''
assert s2.count(old) == 1, s2.count(old)
s2 = s2.replace(old, '''        entry_multiplier = float(os.environ.get(
            "GO1_GAP_ENTRY_MULT", str(intrusion_cfg["entry_multiplier"])))
        gap_intrusion = (
            intrusion + entering_gap.to(intrusion.dtype) * entry_multiplier
        ) * self._gap_family_mask.to(intrusion.dtype)''')

for old, new in [
    ('    "combined_soft20",\n}',
     '    "combined_soft20", "combined_soft21",\n}'),
    ('                           "combined_soft20"})',
     '                           "combined_soft20", "combined_soft21"})'),
]:
    c = s2.count(old)
    print('rt', c, old[:40])
    if c == 1:
        s2 = s2.replace(old, new)
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime', hashlib.sha256(p2.read_bytes()).hexdigest(), 'n21', s2.count('combined_soft21'))
