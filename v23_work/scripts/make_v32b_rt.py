from pathlib import Path
import hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')

pairs = [
    ('    "gaps_only", "gap_course", "course_gaps", "gaps_short", "gaps_long", "combined",\n}',
     '    "gaps_only", "gap_course", "course_gaps", "gaps_short", "gaps_long",\n'
     '    "combined", "combined_soft",\n}'),
    ('JUMP_FAMILIES = frozenset({"gap_course", "course_gaps", "gaps_long", "combined"})',
     'JUMP_FAMILIES = frozenset({"gap_course", "course_gaps", "gaps_long", "combined",\n'
     '                           "combined_soft"})'),
    ('                [str(name) == "combined" for name in families],',
     '                [str(name) in ("combined", "combined_soft") for name in families],'),
    ('                credit_families.add("combined")',
     '                credit_families.update({"combined", "combined_soft"})'),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
import ast
ast.parse(s)
print('runtime combined_soft ok', hashlib.sha256(p.read_bytes()).hexdigest())
