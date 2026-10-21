from pathlib import Path
import ast, hashlib

# --- runtime: family sets
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8-sig')
old = '''    "gaps_only", "gap_course", "course_gaps", "gaps_short", "gaps_long",
    "combined", "combined_soft",
}'''
assert s.count(old) == 1
s = s.replace(old, '''    "gaps_only", "gap_course", "course_gaps", "gaps_short", "gaps_long",
    "combined", "combined_soft", "combined_soft2", "combined_soft3",
}''')
old = '''JUMP_FAMILIES = frozenset({"gap_course", "course_gaps", "gaps_long", "combined",
                           "combined_soft"})'''
assert s.count(old) == 1
s = s.replace(old, '''JUMP_FAMILIES = frozenset({"gap_course", "course_gaps", "gaps_long", "combined",
                           "combined_soft", "combined_soft2", "combined_soft3"})''')
old = '                [str(name) in ("combined", "combined_soft") for name in families],'
assert s.count(old) == 1
s = s.replace(old, '''                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3") for name in families],''')
old = '                credit_families.update({"combined", "combined_soft"})'
assert s.count(old) == 1
s = s.replace(old, '                credit_families.update({"combined", "combined_soft", "combined_soft2",\n'
                   '                                   "combined_soft3"})')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())

# --- roster: generic single:<family>
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py')
p2.write_text('''"""Single-family roster for the v32/v33 soft-curriculum combined ladder."""
from __future__ import annotations


def select_training_roster(config: dict, families: tuple[str, ...]) -> tuple[str, ...]:
    roster = str(config.get('training_roster') or '')
    if not roster.startswith('single:'):
        raise ValueError('roster must be single:<family>')
    family = roster.split(':', 1)[1]
    if family not in families:
        raise ValueError(f'{family} missing from registry')
    return (family,)
''', encoding='utf-8', newline='\n')
import importlib.util
ast.parse(p2.read_text(encoding='utf-8'))
print('roster', hashlib.sha256(p2.read_bytes()).hexdigest())

# --- runner: registry extension
p3 = Path('v23_work/scripts/run_student_v32.py')
s3 = p3.read_text(encoding='utf-8')
old = '    families = select_training_roster(config, TERRAIN_FAMILIES + ("combined_soft",))\n'
assert s3.count(old) == 1
s3 = s3.replace(old, '    families = select_training_roster(config, TERRAIN_FAMILIES + (\n'
                     '        "combined_soft", "combined_soft2", "combined_soft3",\n'
                     '    ))\n')
p3.write_text(s3, encoding='utf-8', newline='\n')
ast.parse(s3)
print('runner', hashlib.sha256(p3.read_bytes()).hexdigest())
