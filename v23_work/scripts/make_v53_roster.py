from pathlib import Path
import ast, hashlib

# 1) roster: support ladder:combined
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/combined_roster_v32.py')
s = p.read_text(encoding='utf-8')
old = """    roster = str(config.get('training_roster') or '')
    if not roster.startswith('single:'):
        raise ValueError('roster must be single:<family>')"""
assert s.count(old) == 1
s = s.replace(old, """    roster = str(config.get('training_roster') or '')
    if roster == 'ladder:combined':
        # Composition ladder: the mixed route with 0/1/2/3 voids plus the real
        # combined tile. Training sees positive experience at the easy rungs
        # while the acceptance family stays in the mixture.
        ladder = ('course', 'combined_soft13', 'combined_soft14',
                  'combined_soft15', 'combined')
        missing = [f for f in ladder if f not in families]
        if missing:
            raise ValueError(f'ladder families missing from registry: {missing}')
        return ladder
    if not roster.startswith('single:'):
        raise ValueError('roster must be single:<family> or ladder:combined')""")
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('roster', hashlib.sha256(p.read_bytes()).hexdigest())

# 2) runner: allow 'course' in the registry tuple
p2 = Path('v23_work/scripts/run_student_v32.py')
s2 = p2.read_text(encoding='utf-8')
old = '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8",
        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",
    ))'''
assert s2.count(old) == 1, s2.count(old)
s2 = s2.replace(old, '''    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "course", "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8",
        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",
    ))''')
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runner', hashlib.sha256(p2.read_bytes()).hexdigest())
