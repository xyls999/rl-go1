from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')
old = '''                [os.environ.get("GO1_V36_SOFT_TERMINATION") == "1"
                 and str(name) == "combined_soft8" for name in families],'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''                [os.environ.get("GO1_V36_SOFT_TERMINATION") == "1"
                 and str(name).startswith("combined_soft") for name in families],''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())
