from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = 'if tile_kind != "combined" or "finishing tyres" not in str(error) or attempt == 5:'
assert s.count(old) == 1
s = s.replace(old, 'if tile_kind not in ("combined", "combined_soft") or "finishing tyres" not in str(error) or attempt == 5:')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('SHA', hashlib.sha256(p.read_bytes()).hexdigest())
