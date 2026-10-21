from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = ("            depth = ((GAP_FLOOR_DEPTH if depth_floor is None else float(depth_floor))\n"
       "                   + GAP_DEPTH_SPAN * difficulty + 0.10 * progress)\n"
       "            drop = depth - GAP_FLOOR_DEPTH\n")
assert s.count(old) == 1
new = ("            depth = ((GAP_FLOOR_DEPTH if depth_floor is None else float(depth_floor))\n"
       "                   + GAP_DEPTH_SPAN * difficulty + 0.10 * progress)\n"
       "            drop = 0.0 if depth_floor is not None else depth - GAP_FLOOR_DEPTH\n")
s = s.replace(old, new)
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('drop clamp ok', hashlib.sha256(p.read_bytes()).hexdigest())
