from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '''        long_gaps = [block for block in profile.get("blocks", [])
                     if block["kind"] == "gap" and block["width"] > GAP_JUMP_WIDTH_M]'''
assert s.count(old) == 1, s.count(old)
new = '''        import os as _os
        _threshold = float(_os.environ.get("GO1_V8_LONG_GAP_THRESHOLD",
                                           GAP_JUMP_WIDTH_M))
        long_gaps = [block for block in profile.get("blocks", [])
                     if block["kind"] == "gap" and block["width"] > _threshold]'''
s = s.replace(old, new)

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry threshold', hashlib.sha256(p.read_bytes()).hexdigest())
