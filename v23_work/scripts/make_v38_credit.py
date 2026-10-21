from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py')
s = p.read_text(encoding='utf-8')

old = '''import math

import torch'''
assert s.count(old) == 1
s = s.replace(old, '''import math
import os

import torch''')

old = '''        self.scale = float(scale)
        self.start = start
        self.valid = finite & ((far - start) > .26) & long_mask[:, None]'''
assert s.count(old) == 1
s = s.replace(old, '''        self.scale = float(scale)
        self.start = start
        # v38: the width floor was a fixed .26 m, which silently excluded the
        # narrow opening voids of the mixed route from any take-off credit.
        # The override is training-only and defaults to the published value.
        min_width = float(os.environ.get("GO1_V19_MIN_WIDTH", ".26"))
        self.valid = finite & ((far - start) > min_width) & long_mask[:, None]''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('prelaunch', hashlib.sha256(p.read_bytes()).hexdigest())
