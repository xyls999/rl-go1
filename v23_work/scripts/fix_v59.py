from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')

# remove the misplaced enablement block
old = '''            self._soft_mask = self._soft_termination_mask.to(self.device) \\
                if hasattr(self, "_soft_termination_mask") else None
            self._soft_respawn = os.environ.get("GO1_SOFT_RESPAWN") == "1"
            self._soft_termination_mask = torch.tensor('''
if s.count(old) == 1:
    s = s.replace(old, '''            self._soft_termination_mask = torch.tensor(''')

# set the flag at construction, lazily read the mask in the hook
old = '''        self._soft_respawn = False
        self._soft_mask = None'''
assert s.count(old) == 1
s = s.replace(old, '''        self._soft_respawn = os.environ.get("GO1_SOFT_RESPAWN") == "1"''')

old = '''        if self._soft_respawn and self._soft_mask is not None:'''
assert s.count(old) == 1
s = s.replace(old, '''        soft_mask = getattr(self, "_soft_termination_mask", None)
        if self._soft_respawn and soft_mask is not None:''')

old = '''            respawn = newly_fallen & (self._soft_mask > 0.5)'''
assert s.count(old) == 1
s = s.replace(old, '''            respawn = newly_fallen & (soft_mask > 0.5)''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())
print('occurrences soft_mask', s.count('_soft_mask'))
