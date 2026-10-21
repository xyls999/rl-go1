from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')
old = '''            means = torch.zeros(self.num_envs, 12, device=self.device)
            valid_flags = [bool(v) for v in valid]
            for name in sorted({n for n, ok in zip(names, valid_flags) if ok and n}):
                ids = torch.tensor([ok and n == name
                                    for n, ok in zip(names, valid_flags)])
                got = self._v29_pool.act(name, self.locomotion_obs, ids.to(self.device))
                means[ids.to(self.device)] = got'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''            means = torch.zeros(self.num_envs, 12, device=self.device)
            valid_flags = [bool(v) for v in valid]
            # Advance every expert's recurrent state on every step, exactly as a
            # real selector would keep all skills running; a cold GRU after a
            # switch produces meaningless actions.
            all_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            means_by_name = {name: self._v29_pool.act(name, self.locomotion_obs, all_ids)
                             for name in self._v29_pool.names}
            for name in sorted({n for n, ok in zip(names, valid_flags) if ok and n}):
                ids = torch.tensor([ok and n == name
                                    for n, ok in zip(names, valid_flags)])
                means[ids.to(self.device)] = means_by_name[name][ids.to(self.device)]''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())
