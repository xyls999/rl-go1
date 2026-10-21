from pathlib import Path
import hashlib
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8-sig')
old = """            self._v19_prelaunch_state = PrelaunchGradientState(
                self._v8_gap_bounds,
                torch.tensor([str(name) == \"gaps_long\" for name in families],
                             dtype=torch.bool, device=self.device),
                scale=float(os.environ.get(\"GO1_V19_PRELAUNCH_SCALE\", \"1\")),
            )"""
new = """            credit_families = {\"gaps_long\"}
            if os.environ.get(\"GO1_V19_COMBINED\") == \"1\":
                # Combined opens with three long-class voids on narrow pads;
                # without takeoff credit every episode fails before any partial
                # success gradient exists.
                credit_families.add(\"combined\")
            self._v19_prelaunch_state = PrelaunchGradientState(
                self._v8_gap_bounds,
                torch.tensor([str(name) in credit_families for name in families],
                             dtype=torch.bool, device=self.device),
                scale=float(os.environ.get(\"GO1_V19_PRELAUNCH_SCALE\", \"1\")),
            )"""
assert old in s
s = s.replace(old, new)
p.write_bytes(s.replace('\n', '\r\n').encode('utf-8-sig').replace(b'\r\n', b'\n'))
print('RUNTIME31', hashlib.sha256(p.read_bytes()).hexdigest())
c = Path('v23_work/source/go1_hier_sim/go1_hier_sim/cli.py')
t = c.read_text()
old2 = '"prelaunch_v19" if os.environ.get("GO1_V19_PRELAUNCH_REWARD") == "1" else None,'
new2 = '"prelaunch_v19+combined" if os.environ.get("GO1_V19_COMBINED") == "1" and os.environ.get("GO1_V19_PRELAUNCH_REWARD") == "1" else ("prelaunch_v19" if os.environ.get("GO1_V19_PRELAUNCH_REWARD") == "1" else None),'
assert old2 in t
t = t.replace(old2, new2)
c.write_text(t)
c.write_bytes(c.read_bytes().replace(b'\r\n', b'\n'))
import ast; ast.parse(t); print('CLI31', hashlib.sha256(c.read_bytes()).hexdigest())