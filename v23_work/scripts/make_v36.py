from pathlib import Path
import ast, hashlib

# 1) geometry: combined_soft8 = exact combined geometry under a training-only name
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = '''        "combined_soft6", "combined_soft7",
    }:'''
assert s.count(old) == 1
s = s.replace(old, '''        "combined_soft6", "combined_soft7", "combined_soft8",
    }:''')
old = '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7"):'''
assert s.count(old) == 1
s = s.replace(old, '''    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8"):''')
old = '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7"):'''
assert s.count(old) == 1
s = s.replace(old, '''        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8"):''')
old = '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7"
                ) or "finishing tyres" not in str(error) or attempt == 5:'''
assert s.count(old) == 1
s = s.replace(old, '''                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7", "combined_soft8"
                ) or "finishing tyres" not in str(error) or attempt == 5:''')
old = '''        "combined_soft6", "combined_soft7",
    )
    if mode != "mixed"'''
assert s.count(old) == 1
s = s.replace(old, '''        "combined_soft6", "combined_soft7", "combined_soft8",
    )
    if mode != "mixed"''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry v36', hashlib.sha256(p.read_bytes()).hexdigest())

# 2) runtime: soft termination mask + lying penalty
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
old = '''    "combined", "combined_soft", "combined_soft2", "combined_soft3",
    "combined_soft4", "combined_soft6", "combined_soft7",
}'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''    "combined", "combined_soft", "combined_soft2", "combined_soft3",
    "combined_soft4", "combined_soft6", "combined_soft7", "combined_soft8",
}''')
old = '''                           "combined_soft4", "combined_soft6", "combined_soft7"})'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''                           "combined_soft4", "combined_soft6", "combined_soft7",
                           "combined_soft8"})''')
old = '''                              "combined_soft3", "combined_soft4", "combined_soft6",
                              "combined_soft7") for name in families],'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''                              "combined_soft3", "combined_soft4", "combined_soft6",
                              "combined_soft7", "combined_soft8") for name in families],''')
old = '''                                   "combined_soft3", "combined_soft4",
                                   "combined_soft6", "combined_soft7"})'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''                                   "combined_soft3", "combined_soft4",
                                   "combined_soft6", "combined_soft7",
                                   "combined_soft8"})''')
# mask init in attach_terrain_task: reuse the block after _combined_family_mask
old = '''            self._combined_family_mask = torch.tensor(
                [str(name) in ("combined", "combined_soft", "combined_soft2",
                              "combined_soft3", "combined_soft4", "combined_soft6",
                              "combined_soft7", "combined_soft8") for name in families],
                dtype=torch.bool, device=self.device,
            )'''
assert s2.count(old) == 1
s2 = s2.replace(old, old + '''
            self._soft_termination_mask = torch.tensor(
                [os.environ.get("GO1_V36_SOFT_TERMINATION") == "1"
                 and str(name) == "combined_soft8" for name in families],
                dtype=torch.bool, device=self.device,
            )''')
# default attr for non-attach paths (termination reads getattr with None default anyway)
old = '        self._v23_tire_mask = None\n'
assert s2.count(old) >= 1
print('runtime patched pre-write')
# lying penalty: extend the v19 integration spot inside _jump_active block
old = '''                terms["takeoff_alignment"] += 25.0 * delta'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''                terms["takeoff_alignment"] += 25.0 * delta
            if getattr(self, "_soft_termination_mask", None) is not None:
                soft_term = self._soft_termination_mask.to(actual.dtype)
                # P10 soft dynamics: falling on the training-only combined_soft8
                # slots is priced per control step instead of ending the episode,
                # so a recovered fall keeps its forward credit and a lying robot
                # pays. Weight matches v11 toml scale for a decisive gradient.
                terms["base_height"] = terms["base_height"] - 2.0 * soft_term * \\
                    self._fallen.to(actual.dtype)''')
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime v36', hashlib.sha256(p2.read_bytes()).hexdigest())

# 3) config.py: suppress fall termination on soft8 envs
p3 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/config.py')
s3 = p3.read_text(encoding='utf-8')
old = '''    return (metrics["fallen"] | metrics["out_of_bounds"]
            | metrics["success"] | metrics["fell_into_gap"])'''
assert s3.count(old) == 1
s3 = s3.replace(old, '''    done = (metrics["fallen"] | metrics["out_of_bounds"]
            | metrics["success"] | metrics["fell_into_gap"])
    soft = getattr(env.hier, "_soft_termination_mask", None)
    if soft is not None and bool(soft.any()):
        done = done & ~(soft & (metrics["fallen"] | metrics["fell_into_gap"]))
    return done''')
p3.write_text(s3, encoding='utf-8', newline='\n')
ast.parse(s3)
print('config v36', hashlib.sha256(p3.read_bytes()).hexdigest())
