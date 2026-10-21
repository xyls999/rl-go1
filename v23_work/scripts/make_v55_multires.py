from pathlib import Path
import ast, hashlib

# ---------- config.py: optional fine scanner ----------
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/config.py')
s = p.read_text(encoding='utf-8-sig')
old = '''    cfg.scene.height_scanner.update_period = CONTROL_DT
    cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
    cfg.scene.height_scanner.debug_vis = False'''
assert s.count(old) == 1
s = s.replace(old, '''    cfg.scene.height_scanner.update_period = CONTROL_DT
    cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
    cfg.scene.height_scanner.debug_vis = False
    if os.environ.get("GO1_MULTIRES_SCAN") == "1":
        # LEEPS-style multi-resolution scan: a 0.05 m patch 0.75 x 1.15 m just
        # ahead of the feet (16 x 24 = 384 samples). The coarse grid is
        # decimated 2x per axis to 8 x 16 = 128, so the terrain vector keeps
        # its published 512-value width.
        cfg.scene.fine_scanner = copy.deepcopy(cfg.scene.height_scanner)
        cfg.scene.fine_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.05, size=(0.75, 1.15), ordering="yx"
        )
        cfg.scene.fine_scanner.offset.pos = (0.55, 0.0, 20.0)
        cfg.scene.fine_scanner.update_period = CONTROL_DT
        cfg.scene.fine_scanner.debug_vis = False''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s.lstrip('\ufeff'))
print('config', hashlib.sha256(p.read_bytes()).hexdigest())

# ---------- runtime.py: fine geometry + merge ----------
p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
old = '''        self._fine_scanner = env.scene["gap_edge_scanner"] if self._v10_observable_leap else None'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''        self._fine_scanner = env.scene["gap_edge_scanner"] if self._v10_observable_leap else None
        self._multires_scan = (os.environ.get("GO1_MULTIRES_SCAN") == "1")
        self._fine_scan = env.scene["fine_scanner"] if self._multires_scan else None''')

old = '''        terrain_value, terrain_valid, _ = self._height_geometry()
        terrain = torch.cat((terrain_value, terrain_valid.to(terrain_value.dtype)), -1)'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''        terrain_value, terrain_valid, _ = self._height_geometry()
        if self._fine_scan is not None:
            fine_value, fine_valid, _ = self._fine_height_geometry()
            coarse_value = terrain_value.reshape(self.num_envs, 16, 32)[:, ::2, ::2]
            coarse_valid = terrain_valid.reshape(self.num_envs, 16, 32)[:, ::2, ::2]
            terrain_value = torch.cat((coarse_value.reshape(self.num_envs, -1),
                                       fine_value.reshape(self.num_envs, -1)), -1)
            terrain_valid = torch.cat((coarse_valid.reshape(self.num_envs, -1),
                                       fine_valid.reshape(self.num_envs, -1)), -1)
        terrain = torch.cat((terrain_value, terrain_valid.to(terrain_value.dtype)), -1)''')

old = '''    def _height_geometry(self):'''
assert s2.count(old) == 1
s2 = s2.replace(old, '''    def _fine_height_geometry(self):
        hits = self._fine_scan.data.ray_hits_w
        dim = 16 * 24
        if hits.shape != (self.num_envs, dim, 3):
            raise ValueError(
                f"Fine scanner must provide [N,{dim},3], got {tuple(hits.shape)}"
            )
        valid = torch.isfinite(hits).all(-1)
        relative = hits[..., 2] - self.robot.data.root_link_pos_w[:, None, 2]
        values = torch.where(valid, relative.clamp(-1.0, 1.0), torch.zeros_like(relative))
        return values, valid, hits

    def _height_geometry(self):''')

p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime', hashlib.sha256(p2.read_bytes()).hexdigest())
