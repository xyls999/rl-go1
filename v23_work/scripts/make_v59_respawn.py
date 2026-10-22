from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s = p.read_text(encoding='utf-8')

# 1) buffers
old = '''        self._fell_into_gap = torch.zeros_like(self._success)'''
assert s.count(old) == 1
s = s.replace(old, '''        self._fell_into_gap = torch.zeros_like(self._success)
        # Soft-dynamics respawn (training only): keep the last upright pose so a
        # failed attempt continues from the lip instead of lying until timeout.
        self._soft_respawn = False
        self._soft_mask = None
        self._safe_pos = self.robot.data.root_link_pos_w.clone()
        self._safe_quat = self.robot.data.root_link_quat_w.clone()
        self._safe_joint = self.robot.data.joint_pos.clone()''')

# 2) hook after newly_fallen is computed
old = '''        newly_fallen &= ~self._fell_into_gap
        if os.environ.get("GO1_DEBUG_FALL") == "1"'''
assert s.count(old) == 1
s = s.replace(old, '''        newly_fallen &= ~self._fell_into_gap
        if self._soft_respawn and self._soft_mask is not None:
            root_z = self.robot.data.root_link_pos_w[:, 2]
            grounded = ~self._fallen & ~newly_fallen & (root_z > 0.22)
            if bool(grounded.any()):
                ids = grounded.nonzero(as_tuple=False).flatten()
                self._safe_pos[ids] = self.robot.data.root_link_pos_w[ids]
                self._safe_quat[ids] = self.robot.data.root_link_quat_w[ids]
                self._safe_joint[ids] = self.robot.data.joint_pos[ids]
            respawn = newly_fallen & (self._soft_mask > 0.5)
            if bool(respawn.any()):
                ids = respawn.nonzero(as_tuple=False).flatten()
                pose = torch.cat((self._safe_pos[ids], self._safe_quat[ids]), dim=-1)
                self.robot.write_root_pose_to_sim(pose, ids)
                self.robot.write_root_velocity_to_sim(
                    torch.zeros(len(ids), 6, device=self.device), ids)
                self.robot.write_joint_state_to_sim(
                    self._safe_joint[ids],
                    torch.zeros(len(ids), self._safe_joint.shape[-1], device=self.device),
                    None, ids)
                newly_fallen = newly_fallen & ~respawn
        if os.environ.get("GO1_DEBUG_FALL") == "1"''')

# 3) enable flag where the soft mask is built
old = '''            self._soft_termination_mask = torch.tensor('''
assert s.count(old) == 1
s = s.replace(old, '''            self._soft_mask = self._soft_termination_mask.to(self.device) \\
                if hasattr(self, "_soft_termination_mask") else None
            self._soft_respawn = os.environ.get("GO1_SOFT_RESPAWN") == "1"
            self._soft_termination_mask = torch.tensor(''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runtime', hashlib.sha256(p.read_bytes()).hexdigest())
