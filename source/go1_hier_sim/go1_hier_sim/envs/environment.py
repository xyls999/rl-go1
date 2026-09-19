"""Manager-based environment with explicit hierarchical control clocks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..contracts import JOINT_NAMES, require_finite
from .runtime import HierarchicalRuntime


class HierarchicalAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, names = self._asset.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"Go1 joint mapping mismatch: {names}")
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 12, device=env.device)
        self._tick = 0

    @property
    def action_dim(self):
        return 12 if self.cfg.stage == "locomotion" else 3

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        require_finite(actions, "policy action")
        self._raw_actions.copy_(actions)
        if self.cfg.stage == "navigation":
            self._env.hier.navigation_action(actions)

    def apply_actions(self):
        runtime = self._env.hier
        runtime.physics_tick(self._tick)
        if self._tick % 4 == 0:
            action = self._raw_actions if self.cfg.stage == "locomotion" else None
            executed = runtime.locomotion_tick(action)
            self._processed_actions.copy_(executed)
        targets = self._asset.data.default_joint_pos[:, self._joint_ids] + 0.25 * self._processed_actions
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        targets = targets.clamp(limits[..., 0], limits[..., 1])
        actual_action = (targets - self._asset.data.default_joint_pos[:, self._joint_ids]) / 0.25
        self._processed_actions.copy_(actual_action)
        runtime.record_executed_action(actual_action)
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)
        self._tick += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._processed_actions[ids] = 0


@configclass
class HierarchicalActionCfg(ActionTermCfg):
    class_type: type = HierarchicalAction
    stage: str = "locomotion"


class HierarchicalEnvironment(ManagerBasedRLEnv):
    def __init__(self, cfg, locomotion_policy=None, render_mode=None):
        self._hier_locomotion_policy = locomotion_policy
        super().__init__(cfg=cfg, render_mode=render_mode)

    def load_managers(self):
        # Observation managers evaluate terms during construction.
        self.hier = HierarchicalRuntime(self, self._hier_locomotion_policy)
        super().load_managers()

    def _reset_idx(self, env_ids):
        if hasattr(self, "hier"):
            metrics = self.hier.get_metrics()
            self.extras["episode_metrics"] = {
                "env_ids": env_ids.clone(),
                **{name: value[env_ids].clone() for name, value in metrics.items()},
            }
        super()._reset_idx(env_ids)
        if hasattr(self, "hier"):
            self.hier.reset(env_ids)


def make_env(stage="locomotion", num_envs=4, device="cuda:0", enable_cameras=False,
             locomotion_policy=None, seed=0, terrain="mixed", purpose="integration",
             episode_seconds=None, render_mode=None, show_direction_arrows=False,
             course_difficulty_min=None, course_difficulty_max=None,
             speed_min=None, speed_max=None, reward_contract=None, robustness=False):
    from .config import make_config

    if stage not in {"locomotion", "navigation"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if purpose not in {"integration", "train"}:
        raise ValueError(f"Unsupported environment purpose: {purpose}")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if stage == "navigation" and locomotion_policy is None:
        raise ValueError("Navigation requires a compatible, explicitly loaded locomotion policy")
    cfg = make_config(stage, num_envs, device, enable_cameras, seed, terrain, purpose,
                      episode_seconds=episode_seconds,
                      show_direction_arrows=show_direction_arrows,
                      course_difficulty_min=course_difficulty_min,
                      course_difficulty_max=course_difficulty_max,
                      speed_min=speed_min, speed_max=speed_max,
                      reward_contract=reward_contract, robustness=robustness)
    return HierarchicalEnvironment(cfg, locomotion_policy, render_mode=render_mode)
