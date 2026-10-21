"""Hierarchical control, sensing, teacher geometry, rewards, and episode metrics."""

from __future__ import annotations

from collections import deque
import math
import os

import torch

from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.warp import raycast_mesh

from ..contracts import (
    GAP_FOOT_DEPTH_M,
    CONTROL_DT,
    DEPTH_DT,
    LOCO_OBS_DIM,
    MAX_SENSOR_AGE,
    NAV_DT,
    PHYSICS_DT,
    RELIEF_COLS,
    RELIEF_FORWARD_M,
    RELIEF_LATERAL_M,
    RELIEF_PROBE_HEIGHT,
    RELIEF_ROWS,
    TERRAIN_FEATURE_DIM,
    TERRAIN_HEIGHT_DIM,
    CommandRouter,
    JOINT_NAMES,
    command_from_action,
    depth_to_input,
    fixed_heading_observation,
    gait_phase_observation,
    normalize_command,
    pack_goal,
    pack_proprio,
    rectangular_region_distance,
    require_finite,
    soft_trot_schedule,
    trench_lip_distance,
    world_goal_to_body,
)
from .geometry import COURSE_GOAL_OFFSET, TILE_SIZE, VisibleSurfaceMemory, camera_frustum, yaw_from_quat

# Trunk-versus-terrain contact threshold in newtons, and the number of
# consecutive control steps such contact must persist before the episode is
# treated as failed. A single-step threshold ended 86-93% of the 0.80-1.00
# difficulty episodes at the first stair edge, so the policy was never given a
# chance to learn how to climb over an edge it had already touched. Scraping
# still pays a graded penalty every step; only sustained contact fails.
TRUNK_CONTACT_FORCE = 25.0
COLLISION_SUSTAIN_STEPS = 5
# Minimum control steps between two leap-clear payouts (25 * 0.02 s = 0.5 s).
LEAP_CLEAR_COOLDOWN_STEPS = 25
# Terrain families that actually contain trenches. The leap bonus is only
# meaningful where there is a void to clear.
GAP_FAMILIES = frozenset({"gaps_only", "gap_course", "course_gaps"})
# A trench is wider than the walking stride, so the slow walking band the
# sampler draws for this family cannot carry it: the void needs a take-off
# speed. The forward command is scaled for trench environments instead of
# leaving the policy to disobey the command (which showed up as the achieved
# speed running ~1.8x the commanded speed for the whole stage).
GAP_SPEED_FACTOR = 1.8
from ..hier_v2.rewards_v2 import finish_crossed
from ..hier_v2.relief import relief_map


class HierarchicalRuntime:
    """Owns the clocks and state that do not belong to an Isaac Lab manager.

    The runtime is deliberately stateful: executed commands/actions, the frozen
    low-level GRU, camera visibility history, and episode statistics are all
    reset per environment rather than reconstructed from privileged state.
    """

    _TERRAIN_DIM = TERRAIN_HEIGHT_DIM
    _CAMERA_TICKS = round(DEPTH_DT / PHYSICS_DT)
    _LOCOMOTION_TICKS = round(CONTROL_DT / PHYSICS_DT)
    _NAVIGATION_TICKS = round(NAV_DT / PHYSICS_DT)
    # A leap may not finish before this many control steps: the ordinary
    # walking stance satisfies the landing test, so without a floor the
    # gate frees the leap before the takeoff ever happens.
    _LEAP_HOLD_STEPS = 10
    # Hard ceiling on a single hand-over. The frozen leap policy is a short
    # burst specialist; if its own gait keeps the robot airborne, the landing
    # release test never fires and the walker would never regain control.
    _LEAP_MAX_STEPS = 60

    def __init__(self, env, locomotion_policy=None, jump_policy=None, base_policy=None):
        self.env = env
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]
        self.height_scanner = env.scene["height_scanner"]
        self.contact_sensor = env.scene["contact_forces"]
        self.stage = env.cfg.hier_stage
        self._camera = env.scene.sensors.get("depth_camera") if env.cfg.hier_cameras else None
        self._mesh_prim_path = env.cfg.scene.height_scanner.mesh_prim_paths[0]

        if self._CAMERA_TICKS != 10 or self._LOCOMOTION_TICKS != 4 or self._NAVIGATION_TICKS != 20:
            raise ValueError("The hierarchical clocks must be exact integer multiples of the 200 Hz physics clock")

        self._joint_ids, joint_names = self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != JOINT_NAMES:
            raise ValueError(f"Expected the twelve Go1 joints, found {joint_names}")
        self._foot_ids = []
        self._robot_foot_ids = []
        for leg in ("FL", "FR", "RL", "RR"):
            ids, names = self.contact_sensor.find_bodies(f"{leg}_foot", preserve_order=True)
            if len(ids) != 1:
                raise ValueError(f"Expected one contact body for {leg}_foot, found {names}")
            self._foot_ids.append(ids[0])
            robot_ids, robot_names = self.robot.find_bodies(f"{leg}_foot", preserve_order=True)
            if len(robot_ids) != 1:
                raise ValueError(f"Expected one robot body for {leg}_foot, found {robot_names}")
            self._robot_foot_ids.append(robot_ids[0])
        # Leg links may brush a stair edge while climbing.  That is not a
        # body-fall failure: only contact of the main body with terrain is a
        # collision termination.  Keep the complete sensor list for foot gait
        # terms, but use only torso/base links for failure shaping.
        main_body_names = {"base", "base_link", "trunk", "torso", "pelvis", "abdomen"}
        self._collision_ids = [
            body_id for body_id, name in enumerate(self.contact_sensor.body_names)
            if str(name).lower() in main_body_names
        ]
        if not self._collision_ids:
            fallback = [
                (body_id, str(name))
                for body_id, name in enumerate(self.contact_sensor.body_names)
                if any(token in str(name).lower() for token in ("base", "trunk", "torso"))
            ]
            self._collision_ids = [body_id for body_id, _ in fallback]
        if not self._collision_ids:
            raise ValueError(
                "Could not identify the Go1 main-body contact link; found "
                f"{self.contact_sensor.body_names}"
            )

        zeros3 = torch.zeros(self.num_envs, 3, device=self.device)
        self._command = zeros3.clone()
        self._previous_command = zeros3.clone()
        self._previous_action = torch.zeros(self.num_envs, 12, device=self.device)
        self._action_delta = torch.zeros_like(self._previous_action)
        self._action_acceleration = torch.zeros_like(self._previous_action)
        self._command_delta = zeros3.clone()
        source = "external" if self.stage == "locomotion" else "navigation"
        self._command_router = CommandRouter(self.num_envs, self.device, source=source)

        # hier_v2 contract selection and sim-to-real robustness state.
        self.reward_contract = getattr(env.cfg, "hier_reward_contract", "stage1")
        self._robust = None
        self._nav_v2 = None
        self._nav_v2_terms_cache = {}
        self._nav_v2_last_index = -1
        self._loco_v2_terms_cache = {}
        # A brief stair scrape should shape the policy, not immediately end the episode.
        # Termination uses a short consecutive-contact streak below.
        self._collision_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._collision_update_tick = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._collision_cost = torch.zeros(self.num_envs, device=self.device)
        self._terminal_course_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._nav_v2_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._nav_v2_action_delta = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_delay_slots = 4
        self._command_history = torch.zeros(
            self.num_envs, self._command_delay_slots + 1, 3, device=self.device
        )
        self._observed_command = zeros3.clone()
        self._obs_noise = torch.zeros(self.num_envs, 193, device=self.device)
        if self.reward_contract == "nav_v2":
            from ..hier_v2.rewards_v2 import NavV2Tracker

            self._nav_v2 = NavV2Tracker(
                self.num_envs, self.device, env.cfg.hier_nav_v2_rewards
            )
        self._robust_curriculum = None
        self._robust_updates = 0
        self._robust_step_counter = 0
        self._robust_steps_per_update = 64 if self.stage == "navigation" else 32
        if getattr(env.cfg, "hier_robustness_enabled", False) and self.stage == "navigation":
            from ..hier_v2.robustness_v2 import RobustnessCurriculum

            self._robust_curriculum = RobustnessCurriculum(
                self.num_envs, self.device, seed=int(env.cfg.seed)
            )
            self.set_robustness(self._robust_curriculum.sample(0))

        self._locomotion_policy = locomotion_policy
        self._locomotion_hidden = torch.zeros(1, self.num_envs, 128, device=self.device)
        # The leap hand-over lives here so the training loop and the evaluation
        # loop share one implementation. Both policies are evaluated on the
        # same observation every control tick; only the action switches.
        self._jump_policy = jump_policy
        if jump_policy is not None:
            jump_policy.to(self.device).eval()
            for parameter in jump_policy.parameters():
                parameter.requires_grad_(False)
        self._jump_hidden = torch.zeros(1, self.num_envs, 128, device=self.device)
        self._leap_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._leap_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._leap_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Leap training arrangement: a frozen walking policy owns the robot
        # everywhere except inside a leap, where the policy under training takes
        # over. ``policy_mask`` records which samples actually came from the
        # trained policy so PPO can drop the walker-driven ones instead of
        # crediting the leap actor with another policy's actions.
        self._base_policy = base_policy
        self._base_hidden = torch.zeros(1, self.num_envs, 128, device=self.device)
        self._policy_mask = torch.ones(self.num_envs, device=self.device)
        if base_policy is not None:
            base_policy.to(self.device).eval()
            for parameter in base_policy.parameters():
                parameter.requires_grad_(False)
        if self.stage == "navigation":
            if locomotion_policy is None:
                raise ValueError("Navigation requires a frozen recurrent locomotion policy")
            locomotion_policy.to(self.device).eval()
            for parameter in locomotion_policy.parameters():
                parameter.requires_grad_(False)

        origins = env.scene.env_origins.to(self.device)
        self._forward_axis = torch.zeros(self.num_envs, 2, device=self.device)
        self._forward_axis[:, 0] = 1.0
        self._goal_world = origins.clone()
        self._goal_world[:, 0] += COURSE_GOAL_OFFSET
        self._goal_relative = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_age = torch.zeros(self.num_envs, device=self.device)
        self._episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rng = torch.Generator(device=self.device).manual_seed(int(env.cfg.seed))

        self._terrain_memory = VisibleSurfaceMemory(origins)
        self._obstacle_memory = VisibleSurfaceMemory(origins)
        self._camera_events = deque()
        self._last_depth = None
        self._depth_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._perception = None
        self._drive_terrain = False
        self._drive_obstacle = False
        self._student_terrain = None
        self._student_obstacle = None

        self._terrain_exact = torch.zeros(self.num_envs, TERRAIN_FEATURE_DIM, device=self.device)
        # Named obstacle_* historically; schema 5 made this the 6x12 terrain relief
        # map (gradient + confidence) that the speed layer reads instead of an
        # obstacle range scan.
        self._obstacle_exact = torch.zeros(self.num_envs, 2 * RELIEF_ROWS * RELIEF_COLS, device=self.device)
        self._terrain_observable = self._terrain_exact.clone()
        self._obstacle_observable = self._obstacle_exact.clone()
        self._observable_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.locomotion_obs = torch.zeros(self.num_envs, LOCO_OBS_DIM, device=self.device)
        self.teacher_locomotion_obs = torch.zeros_like(self.locomotion_obs)
        self.navigation_obs = torch.zeros(self.num_envs, 193, device=self.device)
        self.teacher_navigation_obs = torch.zeros_like(self.navigation_obs)
        self._vision_state = torch.zeros(self.num_envs, 45, device=self.device)

        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_forward_position = torch.zeros(self.num_envs, device=self.device)
        self._previous_touchdown_forward = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_touchdown_z = torch.zeros(self.num_envs, 4, device=self.device)
        self._nominal_foot_lateral = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_foot_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._foot_air_duration = torch.zeros(self.num_envs, 4, device=self.device)
        self._swing_peak_clearance = torch.zeros(self.num_envs, 4, device=self.device)
        self._all_feet_contact_duration = torch.zeros(self.num_envs, device=self.device)
        self._lift_history = torch.zeros(self.num_envs, 4, 50, device=self.device)
        self._lift_history_index = 0
        self._frequency_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._gait_phase = torch.full((self.num_envs,), 0.25, device=self.device)
        self._terrain_gate_ema = torch.zeros(self.num_envs, device=self.device)
        self._foot_load_ema = torch.full((self.num_envs, 4), 0.25, device=self.device)
        self._touchdown_count = torch.zeros(self.num_envs, 4, device=self.device)
        self._phase_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._base_height_sum = torch.zeros(self.num_envs, device=self.device)
        self._gait_metric_samples = torch.zeros(self.num_envs, device=self.device)
        self._course_checkpoints = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
        # LP-ACRL stage attribution: the undiscounted stage-1 return and step
        # count of the episode each environment is currently running, plus the
        # finished-episode records drained by the stage report.
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stage_records: list[dict] = []
        self._no_progress_time = torch.zeros(self.num_envs, device=self.device)
        self._xy_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._velocity_samples = torch.zeros(self.num_envs, device=self.device)
        # Outer-loop speed feasibility: mean commanded vs mean realised forward
        # speed per episode, so the achievable band can be measured, not guessed.
        self._command_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._achieved_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros_like(self._success)
        self._fallen = torch.zeros_like(self._success)
        self._out_of_bounds = torch.zeros_like(self._success)
        self._fell_into_gap = torch.zeros_like(self._success)
        self._gap_foot_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Last control step's trench-intrusion fraction, used to detect the
        # single step where any part of the robot first enters a void.
        self._gap_intrusion = torch.zeros(self.num_envs, device=self.device)
        # Whether the previous control step already paid the leap-cleared bonus.
        self._leap_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._leap_cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Which environments are on a trench terrain, taken from the LP-ACRL
        # stage plan. Without a plan every environment keeps the old behaviour.
        terrain_cfg = env.cfg.scene.terrain
        families = getattr(terrain_cfg, "terrain_families", None)
        if families is None:
            families = getattr(getattr(terrain_cfg, "terrain_generator", None),
                               "terrain_families", None)
        if families is None:
            self._gap_family_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            if len(families) != self.num_envs:
                raise ValueError(
                    f"terrain family list has {len(families)} entries, expected {self.num_envs}"
                )
            self._gap_family_mask = torch.tensor(
                [str(name) in GAP_FAMILIES for name in families],
                dtype=torch.bool, device=self.device,
            )
        # Jump rewards are only paid inside the window that opens when a trench
        # enters the leap gate. Flight on open ground therefore earns nothing,
        # so a policy cannot farm air time by hopping on a flat landing pad.
        self._JUMP_REWARD_WINDOW_STEPS = 60
        self._trench_tick = -1
        self._trench_trigger = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trench_lip = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._jump_reward_window = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_tick = -1
        self._reward_cache_tick = -1
        self._locomotion_reward_cache = {}
        terrain_names = ("flat", "slopes", "rough", "stairs", "course", "gap_course", "course_gaps")
        kinds = torch.arange(self.num_envs, device=self.device) % 4 if env.cfg.hier_terrain == "mixed" else torch.full(
            (self.num_envs,), terrain_names.index(env.cfg.hier_terrain), device=self.device
        )
        self._terrain_kind = kinds.to(torch.long)
        # The jump skill activates on the dedicated gap course: airborne
        # suppressors flip sign and the four jump rewards turn on. The plain
        # course keeps the walking gait and its suppressors bit-for-bit.
        self._jump_active = env.cfg.hier_terrain in ("gap_course", "course_gaps")

    @property
    def camera_events_pending(self) -> int:
        return len(self._camera_events)

    @property
    def current_tick(self) -> int:
        """The next physics tick that will be executed by the action term."""
        return self._last_tick + 1

    @property
    def physics_tick_count(self) -> int:
        return self.current_tick

    def _warp_mesh(self):
        mesh = RayCaster.meshes.get(self._mesh_prim_path)
        if mesh is None:
            raise RuntimeError(f"Unified terrain ray-query mesh is unavailable: {self._mesh_prim_path}")
        return mesh

    def _desired_stage1_command(self):
        command = self.env.command_manager.get_command("base_velocity")
        if command.shape[-1] < 3:
            raise ValueError("Isaac Lab base_velocity command does not contain vx, vy, wz")
        desired = command[:, :3].clone()
        desired[:, :2] *= self._course_speed_scale()[:, None]
        desired[:, 0] = torch.where(
            self._gap_family_mask, desired[:, 0] * GAP_SPEED_FACTOR, desired[:, 0]
        )
        return desired

    def _course_speed_scale(self):
        """Return a smooth speed cap while the ground ahead drops away.

        The direction stays the fixed world-forward direction; only the commanded
        magnitude is reduced. The gate reads the height scan instead of hard-coded
        course positions, so it follows any course layout: the scan points in
        front of the robot are compared with the ground under it.
        """
        if getattr(self.env.cfg, "hier_terrain", "") != "course":
            return torch.ones(self.num_envs, device=self.device)
        _, valid, points = self._height_geometry()
        root = self.robot.data.root_link_pos_w
        forward = points[..., 0] - root[:, None, 0]
        lateral = (points[..., 1] - root[:, None, 1]).abs()
        under = (forward.abs() < 0.45) & (lateral < 0.60) & valid
        ahead = (forward > 0.50) & (forward < 2.00) & (lateral < 0.60) & valid
        under_count = under.sum(-1)
        ahead_count = ahead.sum(-1)
        ground = points[..., 2]
        reference = (ground * under).sum(-1) / under_count.clamp_min(1)
        upcoming = (ground * ahead).sum(-1) / ahead_count.clamp_min(1)
        drop = (reference - upcoming).clamp(min=0.0)
        gate = (drop / 0.25).clamp(0.0, 1.0)
        return torch.where(
            (under_count > 0) & (ahead_count > 0),
            1.0 - 0.55 * gate,
            torch.ones_like(gate),
        )

    def _update_uwb(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        root = self.robot.data.root_link_pos_w[ids]
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w[ids])
        relative = world_goal_to_body(root[:, :2], self._goal_world[ids, :2], yaw)
        if self._robust is None:
            noise_std = float(self.env.cfg.hier_uwb_noise_std)
            if noise_std:
                relative = relative + torch.randn(
                    relative.shape, generator=self._rng, device=self.device
                ) * noise_std
            dropout = torch.rand(
                root.shape[0], generator=self._rng, device=self.device
            ) < float(self.env.cfg.hier_uwb_dropout_probability)
        else:
            noise_std = self._robust.uwb_noise_std[ids][:, None]
            if bool((noise_std > 0).any()):
                relative = relative + torch.randn(
                    relative.shape, generator=self._rng, device=self.device
                ) * noise_std
            dropout = torch.rand(
                root.shape[0], generator=self._rng, device=self.device
            ) < self._robust.uwb_dropout[ids]
        self._goal_relative[ids] = relative
        self._goal_valid[ids] = ~dropout
        self._goal_age[ids] = 0

    def physics_tick(self, tick: int):
        """Advance sensor ages and enqueue every real 20 Hz camera event."""
        if tick <= self._last_tick:
            raise RuntimeError(f"Physics tick must increase monotonically ({tick} after {self._last_tick})")
        self._last_tick = tick
        self._goal_age.add_(PHYSICS_DT)
        if self.stage == "navigation" and tick % self._NAVIGATION_TICKS == 0:
            self._update_uwb()
        if self._camera is not None and tick % self._CAMERA_TICKS == 0:
            self._capture_camera_event(tick)

    def navigation_action(self, action: torch.Tensor):
        if self.stage != "navigation":
            raise RuntimeError("A navigation action cannot own commands in locomotion stage")
        requested = command_from_action(action)
        if self._robust is not None:
            requested = requested * self._robust.command_gain
        usable = self._goal_valid & (self._goal_age <= MAX_SENSOR_AGE)
        previous = self._command.clone()
        self._previous_command.copy_(previous)
        routed = self._command_router.update(
            requested, self._goal_age, usable, NAV_DT, source="navigation"
        )
        self._command.copy_(routed.velocity)
        self._update_observed_command()
        self._command_delta.copy_(self._observed_command - previous)
        clipped = action.clamp(-1.0, 1.0)
        self._nav_v2_action_delta.copy_(clipped - self._nav_v2_action)
        self._nav_v2_action.copy_(clipped)
        self._advance_robustness()

    def _update_observed_command(self):
        """Delay the command the frozen locomotion policy sees, not the target."""
        self._command_history = torch.roll(self._command_history, shifts=1, dims=1)
        self._command_history[:, 0] = self._command
        if self._robust is None:
            self._observed_command.copy_(self._command)
            return
        delay = self._robust.action_delay_steps.clamp(0, self._command_delay_slots)
        index = delay[:, None, None].expand(-1, 1, 3)
        self._observed_command.copy_(torch.gather(self._command_history, 1, index).squeeze(1))

    def _advance_robustness(self):
        """Advance the curriculum one population update every rollout window."""
        if self._robust_curriculum is None:
            return
        self._robust_step_counter += 1
        if self._robust_step_counter < self._robust_steps_per_update:
            return
        self._robust_step_counter = 0
        self._robust_updates += 1
        self.set_robustness(self._robust_curriculum.sample(self._robust_updates))

    def set_robustness(self, batch):
        """Install one curriculum sample; ``None`` restores clean behaviour."""
        self._robust = batch
        if batch is None:
            self._obs_noise.zero_()
            self._observed_command.copy_(self._command)
            return
        std = batch.obs_noise_std[:, None]
        if bool((std > 0).any()):
            self._obs_noise.copy_(
                torch.randn(self.num_envs, 193, generator=self._rng, device=self.device) * std
            )
        else:
            self._obs_noise.zero_()
        self._update_observed_command()

    def _call_locomotion_policy(self, observation):
        with torch.no_grad():
            result = self._locomotion_policy(observation, self._locomotion_hidden)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("The frozen locomotion policy must return (action, next_hidden)")
        action, hidden = result
        if action.shape != (self.num_envs, 12) or hidden.shape != (1, self.num_envs, 128):
            raise ValueError("The frozen locomotion policy returned incompatible action or hidden state")
        require_finite(action, "locomotion policy action")
        require_finite(hidden, "locomotion policy hidden state")
        self._locomotion_hidden.copy_(hidden)
        return action

    def locomotion_tick(self, action: torch.Tensor | None):
        """Run the 50 Hz command/action path and return clipped executed action."""
        if self.stage == "locomotion":
            if action is None or action.shape != (self.num_envs, 12):
                raise ValueError("Stage 1 requires one 12-D action per environment")
            desired = self._desired_stage1_command()
            previous_command = self._command.clone()
            routed = self._command_router.update(
                desired, torch.zeros(self.num_envs, device=self.device),
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                CONTROL_DT, source="external",
            )
            self._command.copy_(routed.velocity)
            self._command_delta.copy_(self._command - previous_command)
            executed = action.clamp(-3.0, 3.0)
        else:
            self.refresh()
            executed = self._call_locomotion_policy(self.locomotion_obs).clamp(-3.0, 3.0)
        previous_delta = self._action_delta.clone()
        self._action_delta.copy_(executed - self._previous_action)
        self._action_acceleration.copy_(self._action_delta - previous_delta)
        phase_cfg = self.env.cfg.hier_stage1_rewards["phase_contact"]
        raw_terrain_gate = self._terrain_complexity_gate()
        # Smooth the sensitive max-edge detector so one scan cannot abruptly
        # switch the gait contract and trigger a high-step/twist response.
        self._terrain_gate_ema.mul_(0.95).add_(raw_terrain_gate * 0.05)
        terrain_gate = self._terrain_gate_ema
        gait_hz = phase_cfg["flat_hz"] + terrain_gate * (
            phase_cfg["terrain_hz"] - phase_cfg["flat_hz"]
        )
        moving = torch.linalg.vector_norm(self._command[:, :2], dim=-1) > 0.10
        self._gait_phase.copy_(torch.remainder(
            self._gait_phase + gait_hz * CONTROL_DT * moving.to(gait_hz.dtype), 1.0
        ))
        return executed

    def record_executed_action(self, action: torch.Tensor):
        if action.shape != self._previous_action.shape:
            raise ValueError("Executed action shape does not match the twelve Go1 joints")
        require_finite(action, "executed joint action")
        self._previous_action.copy_(action)

    def _height_geometry(self):
        hits = self.height_scanner.data.ray_hits_w
        if hits.shape != (self.num_envs, self._TERRAIN_DIM, 3):
            raise ValueError(
                f"Height scanner must provide [N,{self._TERRAIN_DIM},3], got {tuple(hits.shape)}"
            )
        valid = torch.isfinite(hits).all(-1)
        relative = hits[..., 2] - self.robot.data.root_link_pos_w[:, None, 2]
        values = torch.where(valid, relative.clamp(-1.0, 1.0), torch.zeros_like(relative))
        points = torch.where(valid[..., None], hits, self.robot.data.root_link_pos_w[:, None, :])
        return values, valid, points

    def gate_action(self, walk_action):
        """Hand a latched leap over to the frozen jump policy.

        Returns the walk action unchanged when no leap is active, so a
        plain walking environment behaves exactly as before.
        """
        if self._jump_policy is None and self._base_policy is None:
            return walk_action
        trigger, _lip = self.trench_state()
        if os.environ.get("GO1_DEBUG_GATE") == "1":
            print("GATESTEP", "tick=%d" % self.current_tick,
                  "x=%.3f" % float(self.robot.data.root_link_pos_w[0, 0]),
                  "z=%.3f" % float(self.robot.data.root_link_pos_w[0, 2]),
                  "trigger=%s" % bool(trigger[0]),
                  "lip=%.3f" % float(_lip[0]),
                  "leap=%s" % bool(self._leap_mode[0]),
                  flush=True)
        entering = trigger & ~self._leap_mode
        self._leap_mode |= trigger
        self._leap_age = torch.where(
            entering,
            torch.zeros_like(self._leap_age),
            self._leap_age + self._leap_mode.to(self._leap_age.dtype),
        )
        self._leap_hold = torch.where(
            entering,
            torch.full_like(self._leap_hold, self._LEAP_HOLD_STEPS),
            torch.clamp(self._leap_hold - 1, min=0),
        )
        force = self.contact_sensor.data.net_forces_w[:, self._foot_ids]
        contacts = torch.linalg.vector_norm(force, dim=-1) > 1.0
        landed = contacts.all(dim=-1) & (self.robot.data.root_lin_vel_b[:, 2].abs() < 0.25)
        self._leap_mode &= ~((landed & (self._leap_hold == 0))
                              | (self._leap_age >= self._LEAP_MAX_STEPS))
        if self._base_policy is None:
            # Deployment: the frozen leap policy replaces the walking policy
            # that produced this action for as long as the gate stays latched.
            # Its recurrent state still advances every control step so the
            # hand-over starts from the recent observation history; resuming
            # from a zero state instead makes the leap policy behave as if it
            # had just spawned, and the leap never happens.
            with torch.no_grad():
                jump_action, jump_hidden = self._jump_policy(
                    self.locomotion_obs, self._jump_hidden)
                self._jump_hidden = jump_hidden
            # The trained walking policy's action is discarded while the frozen
            # leap policy is latched, so those samples must not contribute a
            # gradient to the walker (it would be learning from actions the
            # environment never applied). The critic still sees every state.
            self._policy_mask.copy_((~self._leap_mode).to(self._policy_mask.dtype))
            if not bool(self._leap_mode.any()):
                return walk_action
            return torch.where(self._leap_mode[:, None], jump_action, walk_action)
        # Leap training: the frozen walking policy drives everywhere except the
        # leap, and the policy under training supplies the action only while the
        # gate is latched.
        with torch.no_grad():
            base_action, base_hidden = self._base_policy(
                self.locomotion_obs, self._base_hidden)
            self._base_hidden = base_hidden
        self._policy_mask.copy_(self._leap_mode.to(self._policy_mask.dtype))
        return torch.where(self._leap_mode[:, None], walk_action, base_action)

    @property
    def policy_mask(self):
        """Per-environment 1.0 where the last action came from the trained policy."""
        return self._policy_mask

    def trench_state(self):
        """Per-control-step trench trigger/lip plus the jump reward window.

        Cached on the control tick so the gate and the reward manager share
        one detector evaluation, and the window advances exactly once per
        step. It stays closed on terrains without trenches.
        """
        tick = self.current_tick
        if self._trench_tick != tick:
            trigger, lip = self.gap_leap_trigger()
            self._trench_tick = tick
            self._trench_trigger = trigger
            self._trench_lip = lip
            if bool(trigger.any()):
                self._jump_reward_window = torch.where(
                    trigger,
                    torch.full_like(self._jump_reward_window, self._JUMP_REWARD_WINDOW_STEPS),
                    self._jump_reward_window,
                )
            self._jump_reward_window = (self._jump_reward_window - 1).clamp(min=0)
        if os.environ.get("GO1_DEBUG_GATE") == "1" and self.current_tick % 10 == 0:
            print("TSTATE", "tick=%d" % self.current_tick,
                  "x=%.3f" % float(self.robot.data.root_link_pos_w[0, 0]),
                  "z=%.3f" % float(self.robot.data.root_link_pos_w[0, 2]),
                  "vx=%.3f" % float(self.robot.data.root_lin_vel_w[0, 0]),
                  "lip=%.3f" % float(self._trench_lip[0]),
                  "trigger=%s" % bool(self._trench_trigger[0]),
                  "window=%d" % int(self._jump_reward_window[0]),
                  flush=True)
        return self._trench_trigger, self._trench_lip

    def gap_leap_trigger(self):
        """Rule-C trench gate from the live height scan.

        Returns (trigger, lip_distance) per environment. Empty (never
        triggered) on terrains without trenches, so the gate is inert on the
        plain course.
        """
        inert = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                 torch.full((self.num_envs,), float("inf"), device=self.device))
        if getattr(self.env.cfg, "hier_terrain", "") not in ("course_gaps", "gap_course"):
            return inert
        values, valid, points = self._height_geometry()
        pattern = self.height_scanner.cfg.pattern_cfg
        rows = int(math.floor(pattern.size[0] / pattern.resolution + 1.0e-6)) + 1
        cols = int(math.floor(pattern.size[1] / pattern.resolution + 1.0e-6)) + 1
        if rows * cols != self._TERRAIN_DIM:
            return inert
        nan = float("nan")
        # This pattern is yx-ordered, so the flat index is i_x * rows + i_y:
        # the first axis spans the sensor's forward axis and the second the
        # lateral one. The per-forward-column profile therefore medians
        # across the lateral axis (dim=-1) without any transpose.
        z = points[..., 2].reshape(self.num_envs, rows, cols)
        x = points[..., 0].reshape(self.num_envs, rows, cols)
        ok = valid.reshape(self.num_envs, rows, cols)
        heights = torch.where(ok, z, torch.full_like(z, nan))
        xs = torch.where(ok, x, torch.full_like(x, nan))
        column_heights = torch.nanmedian(heights, dim=-1).values
        column_x = torch.nanmedian(xs, dim=-1).values
        column_valid = torch.isfinite(column_heights)
        forward = column_x - self.robot.data.root_link_pos_w[:, None, 0]
        forward = torch.where(torch.isfinite(forward), forward, torch.zeros_like(forward))
        if os.environ.get("GO1_DEBUG_SCAN") == "1" and self.current_tick % 25 == 0:
            row_z = column_heights[0].detach().cpu().numpy()
            row_f = forward[0].detach().cpu().numpy()
            row_v = column_valid[0].detach().cpu().numpy()
            print("SCANCOLS", "tick=%d" % self.current_tick,
                  "root_z=%.3f" % float(self.robot.data.root_link_pos_w[0, 2]),
                  "cols=%d" % int(row_v.sum()),
                  "fwd=" + ",".join("%.2f" % f for f in row_f[:6]),
                  "z=" + ",".join(("%.2f" % zz) if vv else "nan"
                                   for zz, vv in zip(row_z[:12], row_v[:12])),
                  "minz=%.2f" % float(row_z[row_v].min() if row_v.any() else 0.0),
                  "maxz=%.2f" % float(row_z[row_v].max() if row_v.any() else 0.0),
                  flush=True)
        return trench_lip_distance(column_heights, column_valid, forward)

    def _terrain_complexity_gate(self):
        """Softly identify local step/rough-terrain edges from the height scan.

        Flat-ground gait timing is useful, but forcing it at a stair edge can
        prevent the deliberate three-foot support phases needed for climbing.
        This gate is intentionally continuous so rewards do not jump at the
        flat/stair boundary.
        """
        values, valid, _ = self._height_geometry()
        pattern = self.height_scanner.cfg.pattern_cfg
        rows = int(math.floor(pattern.size[0] / pattern.resolution + 1.0e-6)) + 1
        columns = int(math.floor(pattern.size[1] / pattern.resolution + 1.0e-6)) + 1
        if rows * columns != self._TERRAIN_DIM:
            return torch.zeros(self.num_envs, device=self.device)
        grid = values.reshape(self.num_envs, rows, columns)
        mask = valid.reshape(self.num_envs, rows, columns)
        dx_valid = mask[:, :, 1:] & mask[:, :, :-1]
        dy_valid = mask[:, 1:, :] & mask[:, :-1, :]
        dx = torch.where(dx_valid, (grid[:, :, 1:] - grid[:, :, :-1]).abs(), 0.0)
        dy = torch.where(dy_valid, (grid[:, 1:, :] - grid[:, :-1, :]).abs(), 0.0)
        # A stair edge occupies only about three percent of this 16 x 32 scan.
        # V13 used an 85th percentile on an incorrectly assumed square grid,
        # which made the gate identically zero for all 512-ray observations.
        # The maximum adjacent height change is the correct local obstacle
        # statistic here because ray hits are deterministic terrain surfaces.
        edge_strength = torch.cat((dx.flatten(1), dy.flatten(1)), dim=1).amax(dim=1)
        return torch.sigmoid((edge_strength - 0.04) / 0.015)

    def _relief_geometry(self):
        """Ground gradient ahead, as the schema-5 6x12 terrain relief map.

        This replaced the obstacle range scan once the speed layer stopped doing
        avoidance. Rows are look-ahead distances along the frozen course axis and
        columns are lateral offsets. Each cell reports the ground gradient
        ``(hit_z - reference_z) / distance`` clamped to +/-1, so a 0.16 m step
        reads 0.27 from 0.6 m away and 0.13 from 1.2 m away: the policy can tell
        a step from a slope. A cell with no hit is unknown space, never free
        space, and gets relief 0 with confidence 0.
        """
        forward = self._forward_axis
        lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        rows = torch.tensor(RELIEF_FORWARD_M, device=self.device)
        columns = torch.linspace(-RELIEF_LATERAL_M, RELIEF_LATERAL_M, RELIEF_COLS, device=self.device)
        root = self.robot.data.root_link_pos_w
        planar = (
            root[:, None, None, :2]
            + rows[None, :, None, None] * forward[:, None, None, :]
            + columns[None, None, :, None] * lateral[:, None, None, :]
        )
        probe_z = (root[:, 2] + RELIEF_PROBE_HEIGHT)[:, None, None, None].expand(
            -1, RELIEF_ROWS, RELIEF_COLS, 1
        )
        origins = torch.cat((planar, probe_z), dim=-1).reshape(self.num_envs, -1, 3)
        directions = torch.zeros_like(origins)
        directions[..., 2] = -1.0
        hits, _, _, face_id = raycast_mesh(
            origins, directions, self._warp_mesh(), max_dist=RELIEF_PROBE_HEIGHT + 2.5,
            return_distance=True, return_face_id=True,
        )
        hit_z = hits[..., 2].view(self.num_envs, RELIEF_ROWS, RELIEF_COLS)
        face_hit = (face_id >= 0).view(self.num_envs, RELIEF_ROWS, RELIEF_COLS)
        gradient, valid_float = relief_map(hit_z, face_hit)
        # A missed probe comes back non-finite; keep those points finite before
        # they reach the camera frustum and the visibility memory, which would
        # otherwise propagate NaN through the observable relief path as well.
        safe_hits = torch.where(
            torch.isfinite(hits).all(-1, keepdim=True),
            hits,
            root[:, None, :].expand_as(hits),
        )
        return gradient, valid_float, safe_hits

    def _occlusion_visible(self, points, in_frustum, camera_position):
        delta = points - camera_position[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1.0e-6)[..., None]
        origins = camera_position[:, None, :].expand_as(points) + direction * 0.01
        _, hit_distance, _, _ = raycast_mesh(
            origins, direction, self._warp_mesh(), max_dist=6.0, return_distance=True
        )
        unobstructed = (~torch.isfinite(hit_distance)) | (hit_distance >= distance - 0.04)
        return in_frustum & (distance >= 0.2) & (distance <= 6.0) & unobstructed

    def _observable_geometry(self, timestamp: float):
        terrain_value, terrain_valid, terrain_points = self._height_geometry()
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w
        terrain_frustum, camera_pos = camera_frustum(terrain_points, root_pos, root_quat)
        terrain_visible = terrain_valid & self._occlusion_visible(terrain_points, terrain_frustum, camera_pos)
        self._terrain_memory.update(terrain_points, terrain_visible, timestamp)
        terrain_known = terrain_valid & self._terrain_memory.query(terrain_points, timestamp)
        terrain = torch.cat((terrain_value, terrain_known.to(terrain_value.dtype)), -1)
        if self.stage == "locomotion":
            return terrain, torch.zeros_like(self._obstacle_observable)
        relief_value, _, relief_points = self._relief_geometry()
        relief_frustum, _ = camera_frustum(relief_points, root_pos, root_quat)
        relief_visible = self._occlusion_visible(relief_points, relief_frustum, camera_pos)
        self._obstacle_memory.update(relief_points, relief_visible, timestamp)
        relief_known = self._obstacle_memory.query(relief_points, timestamp, height_tolerance=0.15)
        return terrain, torch.cat((relief_value, relief_known.to(relief_value.dtype)), -1)

    def _capture_camera_event(self, tick: int):
        if len(self._camera_events) >= 64:
            raise RuntimeError("Camera event consumer fell more than 3.2 seconds behind")
        raw_depth = self._camera.data.output.get("distance_to_image_plane")
        if raw_depth is None:
            raise RuntimeError("The configured depth camera produced no distance_to_image_plane frame")
        image = depth_to_input(raw_depth)
        # The first frame must be able to build its proprioceptive state before
        # student geometry exists. No action is evaluated between this refresh
        # and the provider update below.
        self._refresh_observations(allow_unready_student=True)
        timestamp = tick * PHYSICS_DT
        terrain, obstacle = self._observable_geometry(timestamp)
        self._terrain_observable.copy_(terrain)
        self._obstacle_observable.copy_(obstacle)
        self._observable_ready.fill_(True)
        self._last_depth = image
        self._depth_ready.fill_(True)
        if self._perception is not None:
            self._perception.update_camera(image, self._vision_state)
            if self._drive_terrain:
                self._student_terrain = self._perception.terrain()
            if self._drive_obstacle:
                self._student_obstacle = self._perception.obstacle()
        self._camera_events.append({
            "tick": int(tick),
            "time": float(timestamp),
            "image": image.clone(),
            "state": self._vision_state.clone(),
            "terrain": terrain.clone(),
            "obstacle": obstacle.clone(),
            "episode_id": self._episode_id.clone(),
            "reset_mask": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        })

    def consume_camera_event(self):
        return self._camera_events.popleft() if self._camera_events else None

    def depth_input(self):
        if self._camera is None:
            raise RuntimeError("Depth input requested from an environment without a camera")
        if self._last_depth is None or not self._depth_ready.all():
            raise RuntimeError("Depth input requested before the first real camera sample")
        return self._last_depth

    def vision_state(self):
        self.refresh()
        return self._vision_state

    def teacher_geometry(self, kind: str, observable: bool = True):
        self.refresh()
        if observable and not self._observable_ready.all():
            raise RuntimeError("Observable teacher geometry requested before a camera sample after reset")
        if kind == "terrain":
            return self._terrain_observable if observable else self._terrain_exact
        if kind == "obstacle":
            return self._obstacle_observable if observable else self._obstacle_exact
        raise ValueError(f"Unknown geometry kind: {kind}")

    def set_perception(self, provider, drive_terrain=False, drive_obstacle=False):
        required = ("update_camera", "terrain", "obstacle", "reset")
        if any(not callable(getattr(provider, name, None)) for name in required):
            raise TypeError(f"Perception provider must implement {required}")
        drive_terrain, drive_obstacle = bool(drive_terrain), bool(drive_obstacle)
        if drive_terrain or drive_obstacle:
            ready = getattr(provider, "ready", None)
            if not isinstance(ready, torch.Tensor) or ready.shape != (self.num_envs,) or not ready.bool().all():
                raise RuntimeError("Student-driven perception requires a real camera update for every environment")
        self._perception = provider
        self._drive_terrain = drive_terrain
        self._drive_obstacle = drive_obstacle
        self._student_terrain = provider.terrain() if drive_terrain else None
        self._student_obstacle = provider.obstacle() if drive_obstacle else None

    def clear_perception(self):
        self._perception = None
        self._drive_terrain = self._drive_obstacle = False
        self._student_terrain = self._student_obstacle = None

    def refresh(self):
        self._refresh_observations(allow_unready_student=False)

    def _refresh_observations(self, allow_unready_student: bool):
        data = self.robot.data
        qrel = data.joint_pos[:, self._joint_ids] - data.default_joint_pos[:, self._joint_ids]
        proprio = pack_proprio(
            data.root_ang_vel_b, data.projected_gravity_b, qrel,
            data.joint_vel[:, self._joint_ids], self._previous_action,
        )
        # The frozen locomotion policy follows the (optionally delayed) command
        # that the navigation layer actually issued, not the reward target.
        command = normalize_command(self._observed_command if self.stage == "navigation"
                                    else self._command)
        self._vision_state = torch.cat((proprio, command), -1)

        terrain_value, terrain_valid, _ = self._height_geometry()
        terrain = torch.cat((terrain_value, terrain_valid.to(terrain_value.dtype)), -1)
        if self.stage == "navigation":
            relief_value, relief_valid, _ = self._relief_geometry()
            obstacle = torch.cat((relief_value, relief_valid.to(relief_value.dtype)), -1)
        else:
            # Navigation geometry is outside the Stage-1 interface, so avoid
            # compiling and running its ray query in locomotion environments.
            obstacle = torch.zeros_like(self._obstacle_exact)
        self._terrain_exact.copy_(terrain)
        self._obstacle_exact.copy_(obstacle)

        live_terrain = terrain
        live_obstacle = obstacle
        if self._drive_terrain:
            if self._student_terrain is None:
                if not allow_unready_student:
                    raise RuntimeError("Student terrain geometry requested before a camera update")
            else:
                live_terrain = self._student_terrain
        if self._drive_obstacle:
            if self._student_obstacle is None:
                if not allow_unready_student:
                    raise RuntimeError("Student obstacle geometry requested before a camera update")
            else:
                live_obstacle = self._student_obstacle
        for name, value, expected in (
            ("live terrain geometry", live_terrain, (self.num_envs, TERRAIN_FEATURE_DIM)),
            ("live relief geometry", live_obstacle, (self.num_envs, 2 * RELIEF_ROWS * RELIEF_COLS)),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
            require_finite(value, name)
        goal = pack_goal(self._goal_relative, self._goal_valid, self._goal_age)
        previous_command = normalize_command(self._previous_command)

        phase = gait_phase_observation(self._gait_phase)
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w)
        fixed_heading = fixed_heading_observation(self._forward_axis, yaw)
        loco_prefix = torch.cat((proprio, command, phase, fixed_heading), -1)
        nav_prefix = torch.cat((proprio, goal, previous_command), -1)
        self.teacher_locomotion_obs = torch.cat((loco_prefix, terrain), -1)
        self.locomotion_obs = torch.cat((loco_prefix, live_terrain), -1)
        self.teacher_navigation_obs = torch.cat((nav_prefix, obstacle), -1)
        self.navigation_obs = torch.cat((nav_prefix, live_obstacle), -1)
        if self._robust is not None:
            # Actor-only sensor corruption; the critic keeps clean geometry.
            self.navigation_obs = self.navigation_obs + self._obs_noise

    def privileged_state(self):
        force = self.contact_sensor.data.net_forces_w[:, self._foot_ids]
        contacts = torch.linalg.vector_norm(force, dim=-1) > 1.0
        return torch.cat((self.robot.data.root_lin_vel_b, contacts.to(force.dtype)), -1)

    def _update_failures(self):
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        # Trench falls are classified before the generic fall: any foot below
        # the datum means it is inside a void the course never offers ground in
        # (the floor sits 0.30+ m down), and that is its own failure mode. The
        # generic fall (body low or tilted) would otherwise latch first while
        # the body pitches over the lip.
        # Either one foot this far below the other feet (a void beside a ledge) or
        # the whole robot this far below the datum (it is on a trench floor).
        # Both are measured from the feet, never the trunk: on a stair descent the
        # trunk height alone plus a 0.20 m riser already exceeds a trunk-relative
        # threshold, and that test used to end normal episodes on flat lead-ins.
        foot_z = self.robot.data.body_pos_w[:, self._robot_foot_ids, 2]
        foot_spread = foot_z.amin(dim=-1) - foot_z.amax(dim=-1)
        submerged = foot_z.amax(dim=-1) < -GAP_FOOT_DEPTH_M
        feet_below_datum = (foot_spread > GAP_FOOT_DEPTH_M) | submerged
        next_streak = torch.where(
            feet_below_datum, self._gap_foot_streak + 1, torch.zeros_like(self._gap_foot_streak)
        )
        self._gap_foot_streak.copy_(next_streak)
        gap_falls = (self._gap_foot_streak >= 5) & ~self._fell_into_gap
        if os.environ.get("GO1_DEBUG_GAP") == "1" and bool(gap_falls.any()):
            for env_id in torch.nonzero(gap_falls, as_tuple=False).flatten().tolist():
                local = (self.robot.data.root_link_pos_w[env_id, :2]
                         - self.env.scene.env_origins[env_id, :2])
                print("GAP_FALL", "env=%d" % env_id,
                      "course_x=%.2f" % float(local[0].item()),
                      "root_x=%.2f" % float(self.robot.data.root_link_pos_w[env_id, 0].item()),
                      "root_z=%.3f" % float(self.robot.data.root_link_pos_w[env_id, 2].item()),
                      "foot_zmin=%.3f" % float(self.robot.data.body_pos_w[env_id, self._robot_foot_ids, 2].amin().item()),
                      "origin_x=%.2f" % float(self.env.scene.env_origins[env_id, 0].item()),
                      "step=%d" % self.current_tick,
                      flush=True)
        self._fell_into_gap |= gap_falls
        newly_fallen = ((self.robot.data.root_link_pos_w[:, 2] < 0.18) | (gravity_z > -0.5))
        newly_fallen &= ~self._fell_into_gap
        if os.environ.get("GO1_DEBUG_FALL") == "1" and bool((newly_fallen & ~self._fallen).any()):
            for env_id in torch.nonzero(newly_fallen & ~self._fallen, as_tuple=False).flatten().tolist():
                local = (self.robot.data.root_link_pos_w[env_id, :2]
                         - self.env.scene.env_origins[env_id, :2])
                print("FALL_EVENT", "env=%d" % env_id,
                      "course_x=%.2f" % float(local[0].item()),
                      "z=%.3f" % float(self.robot.data.root_link_pos_w[env_id, 2].item()),
                      "gravity_z=%.3f" % float(gravity_z[env_id].item()),
                      "speed=%.2f" % float(torch.linalg.vector_norm(
                          self.robot.data.root_lin_vel_b[env_id, :2]).item()),
                      flush=True)
        self._fallen |= newly_fallen
        if self._collision_ids:
            forces = self.contact_sensor.data.net_forces_w[:, self._collision_ids]
            force_norms = torch.linalg.vector_norm(forces, dim=-1)
            peak = force_norms.amax(-1)
            touching = peak > TRUNK_CONTACT_FORCE
            # One call per control step (the reward build is cached per tick), so
            # the counter is in control steps: COLLISION_SUSTAIN_STEPS * 0.02 s.
            self._collision_streak.copy_(
                torch.where(touching, self._collision_streak + 1,
                            torch.zeros_like(self._collision_streak))
            )
            self._collision_cost.copy_(
                ((peak - TRUNK_CONTACT_FORCE) / TRUNK_CONTACT_FORCE).clamp(min=0.0, max=2.0)
            )
            self._collision |= self._collision_streak >= COLLISION_SUSTAIN_STEPS
            if os.environ.get("GO1_DEBUG_COLLISION") == "1":
                failed = self._collision_streak == COLLISION_SUSTAIN_STEPS
                for env_id in torch.nonzero(failed, as_tuple=False).flatten().tolist():
                    rows = sorted(
                        ((self.contact_sensor.body_names[body_id],
                          float(force_norms[env_id, column].item()))
                         for column, body_id in enumerate(self._collision_ids)
                         if force_norms[env_id, column] > TRUNK_CONTACT_FORCE),
                        key=lambda row: row[1], reverse=True,
                    )
                    print("COLLISION_EVENT", "env=%d" % env_id,
                          "course_x=%.2f course_y=%.2f" % (
                              float(self.robot.data.root_link_pos_w[env_id, 0].item()
                                    - self.env.scene.env_origins[env_id, 0].item()),
                              float(self.robot.data.root_link_pos_w[env_id, 1].item()
                                    - self.env.scene.env_origins[env_id, 1].item()),
                          ),
                          "speed=%.2f" % float(torch.linalg.vector_norm(
                              self.robot.data.root_lin_vel_b[env_id, :2]).item()),
                          [(name, round(force, 2)) for name, force in rows[:8]], flush=True)
        local = self.robot.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        # Cache the live course position while the episode is still running.
        # The manager teleports the robot back to its spawn before this runtime
        # is reset, so the position has to be captured here to describe where
        # the episode actually ended.
        self._terminal_course_xy.copy_(local)
        self._out_of_bounds |= (local[:, 0] < -2.8) | (local[:, 0] > TILE_SIZE - 3.2) | (local[:, 1].abs() > 7.8)
        # A trench is unrecoverable: the floor sits 0.30 m below the datum, well
        # below the 0.16 m riser envelope, so any body below the datum top means
        # the robot stepped into the gap. This is its own failure mode, stricter
        # than out_of_bounds, and never reachable on the plain course (the datum
        # stays at z = 0 there).
        self._fell_into_gap |= self.robot.data.root_link_pos_w[:, 2] < -0.15
        if self.stage == "locomotion":
            goal = self.env.cfg.hier_stage1_rewards["goal_reached"]
            forward = local[:, 0]
            lateral = local[:, 1]
            # Distance to the rectangular finish region, not to an artificial
            # centre point. It is exactly zero everywhere inside the strip.
            distance = rectangular_region_distance(
                forward, lateral, goal["start_offset"], goal["end_offset"], goal["half_width"]
            )
            reached = distance == 0.0
        else:
            distance = torch.linalg.vector_norm(
                self._goal_world[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=-1
            )
            # Crossing the finish line finishes the task. The line belongs to the
            # navigation contract because the locomotion contract still describes
            # the short legacy course, and using it here fired completion right
            # after the stairs. Overshoot counts and the whole route width is
            # valid, so no standstill or centre-line alignment is required.
            nav_rewards = getattr(self.env.cfg, "hier_nav_v2_rewards", None) or {}
            finish = nav_rewards.get("reach_goal") or self.env.cfg.hier_stage1_rewards["goal_reached"]
            reached = finish_crossed(
                local[:, 0], local[:, 1], finish["start_offset"], finish["half_width"]
            )
        self._success |= reached & self._goal_valid
        return distance

    def _build_locomotion_reward_terms(self):
        if self.stage != "locomotion":
            raise RuntimeError("locomotion rewards are unavailable in navigation stage")
        if self._reward_cache_tick == self.current_tick:
            return self._locomotion_reward_cache
        distance = self._update_failures()
        actual = self.robot.data.root_lin_vel_b
        world_velocity = self.robot.data.root_lin_vel_w[:, :2]
        lateral_axis = torch.stack((-self._forward_axis[:, 1], self._forward_axis[:, 0]), dim=-1)
        target_world_velocity = (
            self._command[:, :1] * self._forward_axis
            + self._command[:, 1:2] * lateral_axis
        )
        # Stage-1 commands are defined in the initial world-frame course basis,
        # not in the robot's continually rotating body frame.
        xy_error = (world_velocity - target_world_velocity).square().sum(-1)
        yaw_error = (self.robot.data.root_ang_vel_b[:, 2] - self._command[:, 2]).square()
        self._xy_error_sum += xy_error
        self._yaw_error_sum += yaw_error
        self._velocity_samples += 1
        self._command_speed_sum += self._command[:, 0]
        upright = (-self.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        cfg = self.env.cfg.hier_stage1_rewards
        air_cfg = cfg["feet_air_time"]
        world_forward_velocity = (world_velocity * self._forward_axis).sum(-1)
        self._achieved_speed_sum += world_forward_velocity
        world_lateral_velocity = (world_velocity * lateral_axis).sum(-1)
        horizontal_speed = torch.linalg.vector_norm(world_velocity, dim=-1)
        root_xy = self.robot.data.root_link_pos_w[:, :2]
        local_xy = root_xy - self.env.scene.env_origins[:, :2]
        cross_track_error = (local_xy * lateral_axis).sum(-1)
        first_contact = self.contact_sensor.compute_first_contact(CONTROL_DT)[:, self._foot_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self._foot_ids]
        last_contact_time = self.contact_sensor.data.last_contact_time[:, self._foot_ids]
        current_air_time = self.contact_sensor.data.current_air_time[:, self._foot_ids]
        current_contact_time = self.contact_sensor.data.current_contact_time[:, self._foot_ids]
        capped_air_time = last_air_time.clamp(max=air_cfg["max_air_time"])
        moving = ((torch.linalg.vector_norm(self._command[:, :2], dim=-1) > air_cfg["command_threshold"])
                  | (self._command[:, 2].abs() > air_cfg["command_threshold"]))
        feet_air_time = ((capped_air_time - air_cfg["threshold"]) * first_contact).sum(-1) * moving
        force_history = self.contact_sensor.data.net_forces_w_history[:, :, self._foot_ids]
        contacts_history = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1) > cfg["feet_slide"]["contact_force_threshold"]
        current_contacts = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:, self._foot_ids], dim=-1) > cfg["feet_slide"]["contact_force_threshold"]
        contacts = current_contacts
        foot_xy_speed = torch.linalg.vector_norm(self.robot.data.body_lin_vel_w[:, self._robot_foot_ids, :2], dim=-1)
        foot_world_z = self.robot.data.body_pos_w[:, self._robot_foot_ids, 2]
        foot_swing = ~contacts
        # Measure swing clearance from the highest supporting foot rather than
        # from this foot's previous touchdown.  The latter folds stair height
        # into clearance and can reward a leg merely for being held above a
        # lower step.  This support-plane estimate follows the FWWB design.
        support_candidates = torch.where(
            contacts, foot_world_z, torch.full_like(foot_world_z, -torch.inf)
        )
        support_z = support_candidates.amax(dim=-1)
        support_z = torch.where(torch.isfinite(support_z), support_z, foot_world_z.amax(dim=-1))
        foot_clearance = foot_world_z - support_z[:, None]
        contact_count = contacts.sum(dim=-1)
        diagonal_support = (contacts[:, 0] & contacts[:, 3]) | (contacts[:, 1] & contacts[:, 2])
        exactly_two_contacts = contact_count == 2
        terrain_gate = self._terrain_gate_ema
        commanded_motion_gate = moving.to(actual.dtype)
        # Gait and alignment bonuses assist real locomotion; they are not a
        # second objective that may be farmed by marching in place.
        forward_motion_gate = (
            world_forward_velocity / (2.0 * cfg["velocity_direction"]["min_speed"])
        ).clamp(0.0, 1.0)

        # Diagonal trot: front-left is coupled to rear-right, and front-right
        # to rear-left.  V13 multiplied six timing scores; one imperfect foot
        # collapsed the whole signal and the contact pattern itself was only
        # weakly constrained.  V14 uses an additive score for usable gradients
        # plus an explicit mismatch penalty below.
        gait_cfg = cfg["trot_gait"]

        def sync_score(foot_0, foot_1):
            air_error = (current_air_time[:, foot_0] - current_air_time[:, foot_1]).square()
            contact_error = (current_contact_time[:, foot_0] - current_contact_time[:, foot_1]).square()
            error = air_error.clamp(max=gait_cfg["max_err"] ** 2)
            error += contact_error.clamp(max=gait_cfg["max_err"] ** 2)
            return torch.exp(-error / gait_cfg["std"])

        def async_score(foot_0, foot_1):
            error_0 = (current_air_time[:, foot_0] - current_contact_time[:, foot_1]).square()
            error_1 = (current_contact_time[:, foot_0] - current_air_time[:, foot_1]).square()
            error = error_0.clamp(max=gait_cfg["max_err"] ** 2)
            error += error_1.clamp(max=gait_cfg["max_err"] ** 2)
            return torch.exp(-error / gait_cfg["std"])

        diagonal_sync_timing = 0.5 * (sync_score(0, 3) + sync_score(1, 2))
        diagonal_anti_timing = 0.25 * (
            async_score(0, 1) + async_score(0, 2)
            + async_score(3, 1) + async_score(3, 2)
        )
        exact_diagonal_support = (
            (contacts[:, 0] & contacts[:, 3] & ~contacts[:, 1] & ~contacts[:, 2])
            | (contacts[:, 1] & contacts[:, 2] & ~contacts[:, 0] & ~contacts[:, 3])
        ).to(actual.dtype)
        active_support_phase = ((contact_count > 0) & (contact_count < 4)).to(actual.dtype)
        trot_timing_score = (
            0.35 * diagonal_sync_timing
            + 0.25 * diagonal_anti_timing
            + 0.40 * exact_diagonal_support
        ) * active_support_phase
        trot_timing_score *= commanded_motion_gate * forward_motion_gate
        trot_timing_score *= 1.0 - gait_cfg["terrain_relief"] * terrain_gate
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w)
        body_forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        body_heading_error = torch.atan2(
            body_forward[:, 0] * self._forward_axis[:, 1]
            - body_forward[:, 1] * self._forward_axis[:, 0],
            (body_forward * self._forward_axis).sum(-1),
        )
        root_xy = self.robot.data.root_link_pos_w[:, :2]
        foot_delta = self.robot.data.body_pos_w[:, self._robot_foot_ids, :2] - root_xy[:, None, :]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        foot_lateral = -sine[:, None] * foot_delta[..., 0] + cosine[:, None] * foot_delta[..., 1]
        lateral_delta = foot_lateral.abs() - self._nominal_foot_lateral.abs()
        lateral_penalty = (
            (lateral_delta - cfg["foot_lateral"]["max_outward_deviation"]).clamp(min=0.0)
            + (-lateral_delta - cfg["foot_lateral"]["max_inward_deviation"]).clamp(min=0.0)
            + (foot_lateral * self._nominal_foot_lateral < 0).to(actual.dtype) * cfg["foot_lateral"]["cross_body_penalty"]
        ).sum(-1)
        forward_position = (root_xy * self._forward_axis).sum(-1)
        forward_delta = forward_position - self._previous_forward_position
        # Speed-neutral progress: the reference is the commanded speed, so moving
        # at the command earns the full term and moving faster earns no extra. With a
        # fixed reference the low level learns "always run flat out" and ignores the
        # upper layer's commanded speed entirely (measured: 0.93 m/s for every command).
        progress_reference = self._command[:, 0].clamp(
            min=cfg["velocity_direction"]["min_speed"]
        )
        progress_ratio = forward_delta / (CONTROL_DT * progress_reference)
        # Speed-neutral forward credit (saturates at 1) plus an asymmetric penalty for
        # moving backwards relative to the command: the old clamp(-1, 1) form supplied
        # that brake, and removing it let the policy drift backwards and laterally.
        forward_progress = progress_ratio.clamp(0.0, 1.0) + 2.0 * progress_ratio.clamp(-1.0, 0.0)
        self._previous_forward_position.copy_(forward_position)
        max_distance = cfg["goal_closeness"]["max_distance"]
        normalized_distance = (1.0 - distance / max_distance).clamp(0.0, 1.0)
        fractions = ((self.robot.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]) * self._forward_axis).sum(-1) / COURSE_GOAL_OFFSET
        checkpoint_cfg = cfg["course_checkpoint"]
        checkpoint_hits = torch.stack((fractions >= checkpoint_cfg["first"], fractions >= checkpoint_cfg["second"], fractions >= checkpoint_cfg["third"]), dim=-1)
        new_checkpoints = checkpoint_hits & ~self._course_checkpoints
        self._course_checkpoints |= checkpoint_hits
        checkpoint_reward = new_checkpoints.to(actual.dtype).sum(-1)
        lift_events = self._last_foot_contacts & ~current_contacts
        touchdown_events = ~self._last_foot_contacts & current_contacts
        # Clearance is rewarded once, at touchdown, rather than every frame in
        # the air.  This removes the profitable failure mode where one leg is
        # held up indefinitely to collect a continuous clearance reward.
        swing_peak_clearance = torch.maximum(
            self._swing_peak_clearance, foot_clearance.clamp(min=0.0)
        )
        touchdown_peak_clearance = swing_peak_clearance.clone()
        self._swing_peak_clearance.copy_(torch.where(
            foot_swing, swing_peak_clearance, torch.zeros_like(swing_peak_clearance)
        ))
        self._foot_air_duration.copy_(torch.where(
            current_contacts,
            torch.zeros_like(self._foot_air_duration),
            self._foot_air_duration + CONTROL_DT,
        ))
        all_feet_contact = current_contacts.all(dim=-1)
        self._all_feet_contact_duration.copy_(torch.where(
            all_feet_contact,
            self._all_feet_contact_duration + CONTROL_DT,
            torch.zeros_like(self._all_feet_contact_duration),
        ))
        foot_forward_position = (self.robot.data.body_pos_w[:, self._robot_foot_ids, :2] * self._forward_axis[:, None, :]).sum(-1)
        step_lengths = foot_forward_position - self._previous_touchdown_forward
        self._previous_touchdown_forward = torch.where(touchdown_events, foot_forward_position, self._previous_touchdown_forward)
        self._previous_touchdown_z = torch.where(touchdown_events, foot_world_z, self._previous_touchdown_z)
        self._last_foot_contacts.copy_(current_contacts)
        self._touchdown_count.add_(touchdown_events.to(actual.dtype))
        self._lift_history[:, :, self._lift_history_index] = lift_events.to(actual.dtype)
        self._lift_history_index = (self._lift_history_index + 1) % self._lift_history.shape[-1]
        frequency_window = cfg["step_frequency"]["window_seconds"]
        self._frequency_elapsed.add_(CONTROL_DT).clamp_(max=frequency_window)
        lift_frequency = self._lift_history.sum(dim=-1) / self._frequency_elapsed.clamp_min(CONTROL_DT)[:, None]
        moving_gate = (world_forward_velocity / cfg["trot_gait"]["velocity_threshold"]).clamp(0.0, 1.0)
        variance_cfg = cfg["air_time_variance"]
        completed_air = last_air_time.clamp(max=variance_cfg["max_time"])
        completed_contact = last_contact_time.clamp(max=variance_cfg["max_time"])
        air_time_variance = completed_air.var(dim=-1, unbiased=False)
        air_time_variance += completed_contact.var(dim=-1, unbiased=False)
        air_time_variance *= commanded_motion_gate
        air_time_variance *= 1.0 - variance_cfg["terrain_relief"] * terrain_gate
        pattern_cfg = cfg["flat_trot_pattern"]
        diagonal_pair_mismatch = 0.5 * (
            (contacts[:, 0] != contacts[:, 3]).to(actual.dtype)
            + (contacts[:, 1] != contacts[:, 2]).to(actual.dtype)
        )
        non_diagonal_two_support = (
            exactly_two_contacts & ~diagonal_support
        ).to(actual.dtype)
        flat_trot_pattern = (
            diagonal_pair_mismatch + non_diagonal_two_support
        ) * commanded_motion_gate
        flat_trot_pattern *= 1.0 - pattern_cfg["terrain_relief"] * terrain_gate

        # Explicit soft trot clock: FL+RR and FR+RL form the two diagonal
        # groups.  Soft transition bands permit temporary stance extension on
        # stair edges instead of hard-locking a joint trajectory.
        phase_cfg = cfg["phase_contact"]
        duty_factor = phase_cfg["flat_duty_factor"] + terrain_gate * (
            phase_cfg["terrain_duty_factor"] - phase_cfg["flat_duty_factor"]
        )
        desired_contact, swing_progress = soft_trot_schedule(
            self._gait_phase, duty_factor, phase_cfg["transition_width"]
        )
        contact_float = current_contacts.to(actual.dtype)
        force_norm = torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w[:, self._foot_ids], dim=-1
        )
        swing_force_score = torch.exp(-torch.square(force_norm / phase_cfg["force_scale"]))
        stance_slip_score = torch.exp(-torch.square(foot_xy_speed / phase_cfg["slip_scale"]))
        phase_contact_quality = (
            desired_contact * (0.55 * contact_float + 0.45 * stance_slip_score)
            + (1.0 - desired_contact)
            * (0.55 * (1.0 - contact_float) + 0.45 * swing_force_score)
        ).mean(dim=-1) * commanded_motion_gate * forward_motion_gate

        phase_clearance_cfg = cfg["phase_clearance"]
        phase_clearance_height = phase_clearance_cfg["flat_height"] + terrain_gate * (
            phase_clearance_cfg["terrain_height"] - phase_clearance_cfg["flat_height"]
        )
        swing_envelope = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        desired_clearance = phase_clearance_height[:, None] * swing_envelope
        clearance_weight = (1.0 - desired_contact) * swing_envelope.square()
        phase_clearance_per_foot = torch.exp(-torch.square(
            (foot_clearance - desired_clearance) / phase_clearance_cfg["std"]
        ))
        phase_clearance_score = (
            (phase_clearance_per_foot * clearance_weight).sum(dim=-1)
            / clearance_weight.sum(dim=-1).clamp_min(1.0e-4)
        ) * commanded_motion_gate * forward_motion_gate

        # Height is relative to the mean active support surface rather than
        # world Z, so ascending and descending stairs keep the same posture.
        support_count = contact_float.sum(dim=-1).clamp_min(1.0)
        support_mean_z = (foot_world_z * contact_float).sum(dim=-1) / support_count
        support_mean_z = torch.where(
            current_contacts.any(dim=-1), support_mean_z, foot_world_z.amin(dim=-1)
        )
        body_height = self.robot.data.root_link_pos_w[:, 2] - support_mean_z
        height_cfg = cfg["base_height"]
        base_height_score = torch.exp(-torch.square(
            (body_height - height_cfg["target_height"]) / height_cfg["std"]
        ))
        base_height_score -= height_cfg["low_height_penalty"] * torch.square(
            (height_cfg["minimum_height"] - body_height).clamp(min=0.0) / height_cfg["std"]
        )

        # Balance load over time, never instantaneously.  This prevents one
        # leg from being abandoned without forbidding normal stair asymmetry.
        load_cfg = cfg["foot_load_balance"]
        vertical_force = self.contact_sensor.data.net_forces_w[:, self._foot_ids, 2].abs()
        force_total = vertical_force.sum(dim=-1, keepdim=True)
        load_fraction = torch.where(
            force_total > load_cfg["force_threshold"],
            vertical_force / force_total.clamp_min(1.0e-6),
            torch.full_like(vertical_force, 0.25),
        )
        self._foot_load_ema.mul_(load_cfg["ema_alpha"]).add_(
            load_fraction * (1.0 - load_cfg["ema_alpha"])
        )
        load_imbalance = (
            (self._foot_load_ema - 0.25).abs() - load_cfg["tolerance"]
        ).clamp(min=0.0)
        load_imbalance = torch.square(load_imbalance / load_cfg["scale"]).sum(dim=-1)
        load_imbalance = load_imbalance.clamp(max=load_cfg["max_cost"]) * commanded_motion_gate
        self._phase_error_sum.add_(1.0 - phase_contact_quality)
        self._base_height_sum.add_(body_height)
        self._gait_metric_samples.add_(1.0)
        below_speed_band = (self.env.cfg.hier_speed_min - world_forward_velocity).clamp(min=0.0)
        above_speed_band = (world_forward_velocity - self.env.cfg.hier_speed_max).clamp(min=0.0)
        speed_band_error = below_speed_band + above_speed_band
        velocity_direction_error = torch.atan2(world_lateral_velocity, world_forward_velocity)
        velocity_direction_gate = (
            horizontal_speed / cfg["velocity_direction"]["min_speed"]
        ).clamp(0.0, 1.0)
        cross_track_cost = (
            (cross_track_error.abs() - cfg["cross_track_l2"]["deadband"]).clamp(min=0.0)
            / cfg["cross_track_l2"]["scale"]
        ).square().clamp(max=cfg["cross_track_l2"]["max_cost"])
        flat_clearance_target = cfg["foot_clearance"]["target_height"]
        terrain_clearance_target = cfg["foot_clearance"]["terrain_target_height"]
        clearance_target = flat_clearance_target + terrain_gate * (
            terrain_clearance_target - flat_clearance_target
        )
        flat_minimum_clearance = cfg["foot_clearance"]["min_height"]
        terrain_minimum_clearance = cfg["foot_clearance"]["terrain_min_height"]
        minimum_clearance = flat_minimum_clearance + terrain_gate * (
            terrain_minimum_clearance - flat_minimum_clearance
        )
        clearance_score = torch.exp(
            -((touchdown_peak_clearance - clearance_target[:, None])
              / cfg["foot_clearance"]["std"]).square()
        )
        low_clearance_penalty = (
            (minimum_clearance[:, None] - touchdown_peak_clearance)
            / minimum_clearance[:, None]
        ).clamp(0.0, 1.0)
        contact_cycle_cfg = cfg["foot_contact_cycle"]
        overdue_foot_contact = (
            (self._foot_air_duration - contact_cycle_cfg["max_air_time"])
            / contact_cycle_cfg["ramp_seconds"]
        ).clamp(0.0, 1.0)
        overdue_foot_contact *= (
            1.0 - contact_cycle_cfg["terrain_relief"] * terrain_gate[:, None]
        )
        grounded_cfg = cfg["four_feet_grounded"]
        prolonged_full_contact = (
            (self._all_feet_contact_duration - grounded_cfg["grace_seconds"])
            / grounded_cfg["ramp_seconds"]
        ).clamp(0.0, 1.0)
        joint_torque = self.robot.data.applied_torque[:, self._joint_ids]
        joint_position_error = (
            self.robot.data.joint_pos[:, self._joint_ids]
            - self.robot.data.default_joint_pos[:, self._joint_ids]
        )
        joint_velocity = self.robot.data.joint_vel[:, self._joint_ids]
        joint_acceleration = self.robot.data.joint_acc[:, self._joint_ids]
        # The first joint of each ordered Go1 leg is hip ab/adduction.  Penalize
        # only excursions beyond a small deadband and rapid lateral motion;
        # thigh/calf flexion remains free to negotiate steps.
        hip_position_error = joint_position_error[:, 0::3]
        hip_velocity = joint_velocity[:, 0::3]
        # Any part of the robot dropping below the route datum means it is
        # inside a trench. Trench floors sit 0.55+ m under the pads while
        # stairs and slopes never leave the datum, so this cannot fire on a
        # legal step. Graded by depth, plus a large one-off hit on the step
        # that enters, so walking into a void can never be a cheaper way to
        # finish the route than picking a take-off point and leaping.
        intrusion_cfg = cfg["gap_intrusion"]
        # Every pad sits exactly on the route datum and slopes/stairs only rise
        # above it, so the only geometry below the datum is a trench floor or
        # its lip. Measuring against the datum therefore flags the very first
        # touch of a void - a shallow toe dip or a grazed edge - instead of
        # waiting for the leg to be deep inside it.
        # Only the trunk and the supporting feet count as being inside a void.
        # Taking the minimum over every body part also charged a swing foot that
        # merely passes through a narrow void's airspace, which made the narrow
        # end of the gap curriculum (0.2 m at difficulty 0.2) unlearnable: every
        # normal stride over it cost the full -80 per step and the family stayed
        # stuck at 0.28 completion while the other seven reached 0.65-0.89.
        contacts_now = torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w[:, self._foot_ids], dim=-1
        ) > 1.0
        stance_foot_z = torch.where(
            contacts_now,
            self.robot.data.body_pos_w[:, self._robot_foot_ids, 2],
            torch.full_like(contacts_now, float("inf"), dtype=self.robot.data.body_pos_w.dtype),
        ).amin(dim=-1)
        body_z_min = torch.minimum(self.robot.data.root_link_pos_w[:, 2], stance_foot_z)
        below_datum = (-body_z_min).clamp(min=0.0)
        base_drop = (-self.robot.data.root_link_pos_w[:, 2]).clamp(min=0.0)
        intrusion = (
            (below_datum - intrusion_cfg["margin"]).clamp(min=0.0) / intrusion_cfg["span"]
            + base_drop / intrusion_cfg["base_margin"]
        ).clamp(0.0, 1.0)
        entering_gap = (intrusion > 0.0) & (self._gap_intrusion <= 0.0)
        self._gap_intrusion.copy_(intrusion)
        gap_intrusion = intrusion + entering_gap.to(intrusion.dtype) * intrusion_cfg["entry_multiplier"]
        # Positive counterpart: pay once for actually clearing a void with a
        # real flight phase, real forward progress and nothing inside the
        # trench. Without it the only escape from the intrusion penalty is to
        # walk timidly; with it leaping a wide gap is the profitable action.
        leap_cfg = cfg["leap_clear"]
        all_feet_flight = self.contact_sensor.data.current_air_time[:, self._foot_ids].amin(dim=-1)
        clearing = ((all_feet_flight >= leap_cfg["flight_seconds"])
                    & (forward_progress >= leap_cfg["min_progress"])
                    & (intrusion <= 0.0))
        # Only the trench families can clear a void. Paying this bonus for any
        # all-feet flight phase taught the policy to bound everywhere and to
        # sprint for it: the term averaged 0.62 of its maximum per step, which
        # is two orders of magnitude above every other reward. A cooldown on top
        # stops repeated firing from a single machine-gunned hop.
        clearing = clearing & self._gap_family_mask
        cooldown_ready = self._leap_cooldown <= 0
        newly_cleared = clearing & ~self._leap_cleared & cooldown_ready
        self._leap_cooldown = torch.where(
            newly_cleared,
            torch.full_like(self._leap_cooldown, LEAP_CLEAR_COOLDOWN_STEPS),
            (self._leap_cooldown - 1).clamp(min=0),
        )
        self._leap_cleared.copy_(clearing)
        leap_clear = newly_cleared.to(intrusion.dtype) * leap_cfg["bonus"]
        # Landing-region preference. A foot should touch down on open ground,
        # never on the lip of a trench or the nosing of a stair. The height
        # scanner already samples the surface around the robot, so the deepest
        # drop within a small radius of each foot is a geometry-agnostic edge
        # test: pad centres, plateaus and slopes show almost no drop, while a
        # trench lip shows the full void depth and a stair nosing shows one
        # riser. Only feet in contact are charged, with an extra factor on the
        # touchdown step so the policy is paid to choose where it lands, not
        # only to recover afterwards.
        edge_cfg = cfg["edge_landing"]
        # _height_geometry returns the world hit points of the current scan, so
        # no grid bookkeeping is needed to find the surface next to a foot.
        _, edge_scan_valid, edge_scan_points = self._height_geometry()
        edge_scan_z = edge_scan_points[..., 2]
        foot_xy = self.robot.data.body_pos_w[:, self._robot_foot_ids, :2]
        foot_z = self.robot.data.body_pos_w[:, self._robot_foot_ids, 2]
        foot_to_scan = (foot_xy[:, :, None, :] - edge_scan_points[:, None, :, :2]).square().sum(-1)
        scan_near = (foot_to_scan <= edge_cfg["radius"] ** 2) & edge_scan_valid[:, None, :]
        scan_near_z = torch.where(
            scan_near, edge_scan_z[:, None, :],
            torch.full_like(edge_scan_z[:, None, :], float("inf")),
        )
        nearest_surface = scan_near_z.amin(dim=-1)
        edge_drop = torch.where(
            torch.isfinite(nearest_surface), foot_z - nearest_surface,
            torch.zeros_like(foot_z),
        )
        edge_risk = (
            (edge_drop - edge_cfg["drop_threshold"]).clamp(min=0.0) / edge_cfg["scale"]
        ).clamp(max=1.0)
        first_contact = self.contact_sensor.compute_first_contact(CONTROL_DT)[:, self._foot_ids]
        edge_landing = (
            edge_risk
            * contacts.to(actual.dtype)
            * (1.0 + (edge_cfg["touchdown_multiplier"] - 1.0) * first_contact.to(actual.dtype))
        ).sum(-1)
        terms = {
            "track_linear_velocity": torch.exp(-xy_error / cfg["track_linear_velocity"]["std"] ** 2),
            # Linear, constant-gradient speed authority: a narrow exponential gives
            # no signal while the policy is far from the command (measured: the
            # realised speed stayed ~0.93 m/s for every command), so the low level
            # is held to the commanded speed by a penalty that never vanishes.
            "speed_error_l1": (world_forward_velocity - self._command[:, 0]).abs()
            / cfg["speed_error_l1"]["scale"],
            "track_yaw_velocity": torch.exp(-yaw_error / cfg["track_yaw_velocity"]["std"] ** 2),
            "forward_velocity": torch.exp(
                -speed_band_error.square() / cfg["track_linear_velocity"]["std"] ** 2
            ) - cfg["forward_velocity"]["negative_velocity_penalty"] * (-world_forward_velocity).clamp(min=0.0),
            "forward_progress": forward_progress,
            "goal_closeness": normalized_distance,
            "course_checkpoint": checkpoint_reward,
            "goal_reached": self._success.to(actual.dtype),
            "straight_line": torch.exp(
                -cross_track_error.square() / cfg["straight_line"]["std"] ** 2
            ) * forward_motion_gate,
            # Linear direction term: the exponential version paid ~0.006 at 39 deg of
            # lateral drift, so it imposed no usable constraint (xy error hit 0.4-0.8 m/s).
            "velocity_direction": (
                1.0 - velocity_direction_error.abs() / cfg["velocity_direction"]["std"]
            ).clamp(0.0, 1.0) * velocity_direction_gate,
            "cross_track_l2": cross_track_cost,
            "heading_alignment": torch.exp(
                -body_heading_error.square() / cfg["heading_alignment"]["std"] ** 2
            ) * forward_motion_gate,
            "upright": upright,
            "phase_contact": phase_contact_quality,
            "phase_clearance": phase_clearance_score,
            "base_height": base_height_score,
            "foot_load_balance": load_imbalance,
            "trot_gait": trot_timing_score,
            "foot_clearance": torch.where(
                touchdown_events,
                clearance_score - low_clearance_penalty,
                torch.zeros_like(clearance_score),
            ).mul(moving_gate[:, None]).sum(-1),
            "small_steps": torch.where(
                touchdown_events,
                torch.where(
                    (step_lengths >= cfg["small_steps"]["min_length"]) & (step_lengths <= cfg["small_steps"]["max_length"]),
                    1.0 - (step_lengths - cfg["small_steps"]["target_length"]).abs() / cfg["small_steps"]["target_length"],
                    -((step_lengths - cfg["small_steps"]["target_length"]).abs() / cfg["small_steps"]["target_length"]).clamp(max=2.0),
                ), torch.zeros_like(step_lengths)
            ).sum(-1) * moving_gate,
            "step_frequency": torch.where(
                (lift_frequency >= cfg["step_frequency"]["min_hz"]) & (lift_frequency <= cfg["step_frequency"]["max_hz"]),
                1.0 - (lift_frequency - cfg["step_frequency"]["target_hz"]).abs() / cfg["step_frequency"]["target_hz"],
                -((lift_frequency - cfg["step_frequency"]["target_hz"]).abs() / cfg["step_frequency"]["target_hz"]).clamp(max=2.0),
            ).mul(moving_gate[:, None]).mul((contact_count[:, None] < 4).to(actual.dtype)).sum(-1) / 4.0,
            "four_feet_grounded": prolonged_full_contact * moving_gate,
            "foot_contact_cycle": overdue_foot_contact.mul(commanded_motion_gate[:, None]).sum(-1),
            "air_time_variance": air_time_variance,
            "flat_trot_pattern": flat_trot_pattern,
            "foot_lateral": lateral_penalty * (
                1.0 - cfg["foot_lateral"]["terrain_relief"] * terrain_gate
            ),
            "feet_air_time": feet_air_time,
            "feet_slide": (foot_xy_speed * contacts_history).sum(-1),
            "four_feet_airborne": (~contacts).all(dim=-1).to(actual.dtype),
            "base_vertical_velocity_l2": self.robot.data.root_lin_vel_b[:, 2].square(),
            "base_roll_pitch_rate_l2": (
                self.robot.data.root_ang_vel_b[:, :2] / cfg["base_roll_pitch_rate_l2"]["scale"]
            ).square().sum(-1),
            "base_orientation_l2": (
                self.robot.data.projected_gravity_b[:, :2] / cfg["base_orientation_l2"]["scale"]
            ).square().sum(-1),
            "action_rate_l2": self._action_delta.square().sum(-1),
            "action_acceleration_l2": self._action_acceleration.square().sum(-1),
            "joint_torque_l2": joint_torque.square().sum(-1),
            "joint_deviation_l2": (
                (joint_position_error.abs() - cfg["joint_deviation_l2"]["deadband"]).clamp(min=0.0)
                / cfg["joint_deviation_l2"]["scale"]
            ).square().sum(-1),
            "hip_abduction_deviation_l2": (
                (hip_position_error.abs() - cfg["hip_abduction_deviation_l2"]["deadband"]).clamp(min=0.0)
                / cfg["hip_abduction_deviation_l2"]["scale"]
            ).square().sum(-1),
            "hip_abduction_velocity_l2": (
                hip_velocity / cfg["hip_abduction_velocity_l2"]["scale"]
            ).square().sum(-1),
            "joint_power_l1": (joint_torque * joint_velocity).abs().sum(-1),
            "joint_acceleration_l2": joint_acceleration.square().sum(-1),
            "fallen": self._fallen.to(actual.dtype),
            "collision": self._collision_cost,
            "out_of_bounds": self._out_of_bounds.to(actual.dtype),
            # Per-environment zeros, NOT zeros_like(actual): `actual` is the
            # three-axis base velocity, and on a terrain without trenches these
            # four keep their placeholder value, which must stay (num_envs,) or
            # the RewardManager sum fails with a shape error.
            "takeoff_alignment": torch.zeros(self.num_envs, dtype=actual.dtype, device=self.device),
            "flight_clearance": torch.zeros(self.num_envs, dtype=actual.dtype, device=self.device),
            "landing_stability": torch.zeros(self.num_envs, dtype=actual.dtype, device=self.device),
            "jump_air_time": torch.zeros(self.num_envs, dtype=actual.dtype, device=self.device),
            "gap_intrusion": gap_intrusion,
            "leap_clear": leap_clear,
            "edge_landing": edge_landing,
            # Per-environment placeholder for the same reason as the jump terms:
            # zeros_like(actual) would be (num_envs, 3).
            "no_progress": torch.zeros(self.num_envs, dtype=actual.dtype, device=self.device),
        }
        if self._jump_active:
            # Gap mode: the four jump rewards activate, while the airborne
            # suppressors (four_feet_airborne, base_vertical_velocity_l2,
            # feet_air_time) keep their raw values — the jump contract file
            # relaxes their weights instead, so one runtime serves both modes
            # bit-for-bit on the plain course.
            airborne = (~contacts).all(dim=-1).to(actual.dtype)
            # Only pay for flight and touchdown inside a trench window.
            self.trench_state()
            window_open = (self._jump_reward_window > 0).to(actual.dtype)
            # Take-off alignment: a coiled, near-default joint pose while
            # airborne reads as a controlled leap rather than a flailing fall
            # (Extreme Parkour §4.2.2 alignment term).
            terms["takeoff_alignment"] = torch.exp(
                -joint_position_error.square().sum(-1) / cfg["takeoff_alignment"]["std"] ** 2
            ) * airborne * window_open
            # Flight clearance: every foot rides above the highest scanned
            # terrain (the far lip) while airborne, so the legs clear it
            # instead of clipping (arXiv:2401.16337 clearance recipe).
            terrain_values, _, _ = self._height_geometry()
            terrain_top = self.robot.data.root_link_pos_w[:, 2:3] + terrain_values.amax(dim=-1, keepdim=True)
            terms["flight_clearance"] = torch.tanh(
                (foot_world_z - terrain_top) / cfg["flight_clearance"]["std"]
            ).mean(dim=-1) * airborne * window_open
            # Landing stability: the body returns to level shortly after any
            # foot touches down, so a hard landing is not rewarded as success.
            recent_landing = (last_contact_time < 0.30).any(dim=-1).to(actual.dtype)
            terms["landing_stability"] = torch.exp(
                -(1.0 - upright) / cfg["landing_stability"]["std"]
            ) * recent_landing * window_open
            # Airborne duration: the longest foot's ongoing air time, capped at
            # the physics envelope, paid per flight step like feet_air_time.
            terms["jump_air_time"] = current_air_time.amax(dim=-1).clamp(
                max=cfg["jump_air_time"]["max_air_time"]
            ) * window_open
            # Parking at a trench lip must cost more than it earns: the posture
            # terms accrue while standing, so without this penalty the safest
            # policy is to wait out the episode. Graded on the frozen forward
            # speed; a real leap moves faster than the floor.
            slow_fraction = (
                (cfg["no_progress"]["min_speed"] - world_forward_velocity)
                / cfg["no_progress"]["min_speed"]
            ).clamp(0.0, 1.0)
            terms["no_progress"] = slow_fraction
        for name, value in terms.items():
            require_finite(value, f"locomotion reward {name}")
        self._locomotion_reward_cache = terms
        self._reward_cache_tick = self.current_tick
        # Stage return for the LP-ACRL sampler: the RewardManager total is the
        # weighted sum of these terms times the step, so accumulate the same sum
        # here and a finished episode can be attributed to its task instance.
        stage_cfg = self.env.cfg.hier_stage1_rewards
        if self.stage == "locomotion" and stage_cfg:
            total = torch.zeros(self.num_envs, device=self.device)
            for name, value in terms.items():
                table = stage_cfg.get(name)
                if isinstance(table, dict) and table.get("weight"):
                    total = total + float(table["weight"]) * value
            self._episode_return += total * CONTROL_DT
            self._episode_steps += 1
        return terms

    def reward_term(self, name: str):
        """Dispatch a RewardManager term name to the active reward contract."""
        if name.startswith("loco2_"):
            return self._loco_v2_terms()[name[len("loco2_"):]]
        if self.reward_contract == "nav_v2":
            terms = self._nav_v2_terms()
            if name not in terms:
                raise ValueError(f"unknown nav_v2 reward: {name}")
            return terms[name]
        return self.locomotion_reward(name)

    def _loco_v2_terms(self):
        from ..hier_v2.rewards_v2 import LocoV2Inputs, compute_loco_v2_terms

        if self.reward_contract != "loco_v2":
            raise RuntimeError("loco_v2 terms are unavailable under the active reward contract")
        if self._reward_cache_tick == self.current_tick and self._loco_v2_terms_cache:
            return self._loco_v2_terms_cache
        data = self.robot.data
        foot_ids = self._robot_foot_ids
        force = self.contact_sensor.data.net_forces_w
        foot_force = force[:, self._foot_ids]
        default_q = data.default_joint_pos[:, self._joint_ids]
        limits = data.soft_joint_pos_limits[:, self._joint_ids]
        terrain_values, _, _ = self._height_geometry()
        step_height = terrain_values.amax(dim=-1).clamp(min=0.0)
        delta = self._action_delta
        acceleration = self._action_acceleration
        previous = self._previous_action
        if self._collision_ids:
            body_force = torch.linalg.vector_norm(force[:, self._collision_ids], dim=-1)
            contact_count = (body_force > 1.0).sum(-1).to(data.root_lin_vel_b.dtype)
        else:
            contact_count = torch.zeros(self.num_envs, device=self.device)
        inputs = LocoV2Inputs(
            command=self._command,
            lin_vel_b=data.root_lin_vel_b,
            ang_vel_b=data.root_ang_vel_b,
            gravity_b=data.projected_gravity_b,
            joint_pos_rel=data.joint_pos[:, self._joint_ids] - default_q,
            joint_vel=data.joint_vel[:, self._joint_ids],
            joint_acc=data.joint_acc[:, self._joint_ids],
            applied_torque=data.applied_torque[:, self._joint_ids],
            # Limits are expressed relative to the default pose so the penalty
            # compares like with like against ``joint_pos_rel``.
            joint_lower=limits[..., 0] - default_q,
            joint_upper=limits[..., 1] - default_q,
            foot_xy_speed=torch.linalg.vector_norm(data.body_lin_vel_w[:, foot_ids, :2], dim=-1),
            foot_z=data.body_pos_w[:, foot_ids, 2],
            foot_contacts=torch.linalg.vector_norm(foot_force, dim=-1) > 1.0,
            foot_force_z=foot_force[..., 2].abs(),
            foot_force_xy=torch.linalg.vector_norm(foot_force[..., :2], dim=-1),
            first_contact=self.contact_sensor.compute_first_contact(CONTROL_DT)[:, self._foot_ids],
            last_air_time=self.contact_sensor.data.last_air_time[:, self._foot_ids],
            action=previous,
            previous_action=previous - delta,
            previous_previous_action=previous - 2.0 * delta + acceleration,
            base_height=data.root_link_pos_w[:, 2],
            terrain_height_delta=step_height,
            contact_count=contact_count,
            failure=self._collision | self._fallen | self._out_of_bounds | self._fell_into_gap,
        )
        terms = compute_loco_v2_terms(inputs, self.env.cfg.hier_loco_v2_rewards)
        self._loco_v2_terms_cache = terms
        self._reward_cache_tick = self.current_tick
        return terms

    def _nav_v2_terms(self):
        from ..hier_v2.rewards_v2 import NavV2Inputs

        if self._nav_v2 is None:
            raise RuntimeError("nav_v2 tracker was not constructed")
        # The tracker owns per-episode progress, so it may only advance once per
        # navigation period. RewardManager asks on every 50 Hz control step;
        # the intermediate steps return the cached zeros so a single navigation
        # reward is not multiplied by the control/navigation ratio.
        nav_index = self.current_tick // self._NAVIGATION_TICKS
        if nav_index == self._nav_v2_last_index and self._nav_v2_terms_cache:
            return self._nav_v2_terms_cache
        self._update_failures()
        data = self.robot.data
        terrain_values, terrain_valid, _ = self._height_geometry()
        gravity = data.projected_gravity_b
        root_xy = data.root_link_pos_w[:, :2]
        goal_xy = self._goal_world[:, :2]
        offset = goal_xy - root_xy
        distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(1.0e-3)
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        goal_dir = torch.stack((
            cosine * offset[:, 0] + sine * offset[:, 1],
            -sine * offset[:, 0] + cosine * offset[:, 1],
        ), dim=-1) / distance[:, None]
        if self._collision_ids:
            body_force = torch.linalg.vector_norm(
                self.contact_sensor.data.net_forces_w[:, self._collision_ids], dim=-1
            )
            contact_count = (body_force > 1.0).sum(-1).to(distance.dtype)
        else:
            contact_count = torch.zeros_like(distance)
        failure = self._collision | self._fallen | self._out_of_bounds
        if nav_index != self._nav_v2_last_index:
            inputs = NavV2Inputs(
                distance=distance,
                goal_dir_body=goal_dir,
                lin_vel_b=data.root_lin_vel_b,
                ang_vel_b=data.root_ang_vel_b,
                roll=torch.atan2(-gravity[:, 1], -gravity[:, 2].clamp(max=-1.0e-6)),
                pitch=torch.atan2(gravity[:, 0], torch.sqrt(
                    gravity[:, 1].square() + gravity[:, 2].square()).clamp_min(1.0e-6)),
                command=self._command,
                scan=terrain_values,
                contact_count=contact_count,
                done=failure | self._success,
                failure=failure,
                dt=NAV_DT,
                scan_valid=terrain_valid,
            )
            self._nav_v2_terms_cache = self._nav_v2.step(inputs)
            self._nav_v2_last_index = nav_index
        return self._nav_v2_terms_cache

    def _trace_env0(self):
        """Temporary probe: log env 0's gait state so a training rollout and an
        evaluation rollout of the same weights can be compared step by step."""
        if os.environ.get("GO1_DEBUG_TRACE") != "1" or self.current_tick % 25 != 0:
            return
        r = self.robot.data
        print("TRACE", "tick=%d" % self.current_tick,
              "x=%.3f" % float(r.root_link_pos_w[0, 0]),
              "z=%.3f" % float(r.root_link_pos_w[0, 2]),
              "vx=%.3f" % float(r.root_lin_vel_w[0, 0]),
              "leap=%s" % bool(self._leap_mode[0]),
              "cmd=%.3f" % float(self._command[0, 0]),
              "meter=%.3f" % float(self.robot.data.joint_pos[0, 0]),
              flush=True)

    def locomotion_reward(self, name: str):
        terms = self._build_locomotion_reward_terms()
        self._trace_env0()
        if name not in terms:
            raise ValueError(f"unknown locomotion reward: {name}")
        return terms[name]

    def compute_reward(self):
        if self.stage == "locomotion":
            if self.reward_contract == "loco_v2":
                terms = self._loco_v2_terms()
                return sum(self.env.cfg.hier_loco_v2_rewards[name]["weight"] * value
                           for name, value in terms.items())
            terms = self._build_locomotion_reward_terms()
            return sum(self.env.cfg.hier_stage1_rewards[name]["weight"] * value for name, value in terms.items())
        if self.reward_contract == "nav_v2":
            terms = self._nav_v2_terms()
            return sum(self.env.cfg.hier_nav_v2_rewards[name]["weight"] * value
                       for name, value in terms.items())
        self._update_failures()
        actual = self.robot.data.root_lin_vel_b
        xy_error = (actual[:, :2] - self._command[:, :2]).square().sum(-1)
        yaw_error = (self.robot.data.root_ang_vel_b[:, 2] - self._command[:, 2]).square()
        self._xy_error_sum += xy_error
        self._yaw_error_sum += yaw_error
        self._velocity_samples += 1

        distance = torch.linalg.vector_norm(self._goal_world[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=-1)
        progress = self._previous_goal_distance - distance
        moved = progress > 0.005
        self._no_progress_time = torch.where(moved, torch.zeros_like(self._no_progress_time), self._no_progress_time + NAV_DT)
        reward = 8.0 * progress + 5.0 * self._success.to(progress.dtype)
        reward -= 5.0 * self._collision + 5.0 * self._fallen + 0.01
        reward -= 0.05 * self._command_delta.square().sum(-1)
        reward -= 0.2 * (self._no_progress_time > 2.0)
        self._previous_goal_distance.copy_(distance)
        return reward

    def pop_stage_records(self) -> list[dict]:
        """Drain the finished-episode records accumulated since the last call."""
        records = self._stage_records
        self._stage_records = []
        return records

    def get_metrics(self):
        distance = self._update_failures()
        count = self._velocity_samples.clamp_min(1.0)
        gait_count = self._gait_metric_samples.clamp_min(1.0)
        elapsed_seconds = gait_count * CONTROL_DT
        touchdown_hz = self._touchdown_count / elapsed_seconds[:, None]
        metrics = {
            "success": self._success.clone(),
            "collision": self._collision.clone(),
            "fallen": self._fallen.clone(),
            "out_of_bounds": self._out_of_bounds.clone(),
            "fell_into_gap": self._fell_into_gap.clone(),
            "stale_goal": ((~self._goal_valid) | (self._goal_age > MAX_SENSOR_AGE)).clone(),
            "goal_distance": distance,
            "xy_velocity_squared_error": self._xy_error_sum.clone(),
            "yaw_squared_error": self._yaw_error_sum.clone(),
            "velocity_sample_count": self._velocity_samples.clone(),
            "mean_command_speed": self._command_speed_sum / count,
            "mean_achieved_speed": self._achieved_speed_sum / count,
            "xy_velocity_rmse": torch.sqrt(self._xy_error_sum / count),
            "yaw_velocity_rmse": torch.sqrt(self._yaw_error_sum / count),
            "terrain_kind": self._terrain_kind.clone(),
            "mean_gait_phase_error": self._phase_error_sum / gait_count,
            "mean_base_height": self._base_height_sum / gait_count,
            "mean_touchdown_hz": touchdown_hz.mean(dim=-1),
            "touchdown_frequency_cv": (
                touchdown_hz.std(dim=-1, unbiased=False)
                / touchdown_hz.mean(dim=-1).clamp_min(1.0e-4)
            ),
        }
        if self._nav_v2 is not None:
            metrics.update(self._nav_v2.metrics())
        return metrics

    def validate_observations(self):
        self.refresh()
        expected = {
            "locomotion": (self.locomotion_obs, (self.num_envs, LOCO_OBS_DIM)),
            "teacher_locomotion": (self.teacher_locomotion_obs, (self.num_envs, LOCO_OBS_DIM)),
            "navigation": (self.navigation_obs, (self.num_envs, 193)),
            "teacher_navigation": (self.teacher_navigation_obs, (self.num_envs, 193)),
            "privileged": (self.privileged_state(), (self.num_envs, 7)),
        }
        for name, (value, shape) in expected.items():
            if value.shape != shape:
                raise ValueError(f"{name} observation has shape {tuple(value.shape)}, expected {shape}")
            require_finite(value, name)
        return {name: shape for name, (_, shape) in expected.items()}

    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self._reward_cache_tick = -1
        self._locomotion_reward_cache = {}
        self._nav_v2_terms_cache = {}
        self._nav_v2_last_index = -1
        self._loco_v2_terms_cache = {}
        if self._camera_events:
            self._camera_events[-1]["reset_mask"][ids] = True
        self._episode_id[ids] += 1
        # Attribute the finished episode to its environment before anything is
        # cleared. The termination flags and the per-episode sums are only valid
        # here: reading them later in this method reports an always-zero stage
        # report, which is exactly the bug this placement fixes.
        for env_id in ids.tolist():
            samples = max(1.0, float(self._velocity_samples[env_id].item()))
            course_xy = self._terminal_course_xy[env_id]
            self._stage_records.append({
                "env": int(env_id),
                "return": float(self._episode_return[env_id].item()),
                "steps": int(self._episode_steps[env_id].item()),
                "fallen": bool(self._fallen[env_id].item()),
                "success": bool(self._success[env_id].item()),
                "collision": bool(self._collision[env_id].item()),
                "out_of_bounds": bool(self._out_of_bounds[env_id].item()),
                "fell_into_gap": bool(self._fell_into_gap[env_id].item()),
                "command_speed": float(self._command_speed_sum[env_id].item()) / samples,
                "achieved_speed": float(self._achieved_speed_sum[env_id].item()) / samples,
                # Where the episode ended, in course coordinates: this is what
                # separates "fails on the staircase" from "fails on the return
                # ramp" without guessing at the geometry.
                "course_x": float(course_xy[0].item()),
                "course_y": float(course_xy[1].item()),
            })
        self._episode_return[ids] = 0
        self._episode_steps[ids] = 0
        for value in (
            self._command, self._previous_command, self._previous_action, self._action_delta,
            self._action_acceleration,
            self._command_delta, self._goal_relative, self._goal_age, self._previous_goal_distance,
            self._no_progress_time, self._xy_error_sum, self._yaw_error_sum, self._velocity_samples,
            self._command_speed_sum, self._achieved_speed_sum,
            self._frequency_elapsed, self._foot_air_duration, self._swing_peak_clearance,
            self._all_feet_contact_duration, self._touchdown_count, self._phase_error_sum,
            self._base_height_sum, self._gait_metric_samples,
            self._nav_v2_action, self._nav_v2_action_delta,
            self._collision_streak, self._collision_cost, self._terminal_course_xy,
        ):
            value[ids] = 0
        self._collision_update_tick[ids] = -1
        for value in (self._success, self._collision, self._fallen, self._out_of_bounds, self._fell_into_gap):
            value[ids] = False
        self._gap_foot_streak[ids] = 0
        self._gap_intrusion[ids] = 0
        self._leap_cleared[ids] = False
        self._leap_cooldown[ids] = 0
        self._trench_tick = -1
        self._trench_trigger[ids] = False
        self._trench_lip[ids] = float("inf")
        self._jump_reward_window[ids] = 0
        self._leap_age[ids] = 0
        self._command_history[ids] = 0
        self._observed_command[ids] = 0
        self._obs_noise[ids] = 0
        if self._nav_v2 is not None:
            self._nav_v2.reset(ids)
        self._goal_valid[ids] = True
        self._gait_phase[ids] = 0.25
        self._terrain_gate_ema[ids] = 0
        self._foot_load_ema[ids] = 0.25
        self._command_router.reset(ids)
        self._locomotion_hidden[:, ids] = 0
        self._jump_hidden[:, ids] = 0
        self._base_hidden[:, ids] = 0
        self._policy_mask[ids] = 1.0
        self._leap_mode[ids] = False
        self._leap_hold[ids] = 0
        self._terrain_memory.reset(ids)
        self._obstacle_memory.reset(ids)
        self._observable_ready[ids] = False
        self._depth_ready[ids] = False
        if self._perception is not None:
            self._perception.reset(ids)
        # Until the next real frame, reset environments carry explicitly
        # unknown student geometry (zero confidence), never teacher geometry.
        if self._student_terrain is not None:
            self._student_terrain[ids] = 0
        if self._student_obstacle is not None:
            self._student_obstacle[ids] = 0
        self._update_uwb(ids)
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w[ids])
        self._forward_axis[ids, 0] = torch.cos(yaw)
        self._forward_axis[ids, 1] = torch.sin(yaw)
        distance = torch.linalg.vector_norm(
            self._goal_world[ids, :2] - self.robot.data.root_link_pos_w[ids, :2], dim=-1
        )
        self._previous_goal_distance[ids] = distance
        self._previous_forward_position[ids] = (self.robot.data.root_link_pos_w[ids, :2] * self._forward_axis[ids]).sum(-1)
        foot_position = self.robot.data.body_pos_w[ids][:, self._robot_foot_ids]
        self._previous_touchdown_forward[ids] = (foot_position[..., :2] * self._forward_axis[ids, None, :]).sum(-1)
        self._previous_touchdown_z[ids] = foot_position[..., 2]
        root_xy = self.robot.data.root_link_pos_w[ids, :2]
        delta = foot_position[..., :2] - root_xy[:, None, :]
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w[ids])
        self._nominal_foot_lateral[ids] = -torch.sin(yaw)[:, None] * delta[..., 0] + torch.cos(yaw)[:, None] * delta[..., 1]
        reset_foot_force = self.contact_sensor.data.net_forces_w[ids][:, self._foot_ids]
        self._last_foot_contacts[ids] = (
            torch.linalg.vector_norm(reset_foot_force, dim=-1)
            > self.env.cfg.hier_stage1_rewards["feet_slide"]["contact_force_threshold"]
        )
        self._lift_history[ids] = 0
        self._course_checkpoints[ids] = False

