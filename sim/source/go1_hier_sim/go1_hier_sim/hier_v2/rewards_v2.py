"""fwwb-inspired reward contracts for the hierarchical v2 system.

Two independent, strictly validated contracts:

``loco_v2``
    Locomotion shaping: velocity tracking, foot quality, joint safety, power and
    posture.  Used to train the low-level 12-DoF policy.

``nav_v2``
    Goal-conditioned obstacle avoidance shaping: signed path progress, bearing
    progress, one-shot and settled completion, braking, stall, tilt safety and
    wall avoidance.  Used to train the 3-DoF ``[vx, vy, yaw_rate]`` policy on top
    of a frozen locomotion policy.

Both are pure tensor functions over an explicit input bundle, so the arithmetic
can be unit-tested without Isaac Lab.  Persistent episode state lives in tracker
objects whose lifecycle matches the environment reset mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import tomllib

import torch

from ..contracts import require_finite


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

LOCO_V2_FIELDS = {
    "track_lin_vel_xy": {"weight", "std"},
    "track_ang_vel_z": {"weight", "std"},
    "feet_air_time": {"weight", "threshold", "command_threshold"},
    "feet_clearance": {
        "weight", "target_height", "std", "terrain_gain",
        "max_extra_height", "command_threshold",
    },
    "feet_slide": {"weight", "force_threshold"},
    "feet_stumble": {"weight", "force_ratio"},
    "action_rate": {"weight"},
    "action_smoothness": {"weight"},
    "action_magnitude_limit": {"weight", "soft_limit", "hard_limit"},
    "joint_acc": {"weight"},
    "joint_torques": {"weight"},
    "joint_power": {"weight"},
    "dof_pos_limits": {"weight", "soft_ratio"},
    "torque_margin": {"weight", "soft_ratio", "max_torque"},
    "lin_vel_z": {"weight"},
    "ang_vel_xy": {"weight"},
    "flat_orientation": {"weight"},
    "base_height": {"weight", "target_height", "std"},
    "hip_to_default": {"weight"},
    "undesired_contacts": {"weight", "threshold"},
    "termination": {"weight"},
}

LOCO_V2_POSITIVE = {
    "track_lin_vel_xy", "track_ang_vel_z", "feet_air_time", "feet_clearance",
}

NAV_V2_FIELDS = {
    "track_lin_vel_xy": {"weight", "std"},
    "track_ang_vel_z": {"weight", "std"},
    "path_progress": {"weight", "max_speed", "negative_scale"},
    "bearing_progress": {"weight", "max_rate", "negative_scale", "stop_radius"},
    "reach_goal": {"weight", "threshold", "start_offset", "half_width"},
    "settled_goal": {"weight", "threshold", "max_linear_speed", "max_yaw_rate"},
    "goal_braking": {"weight", "slow_radius", "stop_radius", "max_decel"},
    "stuck_duration": {
        "weight", "min_command", "still_speed", "grace_steps", "ramp_steps",
        "goal_radius",
    },
    "unsafe_tilt": {"weight", "roll_limit", "pitch_limit"},
    "tilt_recovery": {"weight", "trigger_roll", "trigger_pitch", "scale"},
    "wall_proximity_brake": {
        "weight", "wall_score_threshold", "safe_speed", "temperature",
        "height_threshold", "front_columns",
    },
    "obstacle_evasion": {"weight", "wall_score_threshold", "turn_threshold"},
    "navigation_time": {"weight"},
    "undesired_contacts": {"weight", "threshold"},
    "termination": {"weight"},
}

NAV_V2_POSITIVE = {
    "track_lin_vel_xy", "track_ang_vel_z", "path_progress", "bearing_progress",
    "reach_goal", "settled_goal", "tilt_recovery",
}


def _load_contract(path: str | Path, fields: dict, positive: set, label: str):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} reward config is missing: {path}")
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != set(fields):
        missing = sorted(set(fields) - set(raw))
        unknown = sorted(set(raw) - set(fields))
        raise ValueError(f"{label} reward tables mismatch; missing={missing}, unknown={unknown}")
    result: dict[str, dict[str, float]] = {}
    for name, expected in fields.items():
        table = raw[name]
        if not isinstance(table, dict) or set(table) != expected:
            raise ValueError(f"{label} reward {name} fields must be {sorted(expected)}")
        values = {}
        for key, value in table.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} reward {name}.{key} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} reward {name}.{key} must be finite")
            values[key] = float(value)
        weight = values["weight"]
        if name in positive and weight < 0:
            raise ValueError(f"{label} reward {name}.weight must be nonnegative")
        if name not in positive and weight > 0:
            raise ValueError(f"{label} penalty {name}.weight must be nonpositive")
        result[name] = values
    return result


def load_loco_v2_rewards(path: str | Path):
    rewards = _load_contract(path, LOCO_V2_FIELDS, LOCO_V2_POSITIVE, "loco_v2")
    for name in ("track_lin_vel_xy", "track_ang_vel_z"):
        if rewards[name]["std"] <= 0:
            raise ValueError(f"loco_v2 reward {name}.std must be positive")
    clearance = rewards["feet_clearance"]
    if clearance["target_height"] <= 0 or clearance["std"] <= 0:
        raise ValueError("loco_v2 feet_clearance target_height/std must be positive")
    if clearance["max_extra_height"] < 0 or clearance["terrain_gain"] < 0:
        raise ValueError("loco_v2 feet_clearance terrain terms must be nonnegative")
    if not 0 < clearance["command_threshold"]:
        raise ValueError("loco_v2 feet_clearance.command_threshold must be positive")
    air = rewards["feet_air_time"]
    if air["threshold"] < 0 or air["command_threshold"] < 0:
        raise ValueError("loco_v2 feet_air_time thresholds must be nonnegative")
    limit = rewards["action_magnitude_limit"]
    if not 0 < limit["soft_limit"] < limit["hard_limit"]:
        raise ValueError("loco_v2 action_magnitude_limit needs 0 < soft < hard")
    for name in ("dof_pos_limits", "torque_margin"):
        if not 0 < rewards[name]["soft_ratio"] < 1:
            raise ValueError(f"loco_v2 {name}.soft_ratio must be inside (0, 1)")
    if rewards["torque_margin"]["max_torque"] <= 0:
        raise ValueError("loco_v2 torque_margin.max_torque must be positive")
    if rewards["base_height"]["std"] <= 0:
        raise ValueError("loco_v2 base_height.std must be positive")
    if rewards["feet_slide"]["force_threshold"] <= 0:
        raise ValueError("loco_v2 feet_slide.force_threshold must be positive")
    return rewards


def load_nav_v2_rewards(path: str | Path):
    rewards = _load_contract(path, NAV_V2_FIELDS, NAV_V2_POSITIVE, "nav_v2")
    for name in ("track_lin_vel_xy", "track_ang_vel_z"):
        if rewards[name]["std"] <= 0:
            raise ValueError(f"nav_v2 reward {name}.std must be positive")
    progress = rewards["path_progress"]
    if progress["max_speed"] <= 0:
        raise ValueError("nav_v2 path_progress.max_speed must be positive")
    if not 0 < progress["negative_scale"] <= 1:
        raise ValueError("nav_v2 path_progress.negative_scale must be inside (0, 1]")
    bearing = rewards["bearing_progress"]
    if bearing["max_rate"] <= 0 or bearing["stop_radius"] < 0:
        raise ValueError("nav_v2 bearing_progress bounds are invalid")
    if not 0 < bearing["negative_scale"] <= 1:
        raise ValueError("nav_v2 bearing_progress.negative_scale must be inside (0, 1]")
    if rewards["reach_goal"]["threshold"] <= 0:
        raise ValueError("nav_v2 reach_goal.threshold must be positive")
    finish_line = rewards["reach_goal"]
    if finish_line["start_offset"] <= 0 or finish_line["half_width"] <= 0:
        raise ValueError("nav_v2 reach_goal finish line must be ahead and have a positive half width")
    settled = rewards["settled_goal"]
    if settled["threshold"] <= 0 or settled["max_linear_speed"] <= 0 or settled["max_yaw_rate"] <= 0:
        raise ValueError("nav_v2 settled_goal parameters must be positive")
    braking = rewards["goal_braking"]
    if not 0 <= braking["stop_radius"] < braking["slow_radius"] or braking["max_decel"] <= 0:
        raise ValueError("nav_v2 goal_braking radii/deceleration are invalid")
    stuck = rewards["stuck_duration"]
    if stuck["min_command"] <= 0 or stuck["still_speed"] <= 0:
        raise ValueError("nav_v2 stuck_duration thresholds must be positive")
    if stuck["grace_steps"] < 0 or stuck["ramp_steps"] <= 0 or stuck["goal_radius"] < 0:
        raise ValueError("nav_v2 stuck_duration steps are invalid")
    tilt = rewards["unsafe_tilt"]
    if tilt["roll_limit"] <= 0 or tilt["pitch_limit"] <= 0:
        raise ValueError("nav_v2 unsafe_tilt limits must be positive")
    recovery = rewards["tilt_recovery"]
    if recovery["trigger_roll"] <= 0 or recovery["trigger_pitch"] <= 0 or recovery["scale"] <= 0:
        raise ValueError("nav_v2 tilt_recovery parameters must be positive")
    for name in ("wall_proximity_brake", "obstacle_evasion"):
        table = rewards[name]
        if not 0 <= table["wall_score_threshold"] < 1:
            raise ValueError(f"nav_v2 {name}.wall_score_threshold must be inside [0, 1)")
    proximity = rewards["wall_proximity_brake"]
    if proximity["temperature"] <= 0 or proximity["front_columns"] < 1:
        raise ValueError("nav_v2 wall_proximity_brake sector parameters are invalid")
    if rewards["obstacle_evasion"]["turn_threshold"] < 0:
        raise ValueError("nav_v2 obstacle_evasion.turn_threshold must be nonnegative")
    if rewards["undesired_contacts"]["threshold"] < 0:
        raise ValueError("nav_v2 undesired_contacts.threshold must be nonnegative")
    return rewards


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def finish_crossed(forward: torch.Tensor, lateral: torch.Tensor,
                   start_offset: float, half_width: float) -> torch.Tensor:
    """Completion test for the straight course: crossing the finish line.

    Passing the line anywhere across the drivable width finishes the task.
    Overshooting still counts, and neither a standstill nor alignment on the
    course centre line is required; requiring those turned completion into a
    precision task the navigation reward never asked for.
    """
    if half_width <= 0:
        raise ValueError("finish_crossed half_width must be positive")
    return (forward >= start_offset) & (lateral.abs() <= half_width)


def scan_grid(scan: torch.Tensor, rows: int, columns: int, valid: torch.Tensor | None = None):
    """Reshape a flat height scan into ``[N, rows, columns]`` forward-major grid."""
    if scan.dim() != 2 or scan.shape[-1] != rows * columns:
        raise ValueError(f"height scan must be [N,{rows * columns}], got {tuple(scan.shape)}")
    grid = scan.reshape(scan.shape[0], rows, columns)
    if valid is None:
        return grid, torch.ones_like(grid, dtype=torch.bool)
    return grid, valid.reshape(scan.shape[0], rows, columns)


def wall_score(sector: torch.Tensor, sector_valid: torch.Tensor,
               temperature: float, height_threshold: float, ground: torch.Tensor):
    """Soft fraction of a scan sector that is higher than the local ground.

    ``sector`` holds base-relative hit heights, so a riser appears as a jump
    above the local ground estimate.  Invalid rays never count as free space.
    """
    if temperature <= 0:
        raise ValueError("wall_score temperature must be positive")
    delta = sector - ground[:, None, None]
    score = torch.sigmoid((delta - float(height_threshold)) / float(temperature))
    score = torch.where(sector_valid, score, torch.zeros_like(score))
    weight = sector_valid.to(score.dtype).sum(dim=(-1, -2)).clamp_min(1.0)
    return score.sum(dim=(-1, -2)) / weight


def local_ground(grid: torch.Tensor, grid_valid: torch.Tensor, quantile: float = 0.25) -> torch.Tensor:
    """Lower quantile of valid scan heights, a base-relative ground estimate."""
    flat = grid.reshape(grid.shape[0], -1)
    mask = grid_valid.reshape(grid.shape[0], -1)
    filled = torch.where(mask, flat, torch.full_like(flat, 1.0e3))
    ground = torch.quantile(filled, float(quantile), dim=-1)
    fallback = torch.where(mask, flat, torch.full_like(flat, -1.0e3)).amax(dim=-1)
    return torch.where(ground >= 1.0e3, fallback, ground)


# ---------------------------------------------------------------------------
# Locomotion v2
# ---------------------------------------------------------------------------

@dataclass
class LocoV2Inputs:
    command: torch.Tensor              # [N,3] vx, vy, yaw rate
    lin_vel_b: torch.Tensor            # [N,3]
    ang_vel_b: torch.Tensor            # [N,3]
    gravity_b: torch.Tensor            # [N,3] projected gravity
    joint_pos_rel: torch.Tensor        # [N,12] relative to default
    joint_vel: torch.Tensor            # [N,12]
    joint_acc: torch.Tensor            # [N,12]
    applied_torque: torch.Tensor       # [N,12]
    joint_lower: torch.Tensor          # [N,12] soft limits
    joint_upper: torch.Tensor          # [N,12]
    foot_xy_speed: torch.Tensor        # [N,4] world XY speed
    foot_z: torch.Tensor               # [N,4] world Z
    foot_contacts: torch.Tensor        # [N,4] bool
    foot_force_z: torch.Tensor         # [N,4]
    foot_force_xy: torch.Tensor        # [N,4]
    first_contact: torch.Tensor        # [N,4] bool
    last_air_time: torch.Tensor        # [N,4]
    action: torch.Tensor               # [N,12]
    previous_action: torch.Tensor      # [N,12]
    previous_previous_action: torch.Tensor  # [N,12]
    base_height: torch.Tensor          # [N]
    terrain_height_delta: torch.Tensor | None = None  # [N] local step height under the robot
    hip_joint_ids: tuple = (0, 3, 6, 9)
    contact_count: torch.Tensor | None = None   # [N] non-foot contacts above threshold
    failure: torch.Tensor | None = None         # [N] bool


def compute_loco_v2_terms(inputs: LocoV2Inputs, cfg: dict) -> dict[str, torch.Tensor]:
    """Return the raw (unweighted, per-step) ``loco_v2`` reward terms."""
    cmd = inputs.command
    vel = inputs.lin_vel_b

    xy_error = (cmd[:, :2] - vel[:, :2]).square().sum(-1)
    yaw_error = (cmd[:, 2] - inputs.ang_vel_b[:, 2]).square()
    terms: dict[str, torch.Tensor] = {
        "track_lin_vel_xy": torch.exp(-xy_error / float(cfg["track_lin_vel_xy"]["std"]) ** 2),
        "track_ang_vel_z": torch.exp(-yaw_error / float(cfg["track_ang_vel_z"]["std"]) ** 2),
    }

    air_cfg = cfg["feet_air_time"]
    moving = (torch.linalg.vector_norm(cmd[:, :2], dim=-1) > float(air_cfg["command_threshold"]))
    capped = inputs.last_air_time.clamp(max=2.0)
    terms["feet_air_time"] = (
        ((capped - float(air_cfg["threshold"])) * inputs.first_contact.to(capped.dtype)).sum(-1)
        * moving.to(capped.dtype)
    )

    clear_cfg = cfg["feet_clearance"]
    swing = ~inputs.foot_contacts
    neg_inf = torch.full_like(inputs.foot_z, -torch.inf)
    support = torch.where(~swing, inputs.foot_z, neg_inf).max(dim=-1).values
    no_support = ~torch.isfinite(support)
    support = torch.where(no_support, inputs.foot_z.max(dim=-1).values, support)
    clearance = torch.where(swing, inputs.foot_z - support[:, None], torch.zeros_like(inputs.foot_z))
    if inputs.terrain_height_delta is None:
        terrain_delta = torch.zeros_like(inputs.base_height)
    else:
        terrain_delta = inputs.terrain_height_delta
    terrain_extra = (
        float(clear_cfg["terrain_gain"]) * terrain_delta
    ).clamp(0.0, float(clear_cfg["max_extra_height"]))
    target = float(clear_cfg["target_height"]) + terrain_extra[:, None]
    clearance_reward = torch.exp(
        -((clearance - target) / float(clear_cfg["std"])).square()
    ) * swing.to(clearance.dtype)
    moving_gate = torch.linalg.vector_norm(cmd[:, :2], dim=-1) > float(clear_cfg["command_threshold"])
    terms["feet_clearance"] = clearance_reward.sum(-1) * moving_gate.to(clearance.dtype)

    slide_cfg = cfg["feet_slide"]
    terms["feet_slide"] = (
        (inputs.foot_xy_speed * inputs.foot_contacts.to(inputs.foot_xy_speed.dtype)).sum(-1)
    )

    stumble_cfg = cfg["feet_stumble"]
    ratio = float(stumble_cfg["force_ratio"])
    terms["feet_stumble"] = (
        (inputs.foot_force_xy > ratio * inputs.foot_force_z)
        & inputs.foot_contacts
    ).sum(-1).to(vel.dtype)

    terms["action_rate"] = (inputs.action - inputs.previous_action).square().sum(-1)
    terms["action_smoothness"] = (
        inputs.action - 2.0 * inputs.previous_action + inputs.previous_previous_action
    ).square().sum(-1)

    limit_cfg = cfg["action_magnitude_limit"]
    magnitude = inputs.action.abs()
    soft = (magnitude - float(limit_cfg["soft_limit"])).clamp(min=0.0)
    hard = (magnitude - float(limit_cfg["hard_limit"])).clamp(min=0.0)
    terms["action_magnitude_limit"] = (soft.square() + 3.0 * hard.square()).sum(-1)

    terms["joint_acc"] = inputs.joint_acc.square().sum(-1)
    terms["joint_torques"] = inputs.applied_torque.square().sum(-1)
    terms["joint_power"] = (inputs.applied_torque * inputs.joint_vel).abs().sum(-1)

    pos_cfg = cfg["dof_pos_limits"]
    span = (inputs.joint_upper - inputs.joint_lower) * 0.5
    mid = (inputs.joint_upper + inputs.joint_lower) * 0.5
    soft_span = span * float(pos_cfg["soft_ratio"])
    exceed = ((inputs.joint_pos_rel - mid).abs() - soft_span).clamp(min=0.0)
    terms["dof_pos_limits"] = exceed.square().sum(-1)

    margin_cfg = cfg["torque_margin"]
    ratio_used = inputs.applied_torque.abs() / float(margin_cfg["max_torque"])
    excess = ((ratio_used - float(margin_cfg["soft_ratio"])) / (1.0 - float(margin_cfg["soft_ratio"]))).clamp(min=0.0)
    terms["torque_margin"] = excess.square().mean(-1).clamp(max=1.0)

    terms["lin_vel_z"] = vel[:, 2].square()
    terms["ang_vel_xy"] = inputs.ang_vel_b[:, :2].square().sum(-1)
    terms["flat_orientation"] = inputs.gravity_b[:, :2].square().sum(-1)
    terms["base_height"] = (
        (inputs.base_height - float(cfg["base_height"]["target_height"]))
        / float(cfg["base_height"]["std"])
    ).square()
    terms["hip_to_default"] = inputs.joint_pos_rel[:, list(inputs.hip_joint_ids)].square().sum(-1)

    contacts = inputs.contact_count
    if contacts is None:
        contacts = torch.zeros_like(vel[:, 0])
    terms["undesired_contacts"] = (contacts > float(cfg["undesired_contacts"]["threshold"])).to(vel.dtype)
    failure = inputs.failure
    if failure is None:
        failure = torch.zeros_like(vel[:, 0], dtype=torch.bool)
    terms["termination"] = failure.to(vel.dtype)

    for name, value in terms.items():
        require_finite(value, f"loco_v2 reward {name}")
    return terms


# ---------------------------------------------------------------------------
# Navigation v2
# ---------------------------------------------------------------------------

@dataclass
class NavV2Inputs:
    distance: torch.Tensor        # [N] goal distance (m)
    goal_dir_body: torch.Tensor   # [N,2] unit vector toward the goal
    lin_vel_b: torch.Tensor       # [N,3]
    ang_vel_b: torch.Tensor       # [N,3]
    roll: torch.Tensor            # [N]
    pitch: torch.Tensor           # [N]
    command: torch.Tensor         # [N,3]
    scan: torch.Tensor            # [N, rows*columns] base-relative heights
    contact_count: torch.Tensor   # [N] non-foot contacts above threshold
    done: torch.Tensor            # [N] bool full reset mask
    failure: torch.Tensor         # [N] bool failure termination
    dt: float
    scan_valid: torch.Tensor | None = None
    scan_rows: int = 16
    scan_columns: int = 32


class NavV2Tracker:
    """Per-episode navigation state advanced once per navigation step."""

    def __init__(self, num_envs: int, device, cfg: dict,
                 scan_rows: int = 16, scan_columns: int = 32):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self.scan_rows = int(scan_rows)
        self.scan_columns = int(scan_columns)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_tilt = torch.zeros(self.num_envs, device=self.device)
        self._previous_tilt_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._previous_distance.zero_()
            self._valid.zero_()
            self._stuck_steps.zero_()
            self._previous_tilt.zero_()
            self._previous_tilt_valid.zero_()
            self._reached.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self._previous_distance[ids] = 0
        self._valid[ids] = False
        self._stuck_steps[ids] = 0
        self._previous_tilt[ids] = 0
        self._previous_tilt_valid[ids] = False
        self._reached[ids] = False

    def step(self, inputs: NavV2Inputs) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        if inputs.distance.shape != (self.num_envs,):
            raise ValueError("nav_v2 distance must be [N]")
        if inputs.dt <= 0:
            raise ValueError("nav_v2 dt must be positive")
        done = inputs.done.bool()

        grid, grid_valid = scan_grid(inputs.scan, self.scan_rows, self.scan_columns, inputs.scan_valid)
        ground = local_ground(grid, grid_valid)
        proximity_cfg = cfg["wall_proximity_brake"]
        columns = max(1, min(int(proximity_cfg["front_columns"]), grid.shape[-1]))
        sector = grid[:, :, :columns]
        sector_valid = grid_valid[:, :, :columns]
        front_wall = wall_score(
            sector, sector_valid, float(proximity_cfg["temperature"]),
            float(proximity_cfg["height_threshold"]), ground,
        )

        xy_error = (inputs.command[:, :2] - inputs.lin_vel_b[:, :2]).square().sum(-1)
        yaw_error = (inputs.command[:, 2] - inputs.ang_vel_b[:, 2]).square()
        terms: dict[str, torch.Tensor] = {
            "track_lin_vel_xy": torch.exp(
                -xy_error / float(cfg["track_lin_vel_xy"]["std"]) ** 2
            ),
            "track_ang_vel_z": torch.exp(
                -yaw_error / float(cfg["track_ang_vel_z"]["std"]) ** 2
            ),
        }

        # ---- signed path progress, capped by the commanded forward speed ----
        progress_cfg = cfg["path_progress"]
        first_sample = ~self._valid
        raw = (self._previous_distance - inputs.distance) / float(inputs.dt)
        raw = torch.where(raw < 0.0, raw * float(progress_cfg["negative_scale"]), raw)
        raw = raw.clamp(-float(progress_cfg["max_speed"]), float(progress_cfg["max_speed"]))
        command_vx = inputs.command[:, 0].clamp(min=0.0, max=float(progress_cfg["max_speed"]))
        margin = 0.05 * float(progress_cfg["max_speed"])
        cap = (command_vx + margin).clamp(min=0.0, max=float(progress_cfg["max_speed"]))
        raw = torch.where(raw > 0.0, torch.minimum(raw, cap), raw)
        path_progress = torch.where(self._valid & ~done, raw, torch.zeros_like(raw))
        terms["path_progress"] = path_progress

        # ---- bearing progress: velocity projected onto the goal direction ----
        bearing_cfg = cfg["bearing_progress"]
        radial = (inputs.lin_vel_b[:, :2] * inputs.goal_dir_body).sum(-1)
        radial = torch.where(
            radial < 0.0, radial * float(bearing_cfg["negative_scale"]), radial
        )
        radial = radial.clamp(-float(bearing_cfg["max_rate"]), float(bearing_cfg["max_rate"]))
        terms["bearing_progress"] = torch.where(
            inputs.distance > float(bearing_cfg["stop_radius"]), radial,
            torch.zeros_like(radial),
        )

        # ---- one-shot and settled completion ----
        reach_cfg = cfg["reach_goal"]
        newly_reached = (inputs.distance < float(reach_cfg["threshold"])) & ~self._reached
        terms["reach_goal"] = newly_reached.to(inputs.distance.dtype)
        settle_cfg = cfg["settled_goal"]
        speed = torch.linalg.vector_norm(inputs.lin_vel_b[:, :2], dim=-1)
        settled = (
            (inputs.distance < float(settle_cfg["threshold"]))
            & (speed < float(settle_cfg["max_linear_speed"]))
            & (inputs.ang_vel_b[:, 2].abs() < float(settle_cfg["max_yaw_rate"]))
        )
        terms["settled_goal"] = settled.to(inputs.distance.dtype)

        # ---- braking, stall, tilt, wall terms ----
        brake_cfg = cfg["goal_braking"]
        remaining = (inputs.distance - float(brake_cfg["stop_radius"])).clamp(min=0.0)
        allowed = torch.sqrt(2.0 * float(brake_cfg["max_decel"]) * remaining)
        terms["goal_braking"] = (
            (inputs.distance < float(brake_cfg["slow_radius"])).to(inputs.distance.dtype)
            * (radial - allowed).clamp(min=0.0).square()
        )

        stuck_cfg = cfg["stuck_duration"]
        commanded = torch.linalg.vector_norm(inputs.command[:, :2], dim=-1) > float(stuck_cfg["min_command"])
        stopped = speed < float(stuck_cfg["still_speed"])
        active = commanded & stopped & (inputs.distance > float(stuck_cfg["goal_radius"])) & ~done
        stuck_steps = torch.where(
            active, self._stuck_steps + 1, torch.zeros_like(self._stuck_steps)
        )
        excess = (stuck_steps.float() - float(stuck_cfg["grace_steps"])).clamp(min=0.0)
        terms["stuck_duration"] = (excess / float(stuck_cfg["ramp_steps"])).clamp(max=1.0)

        tilt_cfg = cfg["unsafe_tilt"]
        unsafe = (
            (inputs.roll.abs() > float(tilt_cfg["roll_limit"]))
            | (inputs.pitch.abs() > float(tilt_cfg["pitch_limit"]))
        )
        terms["unsafe_tilt"] = unsafe.to(inputs.distance.dtype)

        recovery_cfg = cfg["tilt_recovery"]
        tilt = inputs.roll.abs() + inputs.pitch.abs()
        dangerous = (
            (inputs.roll.abs() > float(recovery_cfg["trigger_roll"]))
            | (inputs.pitch.abs() > float(recovery_cfg["trigger_pitch"]))
        )
        recovery = (
            (self._previous_tilt - tilt) / (float(inputs.dt) * float(recovery_cfg["scale"]))
        ).clamp(-1.0, 1.0)
        terms["tilt_recovery"] = torch.where(
            self._previous_tilt_valid & dangerous & ~done, recovery, torch.zeros_like(recovery)
        )

        wall_gate = (
            (front_wall - float(proximity_cfg["wall_score_threshold"]))
            / max(1.0 - float(proximity_cfg["wall_score_threshold"]), 1.0e-6)
        ).clamp(0.0, 1.0)
        actual_vx = inputs.lin_vel_b[:, 0]
        terms["wall_proximity_brake"] = (
            wall_gate * (actual_vx - float(proximity_cfg["safe_speed"])).clamp(min=0.0)
        )

        evade_cfg = cfg["obstacle_evasion"]
        blocked = front_wall > float(evade_cfg["wall_score_threshold"])
        not_turning = inputs.command[:, 2].abs() < float(evade_cfg["turn_threshold"])
        terms["obstacle_evasion"] = (
            blocked & not_turning
        ).to(inputs.distance.dtype) * inputs.command[:, 0].clamp(min=0.0)

        terms["navigation_time"] = torch.ones_like(inputs.distance)
        terms["undesired_contacts"] = (
            inputs.contact_count > float(cfg["undesired_contacts"]["threshold"])
        ).to(inputs.distance.dtype)
        terms["termination"] = inputs.failure.to(inputs.distance.dtype)

        # ---- advance persistent state ----
        self._previous_distance.copy_(inputs.distance.detach())
        self._valid.copy_(~done)
        self._stuck_steps.copy_(torch.where(done, torch.zeros_like(stuck_steps), stuck_steps))
        self._previous_tilt.copy_(tilt.detach())
        self._previous_tilt_valid.copy_(~done)
        self._reached.copy_((self._reached | newly_reached) & ~done)

        for name, value in terms.items():
            require_finite(value, f"nav_v2 reward {name}")
        return terms

    def weighted_sum(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, value in terms.items():
            contribution = float(self.cfg[name]["weight"]) * value
            total = contribution if total is None else total + contribution
        if total is None:
            raise ValueError("nav_v2 reward contract is empty")
        return total
