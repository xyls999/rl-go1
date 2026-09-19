"""Versioned deployment inputs, units and explicit command ownership."""

from dataclasses import dataclass

import torch

SCHEMA_VERSION = 5
PHYSICS_DT = 0.005
CONTROL_DT = 0.02
NAV_DT = 0.1
DEPTH_DT = 0.05
MAX_SENSOR_AGE = 0.5
TERRAIN_HEIGHT_DIM = 512
TERRAIN_FEATURE_DIM = 2 * TERRAIN_HEIGHT_DIM
# Safety clamp on the forward command. It is separate from the observation
# normalisation below, which stays at 0.8 so that the manifest - and therefore
# every existing checkpoint - keeps its interface hash.
MAX_FORWARD_COMMAND = 1.0
LOCO_PREFIX_DIM = 49
LOCO_OBS_DIM = LOCO_PREFIX_DIM + TERRAIN_FEATURE_DIM
# Speed-layer geometry: a terrain relief map replaces the obstacle range scan
# that schema 4 used. Rows are look-ahead distances along the frozen course
# axis, columns are lateral offsets; every cell carries a ground gradient and a
# confidence, so the dimensions (and therefore the navigation observation size)
# are unchanged.
RELIEF_ROWS = 6
RELIEF_COLS = 12
RELIEF_DIM = RELIEF_ROWS * RELIEF_COLS
RELIEF_FORWARD_M = (0.6, 1.2, 1.8, 2.6, 3.6, 5.0)
RELIEF_LATERAL_M = 1.1
RELIEF_PROBE_HEIGHT = 3.5
# Navigation prefix: proprio(42) + goal(4) + previous command(3).
NAV_PREFIX_DIM = 49
NAV_GEOMETRY_DIM = 2 * RELIEF_DIM
NAV_OBS_DIM = NAV_PREFIX_DIM + NAV_GEOMETRY_DIM
LOCO_CRITIC_DIM = LOCO_OBS_DIM + 7
NAV_CRITIC_DIM = 200
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


def require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")


def pack_proprio(omega, gravity, qrel, dq, prev_action):
    values = (omega, gravity, qrel, dq, prev_action)
    for value, size, name in zip(values, (3, 3, 12, 12, 12),
                                 ("omega", "gravity", "qrel", "dq", "prev_action")):
        if value.shape[-1] != size:
            raise ValueError(f"{name}: expected last dimension {size}, got {value.shape}")
        require_finite(value, name)
    return torch.cat((omega * 0.25, gravity, qrel, dq * 0.05,
                      prev_action.clamp(-3.0, 3.0) / 3.0), dim=-1)


def normalize_command(command):
    if command.shape[-1] != 3:
        raise ValueError("command must have three components [vx, vy, wz]")
    require_finite(command, "command")
    return command / command.new_tensor([0.8, 0.3, 1.0])


def gait_phase_observation(phase):
    """Encode the explicit locomotion clock without a discontinuity at wrap-around."""
    require_finite(phase, "gait phase")
    angle = 2.0 * torch.pi * torch.remainder(phase, 1.0)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def fixed_heading_observation(forward_axis, yaw):
    """Express the immutable reset heading in the current body frame.

    This is a relative IMU heading observation, not absolute position or an
    oracle velocity.  It is [1, 0] at reset and remains observable after yaw
    drift, so the policy need not integrate gyro noise indefinitely.
    """
    if forward_axis.shape[-1] != 2 or yaw.shape != forward_axis.shape[:-1]:
        raise ValueError("fixed-heading inputs have incompatible shapes")
    require_finite(forward_axis, "fixed world forward axis")
    require_finite(yaw, "heading")
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        cosine * forward_axis[..., 0] + sine * forward_axis[..., 1],
        -sine * forward_axis[..., 0] + cosine * forward_axis[..., 1],
    ), dim=-1)


def soft_trot_schedule(phase, duty_factor, transition_width):
    """Return soft per-foot stance probabilities and normalized swing progress.

    Foot order is FL, FR, RL, RR.  The diagonal pairs FL+RR and FR+RL are
    synchronized and separated by half a cycle.  Only contact timing is
    specified; joint trajectories remain free for terrain adaptation.
    """
    require_finite(phase, "gait phase")
    require_finite(duty_factor, "gait duty factor")
    if phase.shape != duty_factor.shape:
        raise ValueError("phase and duty_factor must have the same shape")
    if not 0.0 < float(transition_width) < 0.25:
        raise ValueError("transition_width must be inside (0, 0.25)")
    if ((duty_factor <= 0.0) | (duty_factor >= 1.0)).any():
        raise ValueError("duty_factor must be inside (0, 1)")
    offsets = phase.new_tensor((0.0, 0.5, 0.5, 0.0))
    leg_phase = torch.remainder(phase[:, None] + offsets[None, :], 1.0)
    center = 0.5 * duty_factor[:, None]
    circular_distance = torch.remainder(leg_phase - center + 0.5, 1.0) - 0.5
    stance_probability = torch.sigmoid(
        (0.5 * duty_factor[:, None] - circular_distance.abs()) / float(transition_width)
    )
    swing_progress = ((leg_phase - duty_factor[:, None]) / (1.0 - duty_factor[:, None])).clamp(0.0, 1.0)
    return stance_probability, swing_progress


def command_from_action(action):
    if action.shape[-1] != 3:
        raise ValueError("navigation action must have three components")
    require_finite(action, "navigation action")
    a = action.clamp(-1.0, 1.0)
    return torch.stack(((a[..., 0] + 1.0) * 0.4, a[..., 1] * 0.3, a[..., 2]), dim=-1)


def pack_goal(relative_xy, valid, age):
    if relative_xy.shape[-1] != 2:
        raise ValueError("goal requires relative XY; a single range is insufficient")
    if valid.shape != relative_xy.shape[:-1] or age.shape != valid.shape:
        raise ValueError("goal validity and age must match its batch dimensions")
    require_finite(age, "goal age")
    if (age < 0).any():
        raise ValueError("goal timestamp is in the future")
    if not torch.isfinite(relative_xy[valid.bool()]).all():
        raise ValueError("a goal marked valid contains nonfinite coordinates")
    usable = valid.bool() & (age <= MAX_SENSOR_AGE)
    xy = torch.where(usable[..., None], relative_xy, torch.zeros_like(relative_xy))
    return torch.cat((xy / 10.0, usable[..., None].to(xy.dtype),
                      (age / MAX_SENSOR_AGE).clamp(0, 1)[..., None]), dim=-1)


def world_goal_to_body(robot_xy, target_xy, yaw):
    """Convert actual localization outputs; this function does not estimate heading."""
    require_finite(robot_xy, "robot localization")
    require_finite(target_xy, "target localization")
    require_finite(yaw, "heading")
    delta = target_xy - robot_xy
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * delta[..., 0] + s * delta[..., 1],
                        -s * delta[..., 0] + c * delta[..., 1]), dim=-1)


def fixed_world_velocity_to_body(forward_axis, speed, yaw):
    """Express one immutable world-frame travel direction in the live body frame."""
    if forward_axis.shape[-1] != 2 or speed.shape != forward_axis.shape[:-1] or yaw.shape != speed.shape:
        raise ValueError("fixed-direction velocity inputs have incompatible shapes")
    require_finite(forward_axis, "fixed world forward axis")
    require_finite(speed, "fixed-direction speed")
    require_finite(yaw, "heading")
    desired_world = forward_axis * speed[..., None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        c * desired_world[..., 0] + s * desired_world[..., 1],
        -s * desired_world[..., 0] + c * desired_world[..., 1],
    ), dim=-1)


def rectangular_region_distance(forward, lateral, start_offset, end_offset, half_width):
    """Distance to a course-local rectangular finish strip (zero inside)."""
    require_finite(forward, "finish-region forward position")
    require_finite(lateral, "finish-region lateral position")
    if forward.shape != lateral.shape:
        raise ValueError("finish-region coordinates must have the same shape")
    if end_offset <= start_offset or half_width <= 0:
        raise ValueError("finish-region bounds are invalid")
    forward_gap = torch.maximum(
        (start_offset - forward).clamp(min=0.0),
        (forward - end_offset).clamp(min=0.0),
    )
    lateral_gap = (lateral.abs() - half_width).clamp(min=0.0)
    return torch.sqrt(forward_gap.square() + lateral_gap.square())


def depth_to_input(z_depth):
    """Preserve invalid-pixel information instead of treating invalid depth as free space."""
    if z_depth.ndim != 4 or z_depth.shape[1:] != (60, 80, 1):
        raise ValueError("camera depth must have shape [N,60,80,1]")
    valid = torch.isfinite(z_depth) & (z_depth >= 0.2) & (z_depth <= 6.0)
    depth = torch.where(valid, z_depth / 6.0, torch.zeros_like(z_depth))
    return torch.cat((depth, valid.to(depth.dtype)), dim=-1).permute(0, 3, 1, 2).contiguous()


@dataclass(frozen=True)
class CommandResult:
    velocity: torch.Tensor
    stopped: torch.Tensor
    reason: str


class CommandRouter:
    """One command owner per session; external coordination uses the same physical units."""

    def __init__(self, num_envs, device="cpu", source="navigation"):
        if source not in ("navigation", "external"):
            raise ValueError(f"unsupported command source: {source}")
        self.source = source
        self.velocity = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.velocity.zero_()
        else:
            self.velocity[env_ids] = 0

    def update(self, velocity, age, valid, dt, source):
        if source != self.source:
            raise ValueError(f"command owner is {self.source}, received {source}")
        if velocity.shape != self.velocity.shape:
            raise ValueError("command batch shape differs from router configuration")
        require_finite(velocity, "requested velocity")
        require_finite(age, "command age")
        if dt <= 0 or not torch.isfinite(torch.tensor(dt)) or (age < 0).any():
            raise ValueError("invalid command timing")
        if valid.shape != velocity.shape[:-1] or age.shape != valid.shape:
            raise ValueError("invalid command validity/age shape")
        stopped = (~valid.bool()) | (age > MAX_SENSOR_AGE)
        lo, hi = velocity.new_tensor([0.0, -0.3, -1.0]), velocity.new_tensor([MAX_FORWARD_COMMAND, 0.3, 1.0])
        target = velocity.clamp(lo, hi)
        target = torch.where(stopped[..., None], torch.zeros_like(target), target)
        alpha = 1.0 - torch.exp(torch.tensor(-dt / 0.2)).item()
        self.velocity += alpha * (target - self.velocity)
        # Sensor loss is a safety state, not an ordinary command transition.
        self.velocity[stopped] = 0
        return CommandResult(self.velocity.clone(), stopped, "invalid_or_stale_input" if stopped.any() else "ok")


def interface_manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "joint_names": list(JOINT_NAMES),
        "locomotion": {"input": LOCO_OBS_DIM, "critic": LOCO_CRITIC_DIM, "hidden": 128, "output": 12},
        "navigation": {"input": 193, "critic": 200, "hidden": 128, "output": 3},
        "vision": {"image": [2, 60, 80], "state": 45, "hidden_per_branch": 256,
                   "terrain": TERRAIN_FEATURE_DIM, "terrain_relief": NAV_GEOMETRY_DIM,
                   "relief_rows": RELIEF_ROWS, "relief_cols": RELIEF_COLS},
        "periods": {"physics": PHYSICS_DT, "locomotion": CONTROL_DT,
                    "navigation": NAV_DT, "depth": DEPTH_DT},
        "normalization": {"omega": 0.25, "qrel": 1.0, "dq": 0.05,
                          "previous_action": "clip(-3,3)/3", "command": [0.8, 0.3, 1.0],
                          "goal_xy": 10.0, "goal_age": 0.5, "height": 1.0, "range": 6.0,
                          "relief": "ground_gradient_clamped_to_1",
                          "gait_phase": "sin_cos", "initial_heading": "body_frame_xy"},
        "frames": {"robot": "x-forward_y-left_z-up", "depth": "optical_axis_z"},
    }
