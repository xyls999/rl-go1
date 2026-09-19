"""Strict loader for the Stage-1 reward contract."""

from __future__ import annotations

import math
from pathlib import Path
import tomllib


REWARD_FIELDS = {
    "track_linear_velocity": {"weight", "std"},
    "track_yaw_velocity": {"weight", "std"},
    "forward_velocity": {"weight", "target_velocity", "negative_velocity_penalty"},
    "forward_progress": {"weight", "max_progress_speed"},
    "goal_closeness": {"weight", "max_distance"},
    "course_checkpoint": {"weight", "first", "second", "third"},
    "goal_reached": {"weight", "start_offset", "end_offset", "half_width"},
    "straight_line": {"weight", "std"},
    "velocity_direction": {"weight", "std", "min_speed"},
    "cross_track_l2": {"weight", "deadband", "scale", "max_cost"},
    "heading_alignment": {"weight", "std"},
    "upright": {"weight"},
    "phase_contact": {
        "weight", "flat_hz", "terrain_hz", "flat_duty_factor",
        "terrain_duty_factor", "transition_width", "force_scale", "slip_scale",
    },
    "phase_clearance": {"weight", "flat_height", "terrain_height", "std"},
    "base_height": {"weight", "target_height", "minimum_height", "std", "low_height_penalty"},
    "foot_load_balance": {"weight", "ema_alpha", "tolerance", "scale", "max_cost", "force_threshold"},
    "trot_gait": {"weight", "std", "max_err", "velocity_threshold", "terrain_relief"},
    "foot_clearance": {
        "weight", "target_height", "min_height", "terrain_target_height",
        "terrain_min_height", "std", "velocity_tanh",
    },
    "small_steps": {"weight", "target_length", "min_length", "max_length"},
    "step_frequency": {"weight", "target_hz", "min_hz", "max_hz", "window_seconds"},
    "four_feet_grounded": {"weight", "grace_seconds", "ramp_seconds"},
    "foot_contact_cycle": {"weight", "max_air_time", "ramp_seconds", "terrain_relief"},
    "air_time_variance": {"weight", "max_time", "terrain_relief"},
    "flat_trot_pattern": {"weight", "terrain_relief"},
    "foot_lateral": {
        "weight", "max_outward_deviation", "max_inward_deviation",
        "cross_body_penalty", "terrain_relief",
    },
    "action_rate_l2": {"weight"},
    "action_acceleration_l2": {"weight"},
    "feet_air_time": {"weight", "threshold", "max_air_time", "command_threshold"},
    "feet_slide": {"weight", "contact_force_threshold", "deadband", "scale", "terrain_relief"},
    "four_feet_airborne": {"weight"},
    "base_vertical_velocity_l2": {"weight"},
    "base_roll_pitch_rate_l2": {"weight", "scale"},
    "base_orientation_l2": {"weight", "scale"},
    "joint_torque_l2": {"weight"},
    "joint_deviation_l2": {"weight", "deadband", "scale"},
    "hip_abduction_deviation_l2": {"weight", "deadband", "scale"},
    "hip_abduction_velocity_l2": {"weight", "scale"},
    "joint_power_l1": {"weight"},
    "joint_acceleration_l2": {"weight"},
    "fallen": {"weight"},
    "collision": {"weight"},
    "out_of_bounds": {"weight"},
}

POSITIVE_TERMS = {
    "track_linear_velocity", "track_yaw_velocity", "forward_velocity", "forward_progress",
    "goal_closeness", "course_checkpoint", "goal_reached", "straight_line",
    "velocity_direction", "heading_alignment", "upright", "phase_contact", "phase_clearance",
    "base_height", "trot_gait", "foot_clearance", "feet_air_time", "small_steps", "step_frequency",
}


def load_stage1_rewards(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 reward config is missing: {path}")
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != set(REWARD_FIELDS):
        missing = sorted(set(REWARD_FIELDS) - set(raw))
        unknown = sorted(set(raw) - set(REWARD_FIELDS))
        raise ValueError(f"reward tables mismatch; missing={missing}, unknown={unknown}")
    result = {}
    for name, expected_fields in REWARD_FIELDS.items():
        table = raw[name]
        if not isinstance(table, dict) or set(table) != expected_fields:
            raise ValueError(f"reward {name} fields must be {sorted(expected_fields)}")
        values = {}
        for key, value in table.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"reward {name}.{key} must be a finite number")
            values[key] = float(value)
        weight = values["weight"]
        if name in POSITIVE_TERMS and weight < 0:
            raise ValueError(f"reward {name}.weight must be nonnegative")
        if name not in POSITIVE_TERMS and weight > 0:
            raise ValueError(f"penalty {name}.weight must be nonpositive")
        result[name] = values
    for name in ("track_linear_velocity", "track_yaw_velocity", "straight_line", "velocity_direction", "heading_alignment",
                 "trot_gait", "foot_clearance", "phase_clearance", "base_height"):
        if result[name]["std"] <= 0:
            raise ValueError(f"reward {name}.std must be positive")
    for name, field in (
        ("forward_velocity", "target_velocity"), ("forward_velocity", "negative_velocity_penalty"),
        ("forward_progress", "max_progress_speed"),
        ("goal_closeness", "max_distance"),
        ("goal_reached", "start_offset"), ("goal_reached", "end_offset"),
        ("goal_reached", "half_width"),
        ("velocity_direction", "min_speed"),
        ("cross_track_l2", "deadband"), ("cross_track_l2", "scale"),
        ("cross_track_l2", "max_cost"),
        ("trot_gait", "max_err"), ("trot_gait", "velocity_threshold"),
        ("phase_contact", "flat_hz"), ("phase_contact", "terrain_hz"),
        ("phase_contact", "transition_width"), ("phase_contact", "force_scale"),
        ("phase_contact", "slip_scale"),
        ("phase_clearance", "flat_height"), ("phase_clearance", "terrain_height"),
        ("base_height", "target_height"), ("base_height", "minimum_height"),
        ("base_height", "low_height_penalty"),
        ("foot_load_balance", "tolerance"), ("foot_load_balance", "scale"),
        ("foot_load_balance", "max_cost"), ("foot_load_balance", "force_threshold"),
        ("foot_clearance", "target_height"), ("foot_clearance", "min_height"),
        ("foot_clearance", "terrain_target_height"), ("foot_clearance", "terrain_min_height"),
        ("foot_clearance", "velocity_tanh"),
        ("feet_slide", "contact_force_threshold"), ("feet_slide", "deadband"),
        ("feet_slide", "scale"), ("feet_slide", "terrain_relief"),
        ("small_steps", "target_length"), ("small_steps", "min_length"), ("small_steps", "max_length"),
        ("step_frequency", "target_hz"), ("step_frequency", "min_hz"), ("step_frequency", "max_hz"), ("step_frequency", "window_seconds"),
        ("four_feet_grounded", "grace_seconds"), ("four_feet_grounded", "ramp_seconds"),
        ("foot_contact_cycle", "max_air_time"), ("foot_contact_cycle", "ramp_seconds"),
        ("air_time_variance", "max_time"),
        ("foot_lateral", "max_outward_deviation"), ("foot_lateral", "max_inward_deviation"), ("foot_lateral", "cross_body_penalty"),
        ("base_roll_pitch_rate_l2", "scale"), ("base_orientation_l2", "scale"),
        ("joint_deviation_l2", "deadband"), ("joint_deviation_l2", "scale"),
        ("hip_abduction_deviation_l2", "deadband"), ("hip_abduction_deviation_l2", "scale"),
        ("hip_abduction_velocity_l2", "scale"),
    ):
        if result[name][field] <= 0:
            raise ValueError(f"reward {name}.{field} must be positive")
    checkpoints = result["course_checkpoint"]
    if not 0 < checkpoints["first"] < checkpoints["second"] < checkpoints["third"] < 1:
        raise ValueError("course checkpoint fractions must be strictly increasing inside (0, 1)")
    goal = result["goal_reached"]
    if goal["end_offset"] <= goal["start_offset"]:
        raise ValueError("goal_reached.end_offset must be greater than start_offset")
    air = result["feet_air_time"]
    if air["threshold"] < 0 or air["max_air_time"] <= air["threshold"] or air["command_threshold"] < 0:
        raise ValueError("feet_air_time thresholds are inconsistent")
    frequency = result["step_frequency"]
    if not 0 < frequency["min_hz"] <= frequency["target_hz"] < frequency["max_hz"]:
        raise ValueError("step_frequency must satisfy 0 < min_hz <= target_hz < max_hz")
    steps = result["small_steps"]
    if not 0 < steps["min_length"] < steps["target_length"] < steps["max_length"]:
        raise ValueError("small_steps must satisfy 0 < min_length < target_length < max_length")
    if result["foot_clearance"]["min_height"] >= result["foot_clearance"]["target_height"]:
        raise ValueError("foot_clearance.min_height must be less than target_height")
    if result["foot_clearance"]["terrain_min_height"] >= result["foot_clearance"]["terrain_target_height"]:
        raise ValueError("foot_clearance.terrain_min_height must be less than terrain_target_height")
    if result["foot_clearance"]["terrain_target_height"] < result["foot_clearance"]["target_height"]:
        raise ValueError("terrain foot-clearance target must not be below the flat-ground target")
    phase = result["phase_contact"]
    if not 0 < phase["terrain_hz"] <= phase["flat_hz"] <= 4.0:
        raise ValueError("phase_contact frequencies must satisfy 0 < terrain_hz <= flat_hz <= 4")
    if not 0 < phase["flat_duty_factor"] < 1 or not 0 < phase["terrain_duty_factor"] < 1:
        raise ValueError("phase_contact duty factors must be inside (0, 1)")
    if not 0 < phase["transition_width"] < 0.25:
        raise ValueError("phase_contact.transition_width must be inside (0, 0.25)")
    if result["phase_clearance"]["terrain_height"] < result["phase_clearance"]["flat_height"]:
        raise ValueError("terrain phase clearance must not be below flat phase clearance")
    if result["base_height"]["minimum_height"] >= result["base_height"]["target_height"]:
        raise ValueError("base_height.minimum_height must be below target_height")
    if not 0 < result["foot_load_balance"]["ema_alpha"] < 1:
        raise ValueError("foot_load_balance.ema_alpha must be inside (0, 1)")
    if result["foot_lateral"]["max_outward_deviation"] <= 0 or result["foot_lateral"]["max_inward_deviation"] <= 0 or result["foot_lateral"]["cross_body_penalty"] <= 0:
        raise ValueError("foot_lateral limits and cross_body_penalty must be positive")
    for name in ("trot_gait", "foot_contact_cycle", "air_time_variance", "flat_trot_pattern", "foot_lateral"):
        if not 0 <= result[name]["terrain_relief"] <= 1:
            raise ValueError(f"reward {name}.terrain_relief must be inside [0, 1]")
    return result


def require_matching_reward_config(metadata: dict, current: dict[str, dict[str, float]]) -> None:
    """Reject a Stage-1 resume whose reward contract differs from this run."""
    saved = metadata.get("reward_config")
    if saved is None:
        raise ValueError(
            "Stage-1 resume checkpoint has no reward_config metadata; start a new run "
            "instead of mixing an older objective with the current reward contract"
        )
    if saved != current:
        raise ValueError(
            "Stage-1 resume checkpoint reward_config differs from configs/rewards_stage1.toml; "
            "restore the saved configuration or start a new run"
        )
