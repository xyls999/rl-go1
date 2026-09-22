"""Isaac Lab configuration using the installed Unitree Go1 asset and actuators."""

from __future__ import annotations

import math
from pathlib import Path
import os

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg

from ..contracts import CONTROL_DT, DEPTH_DT, LOCO_CRITIC_DIM, LOCO_OBS_DIM, MAX_FORWARD_COMMAND, PHYSICS_DT, TERRAIN_FEATURE_DIM
from ..reward_config import REWARD_FIELDS, load_stage1_rewards
from .environment import HierarchicalActionCfg
from .geometry import CAMERA_POSITION, CAMERA_ROTATION, TILE_SIZE, build_scene


class UnifiedTerrainGenerator:
    def __init__(self, cfg, device):
        self.terrain_mesh, origins = build_scene(
            cfg.environment_count, cfg.mode, cfg.obstacles, cfg.seed, cfg.difficulty,
            cfg.course_difficulty_min, cfg.course_difficulty_max,
        )
        self.terrain_origins = origins[:, None, :]
        self.flat_patches = {}


@configclass
class UnifiedTerrainCfg(TerrainGeneratorCfg):
    class_type: type = UnifiedTerrainGenerator
    environment_count: int = 4
    mode: str = "mixed"
    obstacles: bool = False
    difficulty: float = 0.5
    course_difficulty_min: float | None = None
    course_difficulty_max: float | None = None
    size: tuple[float, float] = (TILE_SIZE, TILE_SIZE)
    sub_terrains: dict = {}


class DistinctTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        self.obstacle_faces = torch.as_tensor(mesh.metadata["obstacle_faces"], device=self.device)
        super().import_mesh(name, mesh)

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.terrain_origins.shape[0] != cfg.num_envs:
            raise ValueError("A distinct terrain tile is required for each rendered environment.")
        # Standard terrain sampling can overlap replicas. Cameras must not see
        # other replicated dogs standing at exactly the same physical location.
        self.env_origins = self.terrain_origins[:, 0, :].clone()


class FixedWorldVelocityCommand(UniformVelocityCommand):
    """Use truthful world-frame arrows for the fixed-direction Stage-1 task."""

    def _world_arrow(self, xy_velocity):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.vector_norm(xy_velocity, dim=1) * 3.0
        heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading)
        return arrow_scale, math_utils.quat_from_euler_xyz(zeros, zeros, heading)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        if self._env.cfg.hier_stage == "locomotion":
            forward = self._env.hier._forward_axis
            lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
            desired_w = self.command[:, :1] * forward + self.command[:, 1:2] * lateral
        else:
            yaw = self.robot.data.heading_w
            cosine, sine = torch.cos(yaw), torch.sin(yaw)
            desired_w = torch.stack((
                cosine * self.command[:, 0] - sine * self.command[:, 1],
                sine * self.command[:, 0] + cosine * self.command[:, 1],
            ), dim=-1)
        desired_scale, desired_quat = self._world_arrow(desired_w)
        current_scale, current_quat = self._world_arrow(self.robot.data.root_lin_vel_w[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, desired_quat, desired_scale)
        self.current_vel_visualizer.visualize(base_pos_w, current_quat, current_scale)


def actor_observation(env):
    env.hier.refresh()
    return env.hier.locomotion_obs if env.cfg.hier_stage == "locomotion" else env.hier.navigation_obs


def critic_observation(env):
    env.hier.refresh()
    actor = env.hier.teacher_locomotion_obs if env.cfg.hier_stage == "locomotion" else env.hier.teacher_navigation_obs
    return torch.cat((actor, env.hier.privileged_state()), dim=-1)


def navigation_reward(env):
    return env.hier.compute_reward()


def locomotion_reward(env, name: str):
    return env.hier.reward_term(name)


def task_termination(env):
    metrics = env.hier.get_metrics()
    return metrics["fallen"] | metrics["out_of_bounds"] | metrics["collision"] | metrics["success"]


@configclass
class PolicyObservationCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=actor_observation)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class CriticObservationCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=critic_observation)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy = PolicyObservationCfg()
    critic = CriticObservationCfg()


@configclass
class RewardsCfg:
    navigation_task = RewardTermCfg(func=navigation_reward, weight=1.0)
    track_linear_velocity = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "track_linear_velocity"}
    )
    track_yaw_velocity = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "track_yaw_velocity"}
    )
    upright = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "upright"})
    phase_contact = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "phase_contact"})
    phase_clearance = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "phase_clearance"})
    base_height = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "base_height"})
    foot_load_balance = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "foot_load_balance"})
    action_rate_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "action_rate_l2"})
    action_acceleration_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "action_acceleration_l2"})
    feet_air_time = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "feet_air_time"})
    feet_slide = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "feet_slide"})
    joint_torque_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "joint_torque_l2"})
    joint_power_l1 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "joint_power_l1"})
    joint_acceleration_l2 = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "joint_acceleration_l2"}
    )
    fallen = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "fallen"})
    collision = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "collision"})
    forward_velocity = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "forward_velocity"})
    forward_progress = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "forward_progress"})
    goal_closeness = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "goal_closeness"})
    course_checkpoint = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "course_checkpoint"})
    goal_reached = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "goal_reached"})
    straight_line = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "straight_line"})
    velocity_direction = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "velocity_direction"})
    cross_track_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "cross_track_l2"})
    heading_alignment = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "heading_alignment"})
    trot_gait = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "trot_gait"})
    foot_clearance = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "foot_clearance"})
    small_steps = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "small_steps"})
    step_frequency = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "step_frequency"})
    four_feet_grounded = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "four_feet_grounded"})
    foot_contact_cycle = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "foot_contact_cycle"})
    air_time_variance = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "air_time_variance"})
    flat_trot_pattern = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "flat_trot_pattern"})
    foot_lateral = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "foot_lateral"})
    four_feet_airborne = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "four_feet_airborne"})
    base_vertical_velocity_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "base_vertical_velocity_l2"})
    base_roll_pitch_rate_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "base_roll_pitch_rate_l2"})
    base_orientation_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "base_orientation_l2"})
    joint_deviation_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "joint_deviation_l2"})
    hip_abduction_deviation_l2 = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "hip_abduction_deviation_l2"}
    )
    hip_abduction_velocity_l2 = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "hip_abduction_velocity_l2"}
    )
    out_of_bounds = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "out_of_bounds"})

    # --- hier_v2 contracts -------------------------------------------------
    # nav_v2 terms (obstacle-avoidance stage, three-dimensional velocity action)
    track_lin_vel_xy = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "track_lin_vel_xy"})
    track_ang_vel_z = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "track_ang_vel_z"})
    path_progress = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "path_progress"})
    bearing_progress = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "bearing_progress"})
    reach_goal = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "reach_goal"})
    settled_goal = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "settled_goal"})
    goal_braking = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "goal_braking"})
    stuck_duration = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "stuck_duration"})
    unsafe_tilt = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "unsafe_tilt"})
    tilt_recovery = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "tilt_recovery"})
    wall_proximity_brake = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "wall_proximity_brake"}
    )
    obstacle_evasion = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "obstacle_evasion"})
    navigation_time = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "navigation_time"})
    command_rate_l2 = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "command_rate_l2"})
    nav_termination = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "termination"})
    # loco_v2 terms use a loco2_ prefix so the validated Stage-1 names stay intact
    loco2_feet_air_time = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_feet_air_time"})
    loco2_feet_clearance = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_feet_clearance"})
    loco2_feet_slide = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_feet_slide"})
    loco2_feet_stumble = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_feet_stumble"})
    loco2_joint_acc = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_joint_acc"})
    loco2_joint_torques = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_joint_torques"})
    loco2_joint_power = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_joint_power"})
    loco2_dof_pos_limits = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_dof_pos_limits"})
    loco2_torque_margin = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_torque_margin"})
    loco2_lin_vel_z = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_lin_vel_z"})
    loco2_ang_vel_xy = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_ang_vel_xy"})
    loco2_flat_orientation = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_flat_orientation"}
    )
    loco2_base_height = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_base_height"})
    loco2_hip_to_default = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_hip_to_default"}
    )
    loco2_action_rate = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_action_rate"})
    loco2_action_smoothness = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_action_smoothness"}
    )
    loco2_action_magnitude_limit = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_action_magnitude_limit"}
    )
    loco2_undesired_contacts = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_undesired_contacts"}
    )
    loco2_termination = RewardTermCfg(func=locomotion_reward, weight=0.0, params={"name": "loco2_termination"})
    loco2_track_lin_vel_xy = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_track_lin_vel_xy"}
    )
    loco2_track_ang_vel_z = RewardTermCfg(
        func=locomotion_reward, weight=0.0, params={"name": "loco2_track_ang_vel_z"}
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    task = TerminationTermCfg(func=task_termination)


@configclass
class HierarchicalActionsCfg:
    motion = HierarchicalActionCfg(asset_name="robot")


@configclass
class HierarchicalEnvCfg(UnitreeGo1RoughEnvCfg):
    hier_stage: str = "locomotion"
    hier_purpose: str = "integration"
    hier_terrain: str = "course"
    hier_cameras: bool = False
    # This is explicitly a simulated relative-XY UWB measurement. Hardware
    # transport and localization remain outside the simulator environment.
    hier_uwb_noise_std: float = 0.02
    hier_uwb_dropout_probability: float = 0.01
    hier_stage1_rewards: dict = {}
    hier_speed_min: float = 0.65
    hier_speed_max: float = 0.65
    # "stage1" keeps the validated contract; "loco_v2" / "nav_v2" select the
    # fwwb-inspired hier_v2 contracts implemented in hier_v2/rewards_v2.py.
    hier_reward_contract: str = "stage1"
    hier_nav_v2_rewards: dict = {}
    hier_loco_v2_rewards: dict = {}
    # Sim-to-real curriculum for the navigation actor (see hier_v2/robustness_v2.py).
    hier_robustness_enabled: bool = False


NAV_V2_CFG_TERMS = {
    "track_lin_vel_xy", "track_ang_vel_z", "path_progress", "bearing_progress",
    "reach_goal", "settled_goal", "goal_braking", "stuck_duration", "unsafe_tilt",
    "tilt_recovery", "wall_proximity_brake", "obstacle_evasion", "navigation_time",
    "command_rate_l2", "nav_termination",
}

LOCO_V2_CFG_TERMS = {
    "loco2_feet_air_time", "loco2_feet_clearance", "loco2_feet_slide",
    "loco2_feet_stumble", "loco2_joint_acc", "loco2_joint_torques",
    "loco2_joint_power", "loco2_dof_pos_limits", "loco2_torque_margin",
    "loco2_lin_vel_z", "loco2_ang_vel_xy", "loco2_flat_orientation",
    "loco2_base_height", "loco2_hip_to_default", "loco2_action_rate",
    "loco2_action_smoothness", "loco2_action_magnitude_limit",
    "loco2_undesired_contacts", "loco2_termination", "loco2_track_lin_vel_xy",
    "loco2_track_ang_vel_z",
}


def make_config(stage, num_envs, device, enable_cameras, seed, terrain, purpose,
                episode_seconds=None, show_direction_arrows=False,
                course_difficulty_min=None, course_difficulty_max=None,
                speed_min=None, speed_max=None, reward_contract=None,
                robustness=False):
    cfg = HierarchicalEnvCfg()
    cfg.hier_robustness_enabled = bool(robustness)
    cfg.hier_stage = stage
    cfg.hier_purpose = purpose
    cfg.hier_terrain = terrain
    cfg.hier_cameras = enable_cameras
    reward_path = Path(os.environ.get(
        "GO1_REWARD_CONFIG_PATH",
        Path(__file__).resolve().parents[4] / "configs" / "rewards_stage1.toml",
    ))
    cfg.hier_stage1_rewards = load_stage1_rewards(reward_path)
    target_velocity = cfg.hier_stage1_rewards["forward_velocity"]["target_velocity"]
    speed_min = target_velocity if speed_min is None else float(speed_min)
    speed_max = target_velocity if speed_max is None else float(speed_max)
    if not 0 < speed_min <= speed_max <= MAX_FORWARD_COMMAND:
        raise ValueError(
            f"speed range must satisfy 0 < min <= max <= {MAX_FORWARD_COMMAND} m/s"
        )
    if (course_difficulty_min is None) != (course_difficulty_max is None):
        raise ValueError("course difficulty bounds must be provided together")
    if course_difficulty_min is not None:
        course_difficulty_min = float(course_difficulty_min)
        course_difficulty_max = float(course_difficulty_max)
        if not 0 <= course_difficulty_min <= course_difficulty_max <= 1:
            raise ValueError("course difficulty must satisfy 0 <= min <= max <= 1")
    cfg.hier_speed_min = speed_min
    cfg.hier_speed_max = speed_max
    if stage == "locomotion" and purpose == "train" and terrain != "course":
        raise ValueError("Stage-1 training requires the continuous straight course terrain")
    if purpose == "integration":
        cfg.hier_uwb_noise_std = 0.0
        cfg.hier_uwb_dropout_probability = 0.0
    cfg.seed = seed
    cfg.sim.device = device
    cfg.sim.dt = PHYSICS_DT
    cfg.sim.render_interval = round(DEPTH_DT / PHYSICS_DT)
    cfg.decimation = 4 if stage == "locomotion" else 20
    default_episode_seconds = 30.0 if stage == "locomotion" else 60.0
    if episode_seconds is not None and episode_seconds <= 0:
        raise ValueError("episode_seconds must be positive")
    cfg.episode_length_s = float(episode_seconds or default_episode_seconds)
    cfg.num_rerenders_on_reset = 2 if enable_cameras else 0
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 44.0
    cfg.scene.terrain = TerrainImporterCfg(
        class_type=DistinctTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UnifiedTerrainCfg(
            seed=seed,
            environment_count=num_envs,
            mode=terrain,
            # The speed layer commands vx/wz from the terrain relief map; it does not
        # avoid obstacles, so the scene never spawns obstacle boxes.
        obstacles=False,
            difficulty=0.75 if terrain == "course" else 0.5,
            course_difficulty_min=course_difficulty_min,
            course_difficulty_max=course_difficulty_max,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.42)),
    )
    # 0.1 m resolution over a 1.5 x 3.1 m window keeps 16 x 32 = 512 samples and
    # resolves a 0.16 m stair tread that a 0.2 m grid aliases away. The window is
    # centred 1.05 m ahead, so it spans from just behind the feet to about
    # 2.6 m in front of the robot.
    cfg.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=0.1, size=(1.5, 3.1), ordering="yx"
    )
    cfg.scene.height_scanner.offset.pos = (1.05, 0.0, 20.0)
    cfg.scene.height_scanner.update_period = CONTROL_DT
    cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
    cfg.scene.height_scanner.debug_vis = False
    cfg.scene.contact_forces.update_period = PHYSICS_DT
    cfg.scene.contact_forces.track_air_time = True
    cfg.scene.contact_forces.history_length = round(CONTROL_DT / PHYSICS_DT)
    cfg.scene.sky_light.spawn.texture_file = None
    if enable_cameras:
        cfg.scene.depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/depth_camera",
            update_period=DEPTH_DT,
            width=80,
            height=60,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="none",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, horizontal_aperture=24.0,
                vertical_aperture=18.0, clipping_range=(0.2, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=CAMERA_POSITION, rot=CAMERA_ROTATION, convention="world"),
        )
    cfg.actions = HierarchicalActionsCfg()
    cfg.actions.motion.stage = stage
    cfg.observations = ObservationsCfg()
    cfg.rewards = RewardsCfg()
    contract = reward_contract or "stage1"
    if contract not in {"stage1", "loco_v2", "nav_v2"}:
        raise ValueError(f"unknown reward contract: {contract}")
    if contract != "stage1" and stage != ("navigation" if contract == "nav_v2" else "locomotion"):
        raise ValueError(f"reward contract {contract} is incompatible with stage {stage}")
    cfg.hier_reward_contract = contract
    config_root = Path(__file__).resolve().parents[4] / "configs"
    if contract == "nav_v2":
        from ..hier_v2.rewards_v2 import load_nav_v2_rewards

        cfg.hier_nav_v2_rewards = load_nav_v2_rewards(config_root / "rewards_nav_v2.toml")
        for name in REWARD_FIELDS:
            setattr(cfg.rewards, name, None)
        cfg.rewards.navigation_task = None
        for name in NAV_V2_CFG_TERMS:
            contract_name = "termination" if name == "nav_termination" else name
            getattr(cfg.rewards, name).weight = cfg.hier_nav_v2_rewards[contract_name]["weight"]
    elif contract == "loco_v2":
        from ..hier_v2.rewards_v2 import load_loco_v2_rewards

        cfg.hier_loco_v2_rewards = load_loco_v2_rewards(config_root / "rewards_loco_v2.toml")
        cfg.rewards.navigation_task = None
        for name in REWARD_FIELDS:
            getattr(cfg.rewards, name).weight = 0.0
        for name, table in cfg.hier_loco_v2_rewards.items():
            getattr(cfg.rewards, f"loco2_{name}").weight = table["weight"]
    elif stage == "locomotion":
        cfg.rewards.navigation_task = None
        for name in REWARD_FIELDS:
            getattr(cfg.rewards, name).weight = cfg.hier_stage1_rewards[name]["weight"]
    else:
        for name in REWARD_FIELDS:
            setattr(cfg.rewards, name, None)
    cfg.terminations = TerminationsCfg()
    cfg.curriculum = {}
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.class_type = FixedWorldVelocityCommand
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.debug_vis = bool(show_direction_arrows)
    if stage == "locomotion":
        cfg.commands.base_velocity.ranges.lin_vel_x = (speed_min, speed_max)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands.base_velocity.resampling_time_range = (30.0, 30.0)
    else:
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        cfg.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        cfg.commands.base_velocity.resampling_time_range = (6.0, 10.0)
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.05, 0.05), "y": (-0.1, 0.1),
        "yaw": (-0.05, 0.05) if purpose == "train" else (0.0, 0.0),
    }
    cfg.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 3.0)
    cfg.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
    cfg.events.base_external_force_torque.params["torque_range"] = (-0.2, 0.2)
    if purpose == "integration":
        cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        cfg.events.add_base_mass = None
        cfg.events.base_external_force_torque = None
    return cfg

