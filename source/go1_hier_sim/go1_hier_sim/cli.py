"""Single checked entry point for simulation, training, evaluation and export."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import traceback


WORKSPACE = Path(__file__).resolve().parents[3]


def _parser():
    parser = argparse.ArgumentParser(description="Go1 hierarchical visual navigation")
    # AppLauncher performs an early parse before adding its flags, so this positional
    # cannot be required until the final parse in main().
    parser.add_argument("operation", nargs="?", choices=("check", "train", "distill", "evaluate", "export"))
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3a", "stage3b", "stage4"))
    parser.add_argument("--branch", choices=("terrain", "obstacle"))
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None,
                        help="training/evaluation episode time limit in simulated seconds")
    parser.add_argument("--record-video",
                        help="write an Isaac Sim rgb_array third-person MP4 to this path")
    parser.add_argument("--video-seconds", type=float, default=60.0,
                        help="internal simulation video duration in simulated seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terrain", choices=("course", "flat", "slopes", "rough", "stairs", "mixed"),
                        default="course")
    parser.add_argument("--reward-contract", choices=("stage1", "loco_v2", "nav_v2"), default=None,
                        help="reward/model contract; defaults to stage1 (locomotion) or nav_v2 (navigation)")
    parser.add_argument("--allow-uncertified-locomotion", action="store_true",
                        help="permit an unvalidated frozen locomotion checkpoint (prints its sha256)")
    parser.add_argument("--robustness", choices=("off", "on"), default="off",
                        help="apply the sim-to-real robustness curriculum to the navigation actor")
    parser.add_argument("--course-difficulty-min", type=float, default=None,
                        help="minimum per-environment straight-course difficulty in [0,1]")
    parser.add_argument("--course-difficulty-max", type=float, default=None,
                        help="maximum per-environment straight-course difficulty in [0,1]")
    parser.add_argument("--speed-min", type=float, default=None,
                        help="minimum fixed-world forward command in m/s")
    parser.add_argument("--speed-max", type=float, default=None,
                        help="maximum fixed-world forward command in m/s")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from",
                        help="warm-start policy/value weights but reset optimizer, iteration and action noise")
    parser.add_argument("--critic-warmup-iterations", type=int, default=0,
                        help="with --initialize-from, update only the critic for this many initial iterations")
    parser.add_argument("--locomotion-checkpoint")
    parser.add_argument("--navigation-checkpoint")
    parser.add_argument("--vision-checkpoint")
    parser.add_argument("--teacher-replay")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--follow-env", type=int, default=None,
                        help="follow this environment in the Isaac Sim viewport during evaluation")
    parser.add_argument("--debug-motion", action="store_true",
                        help="print evaluation motion and reward diagnostics")
    parser.add_argument("--show-direction-arrows", action="store_true",
                        help="show fixed world target and actual velocity arrows in evaluation renders")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _required(args, *names):
    missing = [name for name in names if getattr(args, name.replace("-", "_")) in (None, "")]
    if missing:
        raise ValueError("missing required arguments: " + ", ".join("--" + name for name in missing))


def _video_frame_count(seconds: float, step_dt: float) -> int:
    if seconds <= 0 or step_dt <= 0:
        raise ValueError("video seconds and environment step must be positive")
    return max(1, round(seconds / step_dt))


def _contract_table(cfg, contract):
    """The reward table the environment actually applied for this contract.

    Saving and resuming must both read the effective contract, otherwise a
    resume validates a checkpoint against a table the run never used.
    """
    if contract == "loco_v2":
        return cfg.hier_loco_v2_rewards
    if contract == "nav_v2":
        return cfg.hier_nav_v2_rewards
    return cfg.hier_stage1_rewards


def _run_id(args):
    shared = os.environ.get("GO1_RUN_ID")
    if shared:
        return shared
    suffix = f"_{args.run_name}" if args.run_name else ""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix


def _ppo_values():
    path = Path(os.environ.get("GO1_PPO_CONFIG_PATH", WORKSPACE / "configs" / "ppo.json"))
    if not path.is_file():
        raise FileNotFoundError(f"PPO config is missing: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("PPO config must be a JSON object")
    return values


def _ppo_config(stage, args, values=None):
    values = values or _ppo_values()
    kind = "locomotion" if stage == "stage1" else "navigation"
    stage_values = values[kind]
    warm_start = args.initialize_from is not None
    return {
        "seed": args.seed,
        "device": args.device,
        "num_steps_per_env": int(stage_values["num_steps_per_env"]),
        "max_iterations": args.iterations,
        "empirical_normalization": False,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "clip_actions": 3.0 if stage == "stage1" else 1.0,
        "save_interval": int(values["save_interval"]),
        "experiment_name": stage,
        "run_name": args.run_name,
        "logger": "tensorboard",
        "policy": {
            "class_name": ("HierActorCriticV2"
                           if getattr(args, "reward_contract", None) in {"loco_v2", "nav_v2"}
                           else "HierActorCritic"),
            "stage": kind,
            "init_noise_std": float(
                values["warm_start_noise_std"] if warm_start else 1.0
            ),
            "min_noise_std": float(values["min_noise_std"]),
            "max_noise_std": float(values["max_noise_std"]),
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": float(values["value_loss_coef"]),
            "use_clipped_value_loss": True,
            "clip_param": float(
                values["warm_start_clip_param"] if warm_start else values["clip_param"]
            ),
            "entropy_coef": float(
                values["warm_start_entropy_coef"] if warm_start else values["entropy_coef"]
            ),
            "num_learning_epochs": int(values["num_learning_epochs"]),
            "num_mini_batches": int(values["num_mini_batches"]),
            "learning_rate": float(
                values["warm_start_learning_rate"] if warm_start else values["learning_rate"]
            ), "schedule": "fixed" if warm_start else "adaptive",
            "gamma": float(stage_values["gamma"]), "lam": float(values["lam"]),
            "desired_kl": float(
                values["warm_start_desired_kl"] if warm_start else values["desired_kl"]
            ), "max_grad_norm": float(values["max_grad_norm"]),
            "normalize_advantage_per_mini_batch": False, "rnd_cfg": None, "symmetry_cfg": None,
        },
    }


def _resolve_contract(args):
    """Evaluation and checking must use the same reward contract as training."""
    return args.reward_contract or ("nav_v2" if args.stage == "stage2" else "stage1")


def _load_dependency(path, kind, device, validated=True):
    from .checkpoints import load_actor

    return load_actor(path, kind, device=device, require_validated=validated)


def _check(args):
    import torch
    from .contracts import LOCO_CRITIC_DIM, LOCO_OBS_DIM, TERRAIN_FEATURE_DIM
    from .envs import make_env

    count = args.num_envs or 4
    check_stage = args.stage or "stage1"
    if check_stage not in {"stage1", "stage2"}:
        raise ValueError("check supports --stage stage1 or stage2")
    kind = "locomotion" if check_stage == "stage1" else "navigation"
    locomotion = None
    if check_stage == "stage2":
        _required(args, "locomotion-checkpoint")
        # This is data-path verification only. The report cannot certify policy
        # capability, but the checkpoint format and hashes are still enforced.
        locomotion = _load_dependency(args.locomotion_checkpoint, "locomotion", args.device, validated=False)
    env = make_env(stage=kind, num_envs=count, device=args.device,
                   enable_cameras=args.enable_cameras, seed=args.seed,
                   terrain=args.terrain, purpose="integration", locomotion_policy=locomotion,
                   course_difficulty_min=args.course_difficulty_min,
                   course_difficulty_max=args.course_difficulty_max,
                   speed_min=args.speed_min, speed_max=args.speed_max,
                   reward_contract=_resolve_contract(args))
    try:
        observation, _ = env.reset()
        expected_policy = LOCO_OBS_DIM if check_stage == "stage1" else 193
        expected_critic = LOCO_CRITIC_DIM if check_stage == "stage1" else 200
        if (observation["policy"].shape != (count, expected_policy) or
                observation["critic"].shape != (count, expected_critic)):
            raise RuntimeError(f"unexpected observations: {observation.batch_size}")
        if not all(torch.isfinite(value).all() for value in observation.values()):
            raise FloatingPointError("environment returned a nonfinite observation")
        for _ in range(2):
            action_dim = 12 if check_stage == "stage1" else 3
            observation, reward, terminated, truncated, _ = env.step(
                torch.zeros(count, action_dim, device=env.device)
            )
            if not torch.isfinite(reward).all():
                raise FloatingPointError("environment returned a nonfinite reward")
        if args.enable_cameras:
            depth = env.hier.depth_input()
            if depth.shape != (count, 2, 60, 80) or not torch.isfinite(depth).all():
                raise RuntimeError(f"unexpected depth input: {tuple(depth.shape)}")
            if env.hier.camera_events_pending < 1:
                raise RuntimeError("camera was enabled but produced no timestamped event")
            terrain_teacher = env.hier.teacher_geometry("terrain", observable=True)
            if terrain_teacher.shape != (count, TERRAIN_FEATURE_DIM) or not torch.isfinite(terrain_teacher).all():
                raise RuntimeError("camera-observable terrain teacher is invalid")
            if check_stage == "stage2":
                obstacle_teacher = env.hier.teacher_geometry("obstacle", observable=True)
                if obstacle_teacher.shape != (count, 144) or not torch.isfinite(obstacle_teacher).all():
                    raise RuntimeError("camera-observable obstacle teacher is invalid")
        result = {"passed": True, "stage": check_stage, "capability_validation": False,
                  "num_envs": count, "policy_shape": list(observation["policy"].shape),
                  "critic_shape": list(observation["critic"].shape), "camera_enabled": args.enable_cameras,
                  "terrain": args.terrain, "device": args.device}
        output = WORKSPACE / "reports" / f"check_{_run_id(args)}.json"
        from .checkpoints import write_json
        write_json(output, result)
        print(f"CHECK PASSED: {output}")
    finally:
        env.close()


def _train(args):
    import torch

    if args.stage not in {"stage1", "stage2"}:
        raise ValueError("train supports --stage stage1 or stage2")
    if args.iterations is None or args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.resume and args.initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.critic_warmup_iterations < 0:
        raise ValueError("--critic-warmup-iterations must be nonnegative")
    if args.critic_warmup_iterations and not args.initialize_from:
        raise ValueError("--critic-warmup-iterations requires --initialize-from")
    if args.critic_warmup_iterations >= args.iterations:
        raise ValueError("--critic-warmup-iterations must be smaller than --iterations")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from .checkpoints import load_checkpoint
    from .envs import make_env
    from .reward_config import require_matching_reward_config
    from .training.on_policy import HierOnPolicyRunner

    ppo_values = _ppo_values()
    kind = "locomotion" if args.stage == "stage1" else "navigation"
    num_envs = args.num_envs or int(ppo_values[kind]["num_envs"])
    locomotion = None
    if args.stage == "stage2":
        _required(args, "locomotion-checkpoint")
        locomotion = _load_dependency(
            args.locomotion_checkpoint, "locomotion", args.device,
            validated=not args.allow_uncertified_locomotion,
        )
        if args.allow_uncertified_locomotion:
            from .checkpoints import file_hash

            print("WARNING: the frozen locomotion policy is not capability-certified; "
                  f"sha256={file_hash(args.locomotion_checkpoint)}")
    contract = args.reward_contract or ("nav_v2" if args.stage == "stage2" else "stage1")
    raw_env = make_env(stage="locomotion" if args.stage == "stage1" else "navigation",
                       num_envs=num_envs, device=args.device, enable_cameras=False,
                       locomotion_policy=locomotion, seed=args.seed, terrain=args.terrain,
                       purpose="train", episode_seconds=args.episode_seconds,
                       course_difficulty_min=args.course_difficulty_min,
                       course_difficulty_max=args.course_difficulty_max,
                       speed_min=args.speed_min, speed_max=args.speed_max,
                       reward_contract=contract, robustness=args.robustness == "on")
    run_id = _run_id(args)
    log_dir = WORKSPACE / "runs" / args.stage / run_id
    checkpoint_dir = WORKSPACE / "checkpoints" / args.stage / run_id

    def save_runner(runner, requested_path, metrics):
        if getattr(runner, "gpu_global_rank", 0) != 0:
            return
        from .checkpoints import save_checkpoint
        name = Path(requested_path).name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(checkpoint_dir / name, args.stage,
                        {runner.alg.policy.actor.stage: runner.alg.policy.actor,
                         "policy": runner.alg.policy}, runner.alg.optimizer,
                        {"iteration": runner.current_learning_iteration, "metrics": metrics,
                         "purpose": "train", "capability_validated": False,
                         "reward_config": _contract_table(raw_env.cfg, contract),
                         "reward_contract": contract,
                         "course_difficulty": [args.course_difficulty_min, args.course_difficulty_max],
                         "speed_range": [args.speed_min, args.speed_max],
                         "upstream": {"locomotion": args.locomotion_checkpoint,
                                      "initialized_from": args.initialize_from}})

    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=3.0 if args.stage == "stage1" else 1.0)
        runner = HierOnPolicyRunner(env, _ppo_config(args.stage, args, ppo_values), str(log_dir),
                                    device=args.device, checkpoint_callback=save_runner)
        if args.resume:
            resume_data = load_checkpoint(args.resume, expected_stage=args.stage)
            require_matching_reward_config(
                resume_data.get("metadata", {}), _contract_table(raw_env.cfg, contract)
            )
            runner.load(args.resume)
        elif args.initialize_from:
            initialized = load_checkpoint(
                args.initialize_from,
                expected_stage=args.stage,
                allow_locomotion_heading_migration=args.stage == "stage1",
            )
            state = dict(initialized["models"]["policy"])
            if args.stage == "stage1":
                from .models import migrate_locomotion_heading_state
                state = migrate_locomotion_heading_state(state)
            # A warm start intentionally keeps learned actor/critic weights but
            # discards the old objective's runaway exploration parameter.
            state.pop("log_std", None)
            # Bypass the RSL adapter's boolean load_state_dict return value so
            # the warm-start structural check can inspect missing keys.
            incompatible = torch.nn.Module.load_state_dict(
                runner.alg.policy, state, strict=False
            )
            if set(incompatible.missing_keys) != {"log_std"} or incompatible.unexpected_keys:
                raise ValueError(
                    "warm-start checkpoint policy structure is incompatible: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
            with torch.no_grad():
                runner.alg.policy.log_std.fill_(
                    math.log(float(ppo_values["warm_start_noise_std"]))
                )
            # Actor means from the best checkpoint are extremely sensitive:
            # an ordinary shared Adam rate moved the actor by only ~1e-4 in
            # relative L2 yet cut course completion sharply.  Give actor and
            # critic independent fixed rates so the new value function can
            # adapt 100x faster without erasing the gait.
            actor_parameters = list(runner.alg.policy.actor.parameters())
            actor_parameters.append(runner.alg.policy.log_std)
            runner.alg.optimizer = torch.optim.Adam([
                {
                    "params": actor_parameters,
                    "lr": float(ppo_values["warm_start_actor_learning_rate"]),
                },
                {
                    "params": runner.alg.policy.critic.parameters(),
                    "lr": float(ppo_values["warm_start_learning_rate"]),
                },
            ])
        warmup = args.critic_warmup_iterations
        if warmup:
            # A changed reward contract invalidates the old value baseline but
            # not necessarily the actor.  Fit the critic first while keeping
            # the known-good gait and deterministic action mean bit-for-bit
            # fixed.  This avoids the large, misleading advantages that made
            # earlier warm starts destroy the policy within tens of updates.
            for parameter in runner.alg.policy.actor.parameters():
                parameter.requires_grad_(False)
            runner.alg.policy.log_std.requires_grad_(False)
            runner.learn(warmup, init_at_random_ep_len=True)
            for parameter in runner.alg.policy.actor.parameters():
                parameter.requires_grad_(True)
            runner.alg.policy.log_std.requires_grad_(True)
            runner.current_learning_iteration += 1
            runner.learn(args.iterations - warmup, init_at_random_ep_len=False)
        else:
            runner.learn(args.iterations, init_at_random_ep_len=True)
    finally:
        raw_env.close()
    print(f"TRAINING FINISHED; checkpoints: {checkpoint_dir}")


def _evaluate(args):
    import torch
    import gymnasium as gym
    from .checkpoints import certify_checkpoint, file_hash, write_json
    from .envs import make_env

    if args.stage not in {"stage1", "stage2"}:
        raise ValueError("evaluate currently supports stage1 or stage2 policy capability")
    _required(args, "checkpoint")
    episodes = args.episodes or (400 if args.stage == "stage1" else 200)
    if args.certify and episodes < (400 if args.stage == "stage1" else 200):
        raise ValueError("certification requires 400 Stage-1 or 200 Stage-2 episodes")
    count = min(args.num_envs or 64, episodes)
    kind = "locomotion" if args.stage == "stage1" else "navigation"
    actor = _load_dependency(args.checkpoint, kind, args.device, validated=False)
    locomotion = None
    if args.stage == "stage2":
        _required(args, "locomotion-checkpoint")
        locomotion = _load_dependency(
            args.locomotion_checkpoint, "locomotion", args.device,
            validated=not args.allow_uncertified_locomotion,
        )
    raw_env = make_env(stage=kind, num_envs=count, device=args.device,
                       enable_cameras=False, locomotion_policy=locomotion,
                       seed=args.seed, terrain=args.terrain, purpose="integration",
                       episode_seconds=args.episode_seconds,
                       render_mode="rgb_array" if args.record_video else None,
                       show_direction_arrows=args.show_direction_arrows,
                       course_difficulty_min=args.course_difficulty_min,
                       course_difficulty_max=args.course_difficulty_max,
                       speed_min=args.speed_min, speed_max=args.speed_max,
                       reward_contract=_resolve_contract(args))
    env = raw_env
    video_steps = 0
    requested_video = None
    generated_video = None
    if args.record_video:
        requested_video = Path(args.record_video).expanduser().resolve()
        requested_video.parent.mkdir(parents=True, exist_ok=True)
        video_steps = _video_frame_count(args.video_seconds, raw_env.step_dt)
        env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(requested_video.parent),
            step_trigger=lambda step: step == 0,
            video_length=video_steps,
            name_prefix=requested_video.stem,
            disable_logger=True,
        )
    if args.follow_env is not None and not 0 <= args.follow_env < count:
        env.close()
        raise ValueError(f"--follow-env must be between 0 and {count - 1}")
    def update_follow_camera():
        if args.follow_env is None:
            return
        root = raw_env.hier.robot.data.root_link_pos_w[args.follow_env].detach().cpu()
        eye = (root + root.new_tensor([-3.8, -2.6, 2.2])).tolist()
        target = (root + root.new_tensor([1.4, 0.0, 0.25])).tolist()
        raw_env.sim.set_camera_view(eye=eye, target=target)
    records = []
    debug_step = 0
    hidden = torch.zeros(1, count, 128, device=args.device)
    try:
        obs, _ = env.reset()
        update_follow_camera()
        while len(records) < episodes or debug_step < video_steps:
            with torch.no_grad():
                action, hidden = actor(obs["policy"], hidden)
            obs, _, terminated, truncated, extras = env.step(action)
            update_follow_camera()
            debug_step += 1
            if args.debug_motion and debug_step % 25 == 0:
                root = raw_env.hier.robot.data.root_link_pos_w[0]
                vel = raw_env.hier.robot.data.root_lin_vel_b[0]
                terms = (raw_env.hier._build_locomotion_reward_terms()
                         if raw_env.cfg.hier_stage == "locomotion"
                         else raw_env.hier._nav_v2_terms())
                # Per-environment summary so a failure can be located on the course.
                origins = raw_env.scene.env_origins
                pos = raw_env.hier.robot.data.root_link_pos_w
                flags = torch.stack(
                    (raw_env.hier._fallen, raw_env.hier._collision,
                     raw_env.hier._out_of_bounds, raw_env.hier._success), dim=-1
                )
                row = " ".join(
                    "%d:(%.1f,%.1f,z%.2f)%s" % (
                        i,
                        (pos[i, 0] - origins[i, 0]).item(),
                        (pos[i, 1] - origins[i, 1]).item(),
                        pos[i, 2].item(),
                        ("F" if flags[i, 0] else "") + ("C" if flags[i, 1] else "")
                        + ("O" if flags[i, 2] else "") + ("S" if flags[i, 3] else ""),
                    )
                    for i in range(min(6, raw_env.num_envs))
                )
                print("ENVS", debug_step, row, flush=True)
                goal = raw_env.hier._goal_world[0, :2]
                goal_distance = float((goal - root[:2]).norm().item())
                print("MOTION", debug_step,
                      "x=%.3f y=%.3f vx=%.3f z=%.3f goal_dist=%.3f "
                      "fallen=%s collision=%s out=%s success=%s" %
                      (root[0].item(), root[1].item(), vel[0].item(), root[2].item(),
                       goal_distance, bool(raw_env.hier._fallen[0]),
                       bool(raw_env.hier._collision[0]),
                       bool(raw_env.hier._out_of_bounds[0]),
                       bool(raw_env.hier._success[0])))
                terrain_values, terrain_valid, _ = raw_env.hier._height_geometry()
                print("SCAN", "valid=%d/%d min=%.3f max=%.3f ahead_max=%.3f" %
                      (int(terrain_valid[0].sum().item()), terrain_valid.shape[1],
                       terrain_values[0].min().item(), terrain_values[0].max().item(),
                       terrain_values[0, : max(1, terrain_values.shape[1] // 2)].max().item()))
                foot_force = raw_env.hier.contact_sensor.data.net_forces_w[0, raw_env.hier._foot_ids]
                contacts = torch.linalg.vector_norm(foot_force, dim=-1) > 1.0
                torque = raw_env.hier.robot.data.applied_torque[0, raw_env.hier._joint_ids]
                joint_error = (
                    raw_env.hier.robot.data.joint_pos[0, raw_env.hier._joint_ids]
                    - raw_env.hier.robot.data.default_joint_pos[0, raw_env.hier._joint_ids]
                )
                roll_pitch_rate = torch.linalg.vector_norm(
                    raw_env.hier.robot.data.root_ang_vel_b[0, :2]
                )
                print("STABILITY", "contacts=%s max_air=%.3f torque_rms=%.3f joint_err_rms=%.3f vz=%.3f rp_rate=%.3f" %
                      (contacts.to(torch.int).tolist(),
                       raw_env.hier._foot_air_duration[0].max().item(),
                       torque.square().mean().sqrt().item(),
                       joint_error.square().mean().sqrt().item(),
                       vel[2].item(), roll_pitch_rate.item()))
                print("REWARD", {name: round(value[0].item(), 6) for name, value in terms.items()})
            done = terminated | truncated
            hidden[:, done] = 0
            # Isaac Lab keeps extras entries until a later reset overwrites them.
            # Consume each episode record once and reject stale/misaligned data.
            episode = extras.pop("episode_metrics", None)
            if episode is not None and done.any():
                env_ids = episode["env_ids"].tolist()
                done_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
                if sorted(env_ids) != sorted(done_ids):
                    raise RuntimeError(
                        f"episode metrics env_ids {env_ids} do not match completed env_ids {done_ids}"
                    )
                for row, _env_id in enumerate(env_ids):
                    if len(records) >= episodes:
                        continue
                    records.append({key: (float(value[row]) if value[row].dtype != torch.bool else bool(value[row]))
                                    for key, value in episode.items() if key != "env_ids"})
    finally:
        env.close()
        if requested_video is not None:
            candidates = (
                requested_video.with_name(requested_video.stem + "-step-0.mp4"),
                requested_video.with_name(requested_video.stem + "-episode-0.mp4"),
            )
            generated_video = next((path for path in candidates if path.is_file()), requested_video)
            if generated_video.is_file() and generated_video != requested_video:
                requested_video.unlink(missing_ok=True)
                generated_video.replace(requested_video)
                generated_video = requested_video
    completion = sum(item["success"] and not item["fallen"] and not item["out_of_bounds"]
                     and not item["collision"] for item in records) / episodes
    protocol = "stage1-v2" if args.stage == "stage1" else "stage2-v1"
    report = {"protocol": protocol, "stage": args.stage,
              "reward_contract": _resolve_contract(args),
              "checkpoint_sha256": file_hash(args.checkpoint), "episodes": episodes,
              "terrain": args.terrain, "seed": args.seed, "completion_rate": completion}
    if args.stage == "stage1":
        velocity_samples = sum(item["velocity_sample_count"] for item in records)
        if velocity_samples <= 0:
            raise RuntimeError("evaluation produced no velocity samples")
        report.update({
            "success_rate": sum(item["success"] for item in records) / episodes,
            "collision_rate": sum(item["collision"] for item in records) / episodes,
            "fallen_rate": sum(item["fallen"] for item in records) / episodes,
            "out_of_bounds_rate": sum(item["out_of_bounds"] for item in records) / episodes,
            "low_height_failure_rate": sum(
                item.get("low_height_failure", False) for item in records
            ) / episodes,
            "tilt_failure_rate": sum(item.get("tilt_failure", False) for item in records) / episodes,
            "absolute_height_failure_rate": sum(
                item.get("absolute_height_failure", False) for item in records
            ) / episodes,
            "timeout_rate": sum(
                not (
                    item["success"]
                    or item["collision"]
                    or item["fallen"]
                    or item["out_of_bounds"]
                )
                for item in records
            ) / episodes,
            "velocity_xy_rmse": (
                sum(item["xy_velocity_squared_error"] for item in records) / velocity_samples
            ) ** 0.5,
            "yaw_rmse": (
                sum(item["yaw_squared_error"] for item in records) / velocity_samples
            ) ** 0.5,
        })
        report["passed"] = (args.terrain == "course" and completion >= .95 and
                            report["velocity_xy_rmse"] <= .15 and report["yaw_rmse"] <= .25)
    else:
        # Report every termination cause: a Stage-2 report that only exposed
        # success/collision silently hid falls and course exits.
        report.update({
            "success_rate": sum(item["success"] for item in records) / episodes,
            "collision_rate": sum(item["collision"] for item in records) / episodes,
            "fallen_rate": sum(item["fallen"] for item in records) / episodes,
            "out_of_bounds_rate": sum(item["out_of_bounds"] for item in records) / episodes,
            "stale_goal_rate": sum(item["stale_goal"] for item in records) / episodes,
            "timeout_rate": sum(
                not (
                    item["success"]
                    or item["collision"]
                    or item["fallen"]
                    or item["out_of_bounds"]
                )
                for item in records
            ) / episodes,
        })
        report["passed"] = (args.terrain == "course" and report["success_rate"] >= .90 and
                            report["collision_rate"] <= .05)
    output = WORKSPACE / "reports" / f"evaluate_{args.stage}_{_run_id(args)}.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"REPORT: {output}")
    if generated_video is not None:
        if not generated_video.is_file() or generated_video.stat().st_size == 0:
            raise RuntimeError(f"Isaac Sim video was not written: {generated_video}")
        print(f"VIDEO: {generated_video}")
    if args.certify:
        certified = certify_checkpoint(args.checkpoint, output)
        print(f"CERTIFIED: {certified}")


def main():
    parser = _parser()
    args = parser.parse_args()
    if os.environ.get("WORLD_SIZE", "1") != "1":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        args.device = f"cuda:{local_rank}"
        # Give each worker a different simulator RNG stream.  RSL-RL will
        # synchronize/average gradients, but duplicated environment seeds
        # would otherwise waste half of the two-GPU rollout batch.
        args.seed += local_rank
    if args.operation is None:
        parser.error("operation is required")
    if args.record_video:
        args.enable_cameras = True
    if args.enable_cameras and args.device != "cuda:0":
        parser.error("this Isaac Sim renderer supports camera workloads only on --device cuda:0")
    if args.enable_cameras and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("camera rendering requires the container X display (expected DISPLAY=:1)")
    if args.operation == "export":
        _required(args, "checkpoint", "output-dir")
        from .export import export_checkpoint
        print(export_checkpoint(args.checkpoint, args.output_dir))
        return
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        if args.operation == "check":
            _check(args)
        elif args.operation == "train":
            _train(args)
        elif args.operation == "evaluate":
            _evaluate(args)
        elif args.operation == "distill":
            from .training.distillation import run_distillation
            run_distillation(args, WORKSPACE)
        else:
            raise AssertionError(args.operation)
    except BaseException:
        # SimulationApp.close() invokes a successful SystemExit in this
        # installation and would overwrite the actual failure status.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
