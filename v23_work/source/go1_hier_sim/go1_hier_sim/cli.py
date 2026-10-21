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
# Paper success criterion: an episode counts as a survival once the policy stays
# on the terrain for at least 900 control steps (900 x 0.02 s = 18 s). The
# difficulty gate uses completion rate; survival is the reported companion.
SURVIVAL_STEPS = 900


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
    parser.add_argument("--terrain", choices=("course", "flat", "slopes", "rough", "stairs", "mixed",
                                            "gap_course", "course_gaps", "lp_acrl",
                                            "stairs_up_long", "stairs_down_long", "stairs_mixed",
                                            "slope_up", "slope_down", "slope_mixed",
                                            "gaps_only", "combined_flat"),
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
    parser.add_argument("--eval-sample-actions", action="store_true",
                        help="evaluate the stochastic policy (mean + learned std) instead of the mean")
    parser.add_argument("--jump-checkpoint",
                        help="frozen jump skill; with --checkpoint it gates walk vs leap on gap courses")
    parser.add_argument("--leap-base-checkpoint",
                        help="frozen walking policy that drives outside a leap while training the leap actor")
    parser.add_argument("--follow-env", type=int, default=None,
                        help="follow this environment in the Isaac Sim viewport during evaluation")
    parser.add_argument("--debug-motion", action="store_true",
                        help="print evaluation motion and reward diagnostics")
    parser.add_argument("--episode-trace",
                        help="opt-in JSONL of matched completed Stage-1 episodes; diagnostic only")
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


def _evaluation_step_limit(episodes: int, video_steps: int) -> int:
    """Preserve the non-video gate cap while allowing requested video frames."""
    return max(episodes * 40, 2000, video_steps + 1)


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
    variant = os.environ.get("GO1_POLICY_VARIANT", "v1")
    if variant not in {"v1", "map_attn_v3"}:
        raise ValueError(f"unknown GO1 policy variant: {variant}")
    if variant == "map_attn_v3" and (stage != "stage1" or getattr(args, "reward_contract", None) == "loco_v2"):
        raise ValueError("map_attn_v3 supports the v7b stage1 locomotion contract only")
    policy_class = ("HierActorCriticV3" if variant == "map_attn_v3" else
                    "HierActorCriticV2" if getattr(args, "reward_contract", None) in {"loco_v2", "nav_v2"} else
                    "HierActorCritic")
    # Seeding the leap actor from a leap primitive is a new-skill run, not a
    # protected fine-tune of a validated policy. The warm-start schedule caps
    # the actor learning rate at a few parts per million, which is far too small
    # to learn a leap, so leap training keeps the standard PPO schedule.
    warm_start = args.initialize_from is not None and args.leap_base_checkpoint is None
    # Legacy mask zeros the actor surrogate on walker steps, but upstream PPO
    # would still apply entropy and adaptive KL on those off-policy samples.
    # In v10, neither may update the fresh leap actor outside executed steps.
    masked_leap = (os.environ.get("GO1_V10_OBSERVABLE_LEAP") == "1"
                   and args.leap_base_checkpoint is not None)
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
            "class_name": policy_class,
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
            "entropy_coef": (0.0 if masked_leap else float(
                values["warm_start_entropy_coef"] if warm_start else values["entropy_coef"]
            )),
            "num_learning_epochs": int(values["num_learning_epochs"]),
            "num_mini_batches": int(values["num_mini_batches"]),
            "learning_rate": float(
                values["warm_start_learning_rate"] if warm_start else values["learning_rate"]
            ), "schedule": "fixed" if warm_start or masked_leap else "adaptive",
            "gamma": float(stage_values["gamma"]), "lam": float(values["lam"]),
            "desired_kl": (None if masked_leap else float(
                values["warm_start_desired_kl"] if warm_start else values["desired_kl"]
            )), "max_grad_norm": float(values["max_grad_norm"]),
            "normalize_advantage_per_mini_batch": False, "rnd_cfg": None, "symmetry_cfg": None,
        },
    }


def _v10_settings_hashes(cfg, args, ppo_values, contract):
    """Hash the effective PPO, reward and physical scan contracts for resuming."""
    from .checkpoints import file_hash
    from .envs.training_contract_v10 import settings_hash

    ppo = _ppo_config(args.stage, args, ppo_values)
    ppo_settings = {key: ppo[key] for key in
                    ("policy", "algorithm", "num_steps_per_env", "clip_actions", "obs_groups")}
    fine = cfg.scene.gap_edge_scanner.pattern_cfg
    coarse = cfg.scene.height_scanner.pattern_cfg
    physics_settings = {
        "sim_dt": float(cfg.sim.dt), "gravity": list(cfg.sim.gravity),
        "decimation": int(cfg.decimation),
        "physx": str(cfg.sim.physx),
        "coarse_scan": [float(coarse.resolution), *map(float, coarse.size)],
        "fine_scan": [float(fine.resolution), *map(float, fine.size)],
        "fine_offset": list(map(float, cfg.scene.gap_edge_scanner.offset.pos)),
        "v8_reward_terrain_mode": os.environ.get("GO1_V8_TERRAIN_CONTROL") == "1",
        "v9_contact_mode": os.environ.get("GO1_V9_GAP_CONTACT") == "1",
        "source_config_sha256": file_hash(Path(__file__).parent / "envs/config.py"),
        "source_geometry_sha256": file_hash(Path(__file__).parent / "envs/geometry.py"),
        "source_runtime_sha256": file_hash(Path(__file__).parent / "envs/runtime.py"),
        "source_observable_gate_sha256": file_hash(Path(__file__).parent / "envs/observable_gap_v10.py"),
        "source_handoff_sha256": file_hash(Path(__file__).parent / "envs/handoff_v10.py"),
    }
    return {
        "ppo_settings_sha256": settings_hash(ppo_settings),
        "reward_settings_sha256": settings_hash(_contract_table(cfg, contract)),
        "physics_settings_sha256": settings_hash(physics_settings),
    }


def _load_warm_start_weights(policy, state, variant):
    """Require all shared weights; permit a wholly new adapter only for v7b."""
    import torch

    state = dict(state)
    state.pop("log_std", None)
    incompatible = torch.nn.Module.load_state_dict(policy, state, strict=False)
    allowed = {"log_std"}
    if variant == "map_attn_v3":
        adapter = {k for k in policy.state_dict() if k.startswith("actor.map_adapter.")}
        if len(adapter) < 2:
            raise ValueError("v9 policy has no attention adapter")
        present = adapter & state.keys()
        if not present:
            # v7b -> v8: residual starts at its documented identity init.
            allowed.update(adapter)
        elif present != adapter:
            raise ValueError("warm-start checkpoint has a partial attention adapter")
        # v8 -> v9: every trained adapter weight must be preserved.
    if set(incompatible.missing_keys) != allowed or incompatible.unexpected_keys:
        raise ValueError(
            "warm-start checkpoint policy structure is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _warm_start_optimizer(policy, ppo_values, variant):
    """Protect the old gait while allowing the new residual to learn."""
    import torch

    actor_lr = float(ppo_values["warm_start_actor_learning_rate"])
    critic_lr = float(ppo_values["warm_start_learning_rate"])
    if variant == "map_attn_v3":
        legacy = [p for name, p in policy.actor.named_parameters()
                  if not name.startswith("map_adapter.")]
        legacy.append(policy.log_std)
        adapter = list(policy.actor.map_adapter.parameters())
        return torch.optim.Adam([
            {"params": legacy, "lr": actor_lr},
            {"params": adapter, "lr": 3e-5},
            {"params": policy.critic.parameters(), "lr": critic_lr},
        ])
    return torch.optim.Adam([
        {"params": list(policy.actor.parameters()) + [policy.log_std], "lr": actor_lr},
        {"params": policy.critic.parameters(), "lr": critic_lr},
    ])


def _configure_resume_optimizer(runner, saved_optimizer, ppo_values, variant):
    """Restore the original Adam group layout before loading its moment state.

    V8/V9 map-attention warm starts use three parameter groups, whereas a new
    RSL runner starts with one. A different layout silently losing its state
    or loading moments onto the wrong parameters would invalidate a restart.
    """
    if not isinstance(saved_optimizer, dict) or not isinstance(saved_optimizer.get("param_groups"), list):
        raise ValueError("resume checkpoint has no optimizer parameter groups")
    saved_groups = saved_optimizer["param_groups"]
    if variant == "map_attn_v3" and len(saved_groups) == 3:
        runner.alg.optimizer = _warm_start_optimizer(runner.alg.policy, ppo_values, variant)
    elif len(saved_groups) != 1:
        raise ValueError("unsupported resume optimizer group layout")
    expected = [len(group["params"]) for group in runner.alg.optimizer.param_groups]
    actual = [len(group.get("params", [])) for group in saved_groups]
    if expected != actual:
        raise ValueError(f"resume optimizer group sizes differ: checkpoint={actual} runner={expected}")


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
    if os.environ.get("GO1_POLICY_VARIANT") == "map_attn_v3" and os.environ.get("GO1_V8_TERRAIN_CONTROL") != "1":
        raise ValueError("map_attn_v3 training requires GO1_V8_TERRAIN_CONTROL=1")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from .checkpoints import file_hash, load_checkpoint
    from .envs.training_contract_v10 import leap_metadata, validate_jump_checkpoint, validate_new_leap_run
    from .envs import make_env
    from .reward_config import require_matching_reward_config
    from .training.on_policy import HierOnPolicyRunner

    v10 = os.environ.get("GO1_V10_OBSERVABLE_LEAP") == "1"
    base_sha = None
    if v10:
        if (os.environ.get("GO1_POLICY_VARIANT") != "map_attn_v3"
                or os.environ.get("GO1_V8_TERRAIN_CONTROL") != "1"):
            raise ValueError("v10 leap requires map_attn_v3 and v8 reward-only terrain mode")
        if not args.resume:
            validate_new_leap_run(args.stage, args.leap_base_checkpoint,
                                  initialize_from=args.initialize_from,
                                  resume=args.resume, jump_checkpoint=args.jump_checkpoint)
        elif args.stage != "stage1" or not args.leap_base_checkpoint or args.jump_checkpoint:
            raise ValueError("v10 resume requires the original frozen walker")
        base_sha = file_hash(args.leap_base_checkpoint)
        if base_sha != "136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15":
            raise ValueError("v10 walker does not match the pinned v8 model_899.pt SHA-256")
        if args.resume:
            resume_header = load_checkpoint(args.resume, expected_stage="stage1")
            validate_jump_checkpoint(resume_header.get("metadata", {}), frozen_walker_sha=base_sha)
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
    jump_policy = None
    if args.jump_checkpoint:
        if args.stage != "stage1":
            raise ValueError("the jump gate only applies to Stage-1 locomotion")
        jump_policy = _load_dependency(args.jump_checkpoint, "locomotion", args.device,
                                       validated=False)
    base_policy = None
    if args.leap_base_checkpoint:
        if args.stage != "stage1":
            raise ValueError("the leap gate only applies to Stage-1 locomotion")
        if args.jump_checkpoint:
            raise ValueError("--leap-base-checkpoint and --jump-checkpoint are mutually exclusive")
        base_policy = _load_dependency(args.leap_base_checkpoint, "locomotion", args.device,
                                       validated=False)
    contract = args.reward_contract or ("nav_v2" if args.stage == "stage2" else "stage1")
    raw_env = make_env(stage="locomotion" if args.stage == "stage1" else "navigation",
                       num_envs=num_envs, device=args.device, enable_cameras=False,
                       locomotion_policy=locomotion, jump_policy=jump_policy,
                       base_policy=base_policy,
                       seed=args.seed, terrain=args.terrain,
                       purpose="train", episode_seconds=args.episode_seconds,
                       course_difficulty_min=args.course_difficulty_min,
                       course_difficulty_max=args.course_difficulty_max,
                       speed_min=args.speed_min, speed_max=args.speed_max,
                       reward_contract=contract, robustness=args.robustness == "on")
    run_id = _run_id(args)
    log_dir = WORKSPACE / "runs" / args.stage / run_id
    checkpoint_dir = WORKSPACE / "checkpoints" / args.stage / run_id
    v10_settings = (_v10_settings_hashes(raw_env.cfg, args, ppo_values, contract) if v10 else {})
    if v10 and args.resume:
        validate_jump_checkpoint(resume_header.get("metadata", {}),
                                 frozen_walker_sha=base_sha, expected_settings=v10_settings)

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
                         "reward_variant": (
                             "+".join([part for part in (
                                 "teacher_distill_v29" if os.environ.get("GO1_V29_TEACHER_DISTILL") == "1" else None,
                                 "tire_brake_v23" if os.environ.get("GO1_V23_TIRE_BRAKE") == "1" else None,
                                 "prelaunch_v19+combined" if os.environ.get("GO1_V19_COMBINED") == "1" and os.environ.get("GO1_V19_PRELAUNCH_REWARD") == "1" else ("prelaunch_v19" if os.environ.get("GO1_V19_PRELAUNCH_REWARD") == "1" else None),
                                 "prelaunch_v18" if os.environ.get("GO1_V18_PRELAUNCH_REWARD") == "1" else None,
                                 "pure_short_v14" if os.environ.get("GO1_V14_SHORT_REWARD") == "1" else None,
                             ) if part] or ["legacy"])
                         ),
                         "policy_variant": os.environ.get("GO1_POLICY_VARIANT", "v1"),
                         "course_difficulty": [args.course_difficulty_min, args.course_difficulty_max],
                         "speed_range": [args.speed_min, args.speed_max],
                         "upstream": {"locomotion": args.locomotion_checkpoint,
                                      "leap_base": args.leap_base_checkpoint,
                                      "initialized_from": args.initialize_from},
                         **(leap_metadata(base_sha, **v10_settings) if v10 else {})})

    try:
        if (base_policy is not None or jump_policy is not None
                or os.environ.get("GO1_V29_TEACHER_DISTILL") == "1"):
            class LeapMaskWrapper(RslRlVecEnvWrapper):
                """Forward the trench gate's policy mask into the RSL-RL extras."""

                def step(self, actions):
                    obs, rewards, dones, extras = super().step(actions)
                    extras["policy_mask"] = raw_env.hier.policy_mask
                    if getattr(raw_env.hier, "_v29_pool", None) is not None:
                        extras["teacher_mean"] = raw_env.hier.v29_teacher_mean
                        extras["teacher_valid"] = raw_env.hier.v29_teacher_valid
                    return obs, rewards, dones, extras

            env = LeapMaskWrapper(raw_env, clip_actions=3.0 if args.stage == "stage1" else 1.0)
        else:
            env = RslRlVecEnvWrapper(raw_env, clip_actions=3.0 if args.stage == "stage1" else 1.0)
        runner = HierOnPolicyRunner(env, _ppo_config(args.stage, args, ppo_values), str(log_dir),
                                    device=args.device, checkpoint_callback=save_runner)
        if args.resume:
            resume_data = load_checkpoint(args.resume, expected_stage=args.stage)
            require_matching_reward_config(
                resume_data.get("metadata", {}), _contract_table(raw_env.cfg, contract)
            )
            _configure_resume_optimizer(runner, resume_data.get("optimizer"), ppo_values,
                                        os.environ.get("GO1_POLICY_VARIANT", "v1"))
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
            # Retain every v7b actor and critic tensor, allow only the v8
            # adapter to start fresh, and reinitialize exploration separately.
            variant = os.environ.get("GO1_POLICY_VARIANT", "v1")
            _load_warm_start_weights(runner.alg.policy, state, variant)
            with torch.no_grad():
                runner.alg.policy.log_std.fill_(
                    math.log(float(ppo_values["warm_start_noise_std"]))
                )
            # Actor means from the best checkpoint are extremely sensitive:
            # an ordinary shared Adam rate moved the actor by only ~1e-4 in
            # relative L2 yet cut course completion sharply.  Give actor and
            # critic independent fixed rates so the new value function can
            # adapt 100x faster without erasing the gait.
            runner.alg.optimizer = _warm_start_optimizer(runner.alg.policy, ppo_values, variant)
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
        stage_records = raw_env.hier.pop_stage_records() if hasattr(raw_env, "hier") else []
        raw_env.close()
        if stage_records:
            # LP-ACRL stage report: per-environment episode statistics, drained
            # before the environment is destroyed so the curriculum driver can
            # attribute each finished episode to the task instance it ran.
            from .checkpoints import write_json

            per_env: dict[int, dict] = {}
            for record in stage_records:
                slot = per_env.setdefault(
                    record["env"],
                    {"returns": [], "steps": [], "fallen": 0, "success": 0,
                     "collision": 0, "out_of_bounds": 0, "fell_into_gap": 0,
                     "survival": 0, "command_speed": [], "achieved_speed": [],
                     "course_x": [], "course_y": []},
                )
                slot["returns"].append(record["return"])
                slot["steps"].append(record["steps"])
                for flag in ("fallen", "success", "collision", "out_of_bounds", "fell_into_gap"):
                    slot[flag] += int(record.get(flag, False))
                if record["steps"] >= SURVIVAL_STEPS:
                    slot["survival"] += 1
                slot["command_speed"].append(float(record.get("command_speed", 0.0)))
                slot["achieved_speed"].append(float(record.get("achieved_speed", 0.0)))
                slot["course_x"].append(float(record.get("course_x", 0.0)))
                slot["course_y"].append(float(record.get("course_y", 0.0)))
            all_episodes = sum(len(slot["returns"]) for slot in per_env.values())
            all_success = sum(slot["success"] for slot in per_env.values())
            all_survival = sum(slot["survival"] for slot in per_env.values())
            command_speeds = [value for slot in per_env.values() for value in slot["command_speed"]]
            achieved_speeds = [value for slot in per_env.values() for value in slot["achieved_speed"]]
            report = {
                "run": _run_id(args),
                "iterations": args.iterations,
                "task_plan": os.environ.get("GO1_TASK_PLAN", ""),
                "episodes": all_episodes,
                "completion_rate": all_success / all_episodes if all_episodes else 0.0,
                "survival_rate": all_survival / all_episodes if all_episodes else 0.0,
                "mean_command_speed": (
                    sum(command_speeds) / len(command_speeds) if command_speeds else 0.0
                ),
                "mean_achieved_speed": (
                    sum(achieved_speeds) / len(achieved_speeds) if achieved_speeds else 0.0
                ),
                "environments": {
                    str(env_id): {
                        "episodes": len(slot["returns"]),
                        "mean_return": sum(slot["returns"]) / len(slot["returns"]),
                        "mean_steps": sum(slot["steps"]) / len(slot["steps"]),
                        "fallen": slot["fallen"],
                        "success": slot["success"],
                        "collision": slot["collision"],
                        "out_of_bounds": slot["out_of_bounds"],
                        "fell_into_gap": slot["fell_into_gap"],
                        "survival": slot["survival"],
                        "mean_failure_x": sum(slot["course_x"]) / len(slot["course_x"]),
                        "mean_failure_y": sum(slot["course_y"]) / len(slot["course_y"]),
                    }
                    for env_id, slot in sorted(per_env.items())
                },
            }
            output = WORKSPACE / "reports" / f"stage_{_run_id(args)}.json"
            write_json(output, report)
            print(f"STAGE REPORT: {output}")
    print(f"TRAINING FINISHED; checkpoints: {checkpoint_dir}")


def _evaluate(args):
    import torch
    import gymnasium as gym
    from .checkpoints import certify_checkpoint, file_hash, write_json
    from .envs import make_env
    from .envs.episode_trace_v12 import merge_episode_trace

    if args.stage not in {"stage1", "stage2"}:
        raise ValueError("evaluate currently supports stage1 or stage2 policy capability")
    _required(args, "checkpoint")
    if os.environ.get("GO1_V10_OBSERVABLE_LEAP") == "1":
        from .envs.training_contract_v10 import forbid_v10_pooled_certification
        forbid_v10_pooled_certification(args.certify)
    episodes = args.episodes or (400 if args.stage == "stage1" else 200)
    if args.episode_trace and args.stage != "stage1":
        raise ValueError("--episode-trace only supports Stage-1 evaluation")
    trace_path = Path(args.episode_trace).expanduser().resolve() if args.episode_trace else None
    if trace_path is not None and trace_path.exists():
        raise FileExistsError(f"episode trace already exists: {trace_path}")
    if trace_path is not None:
        os.environ["GO1_V12_EPISODE_TRACE"] = "1"
    else:
        os.environ.pop("GO1_V12_EPISODE_TRACE", None)
    if args.certify and episodes < (400 if args.stage == "stage1" else 200):
        raise ValueError("certification requires 400 Stage-1 or 200 Stage-2 episodes")
    count = min(args.num_envs or 64, episodes)
    kind = "locomotion" if args.stage == "stage1" else "navigation"
    from .checkpoints import load_checkpoint
    from .envs.training_contract_v10 import validate_checkpoint_role
    v10_observable = os.environ.get("GO1_V10_OBSERVABLE_LEAP") == "1"
    if kind == "locomotion":
        walk_data = load_checkpoint(args.checkpoint, expected_stage="stage1",
                                    allow_locomotion_heading_migration=True)
        validate_checkpoint_role(walk_data.get("metadata", {}), role="walk",
                                 observable_enabled=v10_observable)
    actor = _load_dependency(args.checkpoint, kind, args.device, validated=False)
    sample_std = None
    if args.eval_sample_actions:
        data = load_checkpoint(args.checkpoint, "stage1" if kind == "locomotion" else "stage2",
                               require_validated=False,
                               allow_locomotion_heading_migration=(kind == "locomotion"))
        log_std = data["models"]["policy"].get("log_std")
        if log_std is None:
            raise ValueError("checkpoint has no learned log_std; cannot sample actions")
        sample_std = torch.exp(log_std.to(args.device))
        print("SAMPLED-EVAL std=" + ",".join("%.3f" % v for v in sample_std.tolist()))
    jump_actor = None
    if args.jump_checkpoint:
        if args.stage != "stage1":
            env_warning = "the jump gate only applies to Stage-1 locomotion"
            raise ValueError(env_warning)
        jump_data = load_checkpoint(args.jump_checkpoint, expected_stage="stage1",
                                    allow_locomotion_heading_migration=True)
        validate_checkpoint_role(jump_data.get("metadata", {}), role="jump",
                                 observable_enabled=v10_observable)
        if v10_observable:
            from .checkpoints import file_hash
            from .envs.training_contract_v10 import validate_jump_checkpoint
            validate_jump_checkpoint(jump_data.get("metadata", {}),
                                     frozen_walker_sha=file_hash(args.checkpoint))
            print("WARNING: v10 jump checkpoint is experimental; deployment is NOT certified; "
                  "a complete 512-env 11-family gate remains mandatory", flush=True)
        jump_actor = _load_dependency(args.jump_checkpoint, "locomotion", args.device,
                                      validated=False)
    locomotion = None
    if args.stage == "stage2":
        _required(args, "locomotion-checkpoint")
        locomotion = _load_dependency(
            args.locomotion_checkpoint, "locomotion", args.device,
            validated=not args.allow_uncertified_locomotion,
        )
    raw_env = make_env(stage=kind, num_envs=count, device=args.device,
                       enable_cameras=False, locomotion_policy=locomotion,
                       jump_policy=jump_actor,
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
    trace_rows: list[dict] = []
    trace_counts: dict[int, int] = {}
    debug_step = 0
    hidden = torch.zeros(1, count, 128, device=args.device)
    # The leap hand-over is owned by the environment, so this loop only feeds
    # the walking policy; the runtime swaps in the frozen jump action while a
    # trench leap is latched.
    try:
        obs, _ = env.reset()
        if trace_path is not None:
            initial = raw_env.hier.pop_stage_records()
            if any(int(row["steps"]) != 0 for row in initial):
                raise RuntimeError("episode trace saw a completed episode during initial reset")
        update_follow_camera()
        # One episode per environment: with an equal instance share per
        # family this makes every family's completion rate comparable.
        seen_envs: set[int] = set()
        while ((len(records) < episodes or len(seen_envs) < count)
               or debug_step < video_steps):
            with torch.no_grad():
                walk_action, hidden = actor(obs["policy"], hidden)
                if sample_std is not None:
                    walk_action = walk_action + sample_std * torch.randn_like(walk_action)
                action = walk_action
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
                print("GATE", "leap_envs=%d" % int(raw_env.hier._leap_mode.sum().item()))
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
                print("REWARD", {name: round(float(value[0].reshape(-1)[0].item()), 6) for name, value in terms.items()})
            if debug_step > _evaluation_step_limit(episodes, video_steps):
                break
            done = terminated | truncated
            hidden[:, done] = 0
            stage_records = raw_env.hier.pop_stage_records() if trace_path is not None else []
            if trace_path is not None and bool(done.any()) != bool(stage_records):
                raise RuntimeError("episode trace terminal records do not match done mask")
            # Isaac Lab keeps extras entries until a later reset overwrites them.
            # Consume each episode record once and reject stale/misaligned data.
            episode = extras.pop("episode_metrics", None)
            if trace_path is not None and bool(done.any()) and episode is None:
                raise RuntimeError("episode trace has a done mask but no episode metrics")
            if episode is not None and done.any():
                env_ids = episode["env_ids"].tolist()
                done_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
                if sorted(env_ids) != sorted(done_ids):
                    raise RuntimeError(
                        f"episode metrics env_ids {env_ids} do not match completed env_ids {done_ids}"
                    )
                episode_rows = [
                    {key: (float(value[row]) if value[row].dtype != torch.bool else bool(value[row]))
                     for key, value in episode.items() if key != "env_ids"} | {"env_id": int(env_id)}
                    for row, env_id in enumerate(env_ids)
                ]
                traced = (merge_episode_trace(done_ids, episode_rows, stage_records)
                          if trace_path is not None else [])
                trace_by_env = {entry["env_id"]: entry for entry in traced}
                for item in episode_rows:
                    env_id = int(item["env_id"])
                    # Coverage first: a plain global quota lets fast
                    # failures fill it while slow families never report,
                    # which makes the per-family gate unpassable.
                    if len(records) >= episodes and env_id in seen_envs:
                        continue
                    seen_envs.add(env_id)
                    records.append(item)
                    if trace_path is not None:
                        entry = trace_by_env[env_id]
                        entry["episode_index"] = trace_counts.get(env_id, 0)
                        trace_counts[env_id] = entry["episode_index"] + 1
                        trace_rows.append(entry)
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
    if trace_path is not None:
        if len(trace_rows) != len(records):
            raise RuntimeError("episode trace and evaluation record counts differ")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_name(trace_path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        temporary.replace(trace_path)
        print(f"EPISODE_TRACE: {trace_path} sha256={file_hash(trace_path)} rows={len(trace_rows)}", flush=True)
    # Per-environment roll-up so the phased driver can gate on a
    # deterministic evaluation instead of the noisy training rollout.
    by_env: dict = {}
    for item in records:
        slot = by_env.setdefault(str(int(item["env_id"])), {
            "episodes": 0, "success": 0, "survival": 0, "collision": 0,
            "out_of_bounds": 0, "fell_into_gap": 0, "mean_failure_x": 0.0,
            "sum_return": 0.0, "sum_steps": 0.0, "mean_return": 0.0,
            "mean_steps": 0.0,
        })
        slot["episodes"] += 1
        slot["success"] += int(bool(item.get("success", False)))
        # Same rule as the train-path stage report: an episode counts as a
        # survival when it lasted at least SURVIVAL_STEPS.
        slot["survival"] += int(float(item.get("steps", 0.0)) >= SURVIVAL_STEPS)
        slot["collision"] += int(bool(item.get("collision", False)))
        slot["out_of_bounds"] += int(bool(item.get("out_of_bounds", False)))
        slot["fell_into_gap"] += int(bool(item.get("fell_into_gap", False)))
        slot["sum_return"] += float(item.get("mean_return", 0.0))
        slot["sum_steps"] += float(item.get("steps", 0.0))
    completion = sum(item["success"] and not item["fallen"] and not item["out_of_bounds"]
                     and not item["collision"] and not item.get("fell_into_gap", False)
                     for item in records) / max(len(records), 1)
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
            "command_speed_mean": sum(
                item.get("mean_command_speed", 0.0) * item["velocity_sample_count"]
                for item in records) / velocity_samples,
            "achieved_speed_mean": sum(
                item.get("mean_achieved_speed", 0.0) * item["velocity_sample_count"]
                for item in records) / velocity_samples,
            "success_rate": sum(item["success"] for item in records) / max(len(records), 1),
            "collision_rate": sum(item["collision"] for item in records) / max(len(records), 1),
            "fallen_rate": sum(item["fallen"] for item in records) / max(len(records), 1),
            "out_of_bounds_rate": sum(item["out_of_bounds"] for item in records) / max(len(records), 1),
            "low_height_failure_rate": sum(
                item.get("low_height_failure", False) for item in records
            ) / episodes,
            "tilt_failure_rate": sum(item.get("tilt_failure", False) for item in records) / max(len(records), 1),
            "absolute_height_failure_rate": sum(
                item.get("absolute_height_failure", False) for item in records
            ) / episodes,
            "fell_into_gap_rate": sum(
                item.get("fell_into_gap", False) for item in records
            ) / episodes,
            "timeout_rate": sum(
                not (
                    item["success"]
                    or item["collision"]
                    or item["fallen"]
                    or item["out_of_bounds"]
                    or item.get("fell_into_gap", False)
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
        # A trench entry ends the episode without setting fallen/collision/
        # out-of-bounds, so report it explicitly: a zero completion rate must
        # never be misread as "nothing happened".
        report["gap_fall_rate"] = sum(item["fell_into_gap"] for item in records) / max(len(records), 1)
        report["passed"] = (args.terrain == "course" and completion >= .95 and
                            report["velocity_xy_rmse"] <= .15 and report["yaw_rmse"] <= .25)
    else:
        # Report every termination cause: a Stage-2 report that only exposed
        # success/collision silently hid falls and course exits.
        report.update({
            "success_rate": sum(item["success"] for item in records) / max(len(records), 1),
            "collision_rate": sum(item["collision"] for item in records) / max(len(records), 1),
            "fallen_rate": sum(item["fallen"] for item in records) / max(len(records), 1),
            "out_of_bounds_rate": sum(item["out_of_bounds"] for item in records) / max(len(records), 1),
            "stale_goal_rate": sum(item["stale_goal"] for item in records) / max(len(records), 1),
            "fell_into_gap_rate": sum(
                item.get("fell_into_gap", False) for item in records
            ) / episodes,
            "timeout_rate": sum(
                not (
                    item["success"]
                    or item["collision"]
                    or item["fallen"]
                    or item["out_of_bounds"]
                    or item.get("fell_into_gap", False)
                )
                for item in records
            ) / episodes,
        })
        report["passed"] = (args.terrain == "course" and report["success_rate"] >= .90 and
                            report["collision_rate"] <= .05)
        # Speed-layer stability is reported, not assumed: the mean per-step change
        # of the commanded velocity and the share of large forward jumps.
        stability = [item for item in records if "mean_command_delta_l1" in item]
        if stability:
            report["mean_command_delta_l1"] = sum(
                item["mean_command_delta_l1"] for item in stability
            ) / len(stability)
            report["command_step_large_fraction"] = sum(
                item["command_step_large_fraction"] for item in stability
            ) / len(stability)
    # The phased driver reads mean_return/mean_steps off each entry; without
    # them it raised KeyError after the gate and never persisted the band
    # decision, so the curriculum re-ran the same stage forever.
    for entry in by_env.values():
        entry["mean_return"] = entry["sum_return"] / max(1, entry["episodes"])
        entry["mean_steps"] = entry["sum_steps"] / max(1, entry["episodes"])
    report["environments"] = by_env
    report["mean_command_speed"] = report.get("command_speed_mean", 0.0)
    report["mean_achieved_speed"] = report.get("achieved_speed_mean", 0.0)
    report["survival_rate"] = (
        sum(entry["survival"] for entry in by_env.values()) / max(1, len(records))
    )
    report["failure_rate"] = (
        sum(entry["collision"] + entry["out_of_bounds"] + entry["fell_into_gap"]
            for entry in by_env.values()) / max(1, len(records))
    )
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
