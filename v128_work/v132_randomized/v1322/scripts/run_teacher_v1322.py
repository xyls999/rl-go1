"""Phased difficulty curriculum driver.

Outer loop (user rule): train every terrain family in one band; raise the
difficulty band only when the stage completion rate clears the gate; never drop
the lower bands from the mixture (they are replayed with a fixed share).
Inner loop: the LP-ACRL sampler decides, inside the allowed band, which
family/instance deserves the samples.

Terrain geometry is baked per stage, so the curriculum advances in stages rather
than inside a single training process. State lives in JSON files so a crash can
be resumed without losing the sampler or the band position.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

WORKSPACE = Path('/config/Desktop/sim')  # pinned remote workspace; driver itself is isolated
sys.path.insert(0, str(WORKSPACE / "source" / "go1_hier_sim"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from go1_hier_sim.envs.task_curriculum import (
    difficulty_value,  # noqa: E402
    BandCurriculum,
    DIFFICULTY_LEVELS,
    LPSampler,
    TERRAIN_FAMILIES,
)
from go1_hier_sim.envs.combined_roster_v32 import select_training_roster  # noqa: E402
from lp_acrl_stage import gate_decision  # noqa: E402

SENSOR_ENV = {
    # Sim-to-real sensing: latency, white noise, per-episode bias, scan dropout.
    # These are only the defaults; each phase overrides them from its disturbance
    # block so the disturbances ramp with the terrain band.
    "GO1_OBS_DELAY_STEPS": "1",
    "GO1_OBS_DELAY_RANDOM": "1",
    "GO1_OBS_NOISE_STD": "0.01",
    "GO1_OBS_BIAS_RAD": "0.02",
    "GO1_OBS_SCAN_DROPOUT": "0.05",
}


def disturbance_env(phase: dict) -> dict:
    """Map a phase's disturbance block onto the environment variables it drives."""
    block = phase.get("disturbance") or {}
    delay = int(block.get("obs_delay_steps", 1))
    return {
        "GO1_DR_MASS_UP": str(block.get("mass_up", 3.0)),
        "GO1_DR_PUSH_VEL": str(block.get("push_vel", 0.4)),
        "GO1_DR_MOTOR_PCT": str(block.get("motor_pct", 0.10)),
        "GO1_DR_COM_SCALE": str(block.get("com_scale", 1.0)),
        "GO1_OBS_DELAY_STEPS": str(delay),
        "GO1_OBS_DELAY_RANDOM": "1" if delay > 0 else "0",
        "GO1_OBS_NOISE_STD": str(block.get("obs_noise_std", 0.01)),
        "GO1_OBS_BIAS_RAD": str(block.get("obs_bias_rad", 0.02)),
        "GO1_OBS_SCAN_DROPOUT": str(block.get("obs_scan_dropout", 0.05)),
        # Joint friction spread (also covers stair risers and trench walls).
        "GO1_DR_FRICTION_STATIC_MIN": str(block.get("friction_static_min", 0.6)),
        "GO1_DR_FRICTION_STATIC_MAX": str(block.get("friction_static_max", 1.2)),
        "GO1_DR_FRICTION_DYNAMIC_MIN": str(block.get("friction_dynamic_min", 0.5)),
        "GO1_DR_FRICTION_DYNAMIC_MAX": str(block.get("friction_dynamic_max", 1.0)),
    }


def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def ensure_idle() -> None:
    """Never start a second simulator: 64 MB of /dev/shm cannot hold two.

    Match on the executable plus the simulator script path rather than on the
    bare string "sim.py": shell wrappers whose *text* merely mentions sim.py
    (status greps, watchdogs) are not simulators and used to stall the driver.
    """
    listing = _run([
        "bash", "-lc",
        "ps -eo comm=,args= | awk '$1 ~ /^(python|python3|torchrun)$/ "
        "&& $0 ~ /scripts\\/sim[.]py/ {print}' | head -5",
    ]).stdout.strip()
    if listing:
        raise SystemExit(f"another simulator process is running; refusing to start:\n{listing}")


def clean_shm() -> None:
    _run(["bash", "-lc", "rm -f /dev/shm/nccl-* 2>/dev/null || true"])


def latest_checkpoint(run_dir: Path) -> Path:
    files = list(Path(run_dir).glob("model_*.pt"))
    if not files:
        raise SystemExit(f"no checkpoint was written under {run_dir}")
    return max(files, key=lambda path: int(path.stem.split("_")[1]))


def results_path_for(config: dict) -> Path:
    """Keep new reward runs from appending to the legacy curriculum log."""
    return WORKSPACE / config.get("results_file", "lp_phased_results.txt")


def require_deterministic_gate_report(path: Path | None, config: dict) -> None:
    """A gated v1 stage must not substitute stochastic training rollout data."""
    if config.get("require_deterministic_gate") and (
        path is None or not path.is_file() or path.stat().st_size == 0
    ):
        raise SystemExit("deterministic gate report missing; stage left uncommitted")


def require_strict_gate_report(report_path: Path | None, trace_path: Path, count: int,
                               checkpoint: Path, seed: int, run_id: str | None = None) -> dict:
    """Fail closed when first-episode trace and report disagree on terminal outcomes."""
    if report_path is None or not report_path.is_file() or not trace_path.is_file():
        raise SystemExit("strict gate report or trace missing")
    if run_id is not None and report_path.name != f"evaluate_stage1_gate-{run_id}.json":
        raise SystemExit("strict gate report run provenance mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_env = report.get("environments", {})
    expected = {str(index) for index in range(count)}
    if (report.get("episodes") != count or not isinstance(by_env, dict) or set(by_env) != expected or
            any(not isinstance(slot, dict) or slot.get("episodes") != 1
                for slot in by_env.values())):
        raise SystemExit("strict gate report missing or duplicate environment episodes")
    if report.get("stage") != "stage1" or report.get("terrain") != "lp_acrl" or report.get("seed") != seed:
        raise SystemExit("strict gate report evaluation provenance mismatch")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if report.get("checkpoint_sha256") != digest.hexdigest():
        raise SystemExit("strict gate checkpoint hash mismatch")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    terminal_flags = ("success", "fallen", "collision", "out_of_bounds", "fell_into_gap")
    if (len(trace) != count or
            {str(row.get("env_id")) for row in trace} != expected or
            any(row.get("episode_index") != 0 or
                not {"required_gap_count", "credited_gap_count"} <= row.keys() or
                any(type(row.get(flag)) is not bool for flag in terminal_flags)
                for row in trace)):
        raise SystemExit("strict gate trace missing, duplicate or incomplete")
    for row in trace:
        slot = by_env[str(row["env_id"])]
        if any(slot.get(flag) != int(row[flag]) for flag in terminal_flags):
            raise SystemExit("strict gate report/trace terminal mismatch")
    rates = {f"{flag}_rate": sum(row[flag] for row in trace) / count for flag in terminal_flags}
    rates["completion_rate"] = sum(
        row["success"] and not any(row[flag] for flag in terminal_flags[1:])
        for row in trace) / count
    for name, expected_rate in rates.items():
        value = report.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or not math.isclose(
                value, expected_rate, abs_tol=1e-9):
            raise SystemExit(f"strict gate report/trace {name} mismatch")
    return report


def rebalance_family_plan(plan: dict, families: tuple[str, ...], difficulty_max: float,
                          family_priority: dict[str, float] | None = None) -> None:
    """Keep every family and preserve each slot's sampled difficulty and speed."""
    names = sorted(set(families))
    count = len(plan["instances"])
    if not names or count < len(names):
        raise ValueError("the stage needs at least one environment per terrain family")
    if family_priority is None:
        per, remainder = divmod(count, len(names))
        quota = {family: per + (index < remainder) for index, family in enumerate(names)}
    else:
        if set(family_priority) - set(names):
            raise ValueError("unknown family in family_priority")
        weights = {}
        for family in names:
            value = family_priority.get(family, 1.)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"family_priority[{family}] must be finite and positive")
            weights[family] = float(value)
        remaining = count - len(names)
        total = sum(weights.values())
        raw = {family: remaining * weights[family] / total for family in names}
        quota = {family: 1 + math.floor(raw[family]) for family in names}
        missing = count - sum(quota.values())
        for family in sorted(names, key=lambda name: (-(raw[name] - math.floor(raw[name])), name))[:missing]:
            quota[family] += 1
    assignment = [family for family in names for _ in range(quota[family])]
    plan["instances"] = [[assignment[i], level, speed]
                         for i, (_, level, speed) in enumerate(plan["instances"])]
    plan["families"] = [family for family, _, _ in plan["instances"]]
    plan["difficulty_level"] = [level for _, level, _ in plan["instances"]]
    plan["speed_level"] = [level for _, _, level in plan["instances"]]
    plan["difficulty"] = [difficulty_value(level, difficulty_max)
                          for level in plan["difficulty_level"]]


def make_balanced_gate_plan(train_plan: dict, families: tuple[str, ...],
                            difficulty_max: float) -> dict:
    gate_plan = copy.deepcopy(train_plan)
    rebalance_family_plan(gate_plan, families, difficulty_max)
    return gate_plan


def run_stage(run_id: str, plan_path: Path, phase: dict, config: dict, envs: int,
              episode_seconds: float, warmup: int, init_checkpoint: Path) -> None:
    command = [
        "bash", str(WORKSPACE / "scripts" / "torchrun_sim.sh"), "train",
        "--stage", "stage1", "--terrain", "lp_acrl", "--headless",
        "--num-envs", str(envs), "--iterations", str(int(phase["iterations"])),
        "--episode-seconds", str(episode_seconds),
        "--course-difficulty-min", "0.0",
        "--course-difficulty-max", str(config["difficulty_max"]),
        "--speed-min", str(config["speed_min"]), "--speed-max", str(config["speed_max"]),
        "--initialize-from", str(init_checkpoint),
        "--critic-warmup-iterations", str(int(warmup)),
        "--run-name", run_id,
    ]
    environment = dict(os.environ)
    environment.update(SENSOR_ENV)
    environment.update(disturbance_env(phase))
    environment["GO1_RUN_ID"] = run_id
    environment["GO1_TASK_PLAN"] = str(plan_path)
    environment.setdefault("GO1_REWARD_CONFIG_PATH",
                           str(WORKSPACE / "configs" / "rewards_stage1_gaps_v5.toml"))
    environment.setdefault("GO1_PPO_CONFIG_PATH", str(WORKSPACE / "configs" / "ppo_gaps_long.json"))
    log_path = WORKSPACE / f"v35_phased_{run_id}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise SystemExit(f"stage {run_id} failed (exit {completed.returncode}); see {log_path}")


def holdout_seed(config: dict) -> int:
    """Require a deliberately selected seed; Task 3 verifies it has not been used."""
    seed = config.get("gate_holdout_seed")
    if type(seed) is not int or seed < 0 or seed == 2028:
        raise ValueError("V132.2 requires a fresh explicit gate_holdout_seed (not 2028)")
    return seed


def gate_trace_path(workspace: Path, run_id: str) -> Path:
    """Each gate owns a distinct trace; never reuse even its own previous output."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("gate run_id must be one safe path component")
    trace_path = workspace / "reports" / f"v1322_{run_id}_trace.jsonl"
    if trace_path.exists():
        raise SystemExit(f"strict gate trace already exists: {trace_path}")
    return trace_path


def make_gate_command(workspace: Path, run_id: str, envs: int,
                      episode_seconds: float, checkpoint: Path, config: dict,
                      trace_path: Path) -> list[str]:
    """Evaluate the training speed band using the same trace path as the validator."""
    if envs != 512:
        raise ValueError("V132.2 strict gate requires 512 environments")
    return [
        str(workspace / "sim.sh"), "evaluate",
        "--stage", "stage1", "--terrain", "lp_acrl", "--headless",
        "--num-envs", str(envs), "--episodes", str(envs),
        "--strict-first-episode", "--episode-trace", str(trace_path),
        "--episode-seconds", str(episode_seconds),
        "--speed-min", str(config.get("gate_speed_min", config["speed_min"])),
        "--speed-max", str(config.get("gate_speed_max", config["speed_max"])),
        "--seed", str(holdout_seed(config)), "--checkpoint", str(checkpoint),
        "--run-name", f"gate-{run_id}",
    ]


def run_gate_eval(run_id: str, plan_path, phase, config, envs, episode_seconds,
                  checkpoint, trace_path: Path) -> Path | None:
    """Deterministic (argmax-action) evaluation of the finished stage."""
    command = make_gate_command(WORKSPACE, run_id, envs, episode_seconds, checkpoint,
                                config, trace_path)
    environment = dict(os.environ)
    environment.update(SENSOR_ENV)
    environment.update(disturbance_env(phase))
    environment["GO1_TASK_PLAN"] = str(plan_path)
    environment["GO1_RUN_ID"] = f"gate-{run_id}"
    environment.setdefault("GO1_REWARD_CONFIG_PATH",
                           str(WORKSPACE / "configs" / "rewards_stage1_gaps_v5.toml"))
    environment.setdefault("GO1_PPO_CONFIG_PATH", str(WORKSPACE / "configs" / "ppo_gaps_long.json"))
    if (WORKSPACE / "reports" / f"evaluate_stage1_gate-{run_id}.json").exists():
        raise SystemExit("strict gate report already exists")
    log_path = WORKSPACE / f"v35_gate_{run_id}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=stream, text=True)
    if completed.returncode != 0:
        return None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("REPORT:"):
            return Path(line.split("REPORT:", 1)[1].strip())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="phased difficulty curriculum driver")
    parser.add_argument("--config", default=str(WORKSPACE / "configs" / "curriculum_phases.json"))
    parser.add_argument("--base-id", default=None)
    parser.add_argument("--max-stages", type=int, default=None)
    parser.add_argument("--target-level", type=int, default=None,
                        help="initial difficulty band (default: lowest)")
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    families = select_training_roster(config, TERRAIN_FAMILIES + (
        "course", "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8",
        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",
        "combined_soft17", "combined_soft18", "combined_soft19", "combined_soft20", "combined_soft21",
        "combined_soft22", "combined_soft23", "combined_part_highstairs", "combined_part_entrance",
        "combined_part_slopes", "combined_part_midstairs",
        "combined_slice_a", "combined_slice_b", "combined_slice_c",
        "combined_junction",
    ))
    gate_families = (str(config["gate_family"]),) if config.get("gate_family") else families
    gate_spec = dict(config["gate"])
    envs = int(args.envs or config["num_envs"])
    episode_seconds = float(args.episode_seconds or config["episode_seconds"])
    max_stages = int(args.max_stages or config["max_stages"])
    phases = config["phases"]
    base_id = args.base_id or __import__("datetime").datetime.utcnow().strftime(
        "%Y-%m-%d_%H-%M-%S") + "_go1-phased"

    state_path = WORKSPACE / config["state"]
    sampler_state = WORKSPACE / config["sampler_state"]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"RESUME state={state_path} {json.dumps(state['position'])}")
    else:
        state = {
            "position": {"target_level": 1 if args.target_level is None else int(args.target_level),
                         "phase_index": 0, "stage": 0},
            "history": [],
        }
    if args.target_level is not None:
        state["position"]["target_level"] = int(args.target_level)

    results = results_path_for(config)

    def record(line: str) -> None:
        print(line, flush=True)
        with results.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    band = BandCurriculum()
    init_checkpoint = WORKSPACE / config["init_checkpoint"]
    # Resume from the recorded position instead of restarting the curriculum.
    # Without this a watchdog relaunch after a crash re-ran stage 1 from the
    # original warm-start checkpoint and threw away every finished stage.
    stage_count = int(state["position"].get("stage", 0))
    if state["history"]:
        last_run = str(state["history"][-1].get("run", ""))
        if last_run:
            try:
                resumed = latest_checkpoint(WORKSPACE / "checkpoints" / "stage1" / last_run)
            except (FileNotFoundError, StopIteration):
                resumed = None
            if resumed is not None:
                init_checkpoint = resumed
                print(f"RESUME checkpoint={resumed}")
    holds = 0
    while stage_count < max_stages:
        position = state["position"]
        target_level = int(position["target_level"])
        # Disturbance strength is a function of the difficulty level, not a
        # second thing that jumps on every advance: one phase per two levels.
        phase_index = min(max(0, (int(target_level) - 1) // 3), len(phases) - 1)
        position["phase_index"] = phase_index
        phase = phases[phase_index]
        stage_count += 1
        run_id = f"{base_id}-s{stage_count:02d}-{phase['name']}-d{target_level}"
        trace_path = gate_trace_path(WORKSPACE, run_id)
        holdout_seed(config)
        plan_path = WORKSPACE / "reports" / f"lp_plan_{run_id}.json"
        warmup = int(config["warmup_first"]) if stage_count == 1 else int(config["warmup"])
        speed_levels = phase.get("speed_levels")

        sampler = LPSampler(families, temperature=float(config["temperature"]),
                            min_family_share=float(config.get("min_family_share", 0.06)),
                            max_family_share=float(config.get("max_family_share", 0.25)),
                            state_path=sampler_state)
        weights = band.instance_weights(sampler.instances, target_level, speed_levels)
        rng = np.random.default_rng(2026 + sampler.stage)
        plan = sampler.stage_plan(
            envs, rng,
            max_difficulty=float(config["difficulty_max"]),
            speed_min=float(config["speed_min"]), speed_max=float(config["speed_max"]),
            instance_weights=weights, target_level=target_level,
            allowed_speed_levels=speed_levels,
            min_family_share=float(config.get("min_family_share", 0.06)),
            max_family_share=float(config.get("max_family_share", 0.25)),
        )
        # Preserve LP-ACRL's sampled difficulty/speed axis. Explicit v11 family
        # priorities improve mixed exposure without deleting any of the 11 maps.
        rebalance_family_plan(plan, families, float(config["difficulty_max"]),
                              family_priority=config.get("family_priority"))
        if len(plan["instances"]) != envs or set(plan["families"]) != set(families):
            raise SystemExit("Stage-C teacher plan must contain only the combined family")

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for family in plan["families"]:
            counts[family] = counts.get(family, 0) + 1
        record(f"[{run_id}] PLAN target={target_level} band="
               f"{ {str(k): round(v, 3) for k, v in band.level_shares(target_level).items()} } "
               f"replay={band.replay_share(target_level):.3f} speeds={speed_levels} "
               f"mean_difficulty={np.mean(plan['difficulty']):.3f}")
        record(f"[{run_id}] DISTURBANCE {json.dumps(disturbance_env(phase), sort_keys=True)}")
        record(f"[{run_id}] FAMILIES {json.dumps(counts, sort_keys=True)}")

        ensure_idle()
        clean_shm()
        run_stage(run_id, plan_path, phase, config, envs, episode_seconds, warmup, init_checkpoint)
        init_checkpoint = latest_checkpoint(WORKSPACE / "checkpoints" / "stage1" / run_id)

        report_path = WORKSPACE / "reports" / f"stage_{run_id}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # The training emphasis must not weaken the 40-episode-per-family gate.
        gate_plan = make_balanced_gate_plan(plan, gate_families, float(config["difficulty_max"]))
        if (len(gate_plan["instances"]) != envs or set(gate_plan["families"]) != set(gate_families)
                or min(gate_plan["families"].count(family) for family in gate_families) < 40):
            raise SystemExit("Stage-C teacher gate must cover the real family with 40+ slots")
        gate_plan_path = WORKSPACE / "reports" / f"lp_gate_plan_{run_id}.json"
        gate_plan_path.write_text(json.dumps(gate_plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        record(f"[{run_id}] GATE_PLAN {gate_plan_path.name} balanced_{len(gate_families)}_families")
        gate_report_path = run_gate_eval(run_id, gate_plan_path, phase, config, envs,
                                        episode_seconds, init_checkpoint, trace_path)
        report = require_strict_gate_report(gate_report_path, trace_path, envs,
                                            init_checkpoint, holdout_seed(config), run_id)
        record(f"[{run_id}] GATE source=deterministic_eval {gate_report_path.name}")
        decision, stats = gate_decision(gate_plan, report, gate_spec)
        record(f"[{run_id}] GATE result={decision} completion={stats['completion_rate']:.4f} "
               f"episodes={stats['episodes']} survival={stats['survival_rate']:.4f} "
               f"failure={stats['failure_rate']:.4f} families_ok={stats['families_ok']}/"
               f"{len(stats['families'])} cmd={stats['mean_command_speed']:.3f} "
               f"achv={stats['mean_achieved_speed']:.3f}")
        for family, entry in sorted(stats["families"].items(),
                                    key=lambda kv: kv[1]["completion_rate"]):
            record(f"[{run_id}]   {family:18s} {entry['completion_rate']:.3f} "
                   f"({entry['episodes']} ep, fail_x={entry['mean_failure_x']:.1f})")

        per_env = []
        environments = report.get("environments", {})
        stage_mean = float(np.mean([entry["mean_return"] for entry in environments.values()])) \
            if environments else 0.0
        for index in range(len(gate_plan["instances"])):
            entry = environments.get(str(index))
            per_env.append(stage_mean if entry is None else float(entry["mean_return"]))
        sampler.observation_from_report(gate_plan, per_env)
        sampler.update()
        sampler.save()
        record(f"[{run_id}] UPDATE sampler_stage={sampler.stage} ckpt={init_checkpoint.name}")

        top_level = DIFFICULTY_LEVELS - 1
        regress_after = int(config.get("regress_after_holds", 3))
        regress_floor = float(config.get("regress_completion_floor", 0.3))
        if decision == "hold":
            holds += 1
            # A hard band opens with near-zero completion by construction, so a
            # low completion alone must not lower the band (the user rule is
            # "difficulty only rises"). Only a band that stays near-zero after
            # several consecutive holds escapes one level down.
            if holds >= regress_after and stats["completion_rate"] < regress_floor:
                decision = "regress"
                record(f"[{run_id}] ESCAPE {holds} consecutive holds at level {target_level} "
                       f"with completion={stats['completion_rate']:.4f}; stepping down")
        else:
            holds = 0
        if decision == "advance":
            position["target_level"] = min(top_level, target_level + 1)
            position["phase_index"] = min(
                len(phases) - 1, max(0, (int(target_level) - 1) // 3)
            )
        elif decision == "regress":
            position["target_level"] = max(0, target_level - 1)
        position["stage"] = stage_count
        state["history"].append({"run": run_id, "target_level": target_level,
                                 "phase": phase["name"], "decision": decision, **stats})
        state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        if decision == "advance" and target_level >= top_level:
            record(f"[{run_id}] TOP BAND CLEARED at level {top_level}; stopping for validation")
            # The watchdog uses this flag to stop relaunching the driver, so the
            # completed curriculum is left untouched for evaluation.
            done_flag = WORKSPACE / f"lp_phased_done_{base_id}.flag"
            done_flag.write_text(f"cleared level {top_level} in {run_id}\n", encoding="utf-8")
            break

    record(f"[{base_id}] PHASED CURRICULUM STOPPED after {stage_count} stage(s)")
    print(json.dumps(state["position"], indent=2))


if __name__ == "__main__":
    main()
