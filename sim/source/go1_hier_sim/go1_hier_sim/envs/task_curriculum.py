"""LP-ACRL task sampler (paper Sec. III-B): learning-progress automatic curriculum.

The task space is discrete: each instance is one terrain family crossed with a
difficulty level and a speed level. The sampler estimates, per instance, the
change in mean episodic reward between successive curriculum stages --

    LP(instance) = R_new(instance) - R_old(instance)

-- and turns those estimates into a sampling distribution with a temperatured
softmax, so the next stage is spent where learning is still progressing instead
of on tasks that are already solved or permanently out of reach. A uniform
mixture floor keeps any single instance from starving, which is what the paper
reports distinguishes LP from absolute-learning-progress baselines.

The module is intentionally free of simulation imports so the curriculum logic
can be tested without Isaac Lab. State is a plain JSON file so a training stage
can be resumed after a crash and still continue the same curriculum.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# The paper's categorical terrain dimension: each family is a task dimension of
# its own, so terrain type is sampled, not blended into one mixture.
TERRAIN_FAMILIES = (
    "stairs_up_long", "stairs_down_long", "stairs_mixed",
    "slope_up", "slope_down", "slope_mixed", "gaps_only", "combined_flat",
)

# The paper's extended task space: terrain type x 4 geometry-difficulty levels x
# speed levels (5 m/s levels in Fig. 8). The exact numbers are ours; the shape
# of the space is the paper's.
DIFFICULTY_LEVELS = 4
SPEED_LEVELS = 5
# Uniform-mixture floor: the same weight is mixed into every instance so a task
# that plateaus (LP -> 0) is still revisited occasionally and the policy cannot
# catastrophically forget an earlier family.
UNIFORM_FLOOR = 0.25
# Hard cap on any single instance's share of a stage. Without it a large progress
# spike collapses the softmax onto one instance (observed: 180 of 256 envs on a
# single slope instance), which contradicts "train every family every stage" and
# risks forgetting the rest.
MAX_INSTANCE_SHARE = 0.04
# Minimum share of a stage that each terrain family keeps. Learning-progress
# sampling starves exactly the families that are not improving, but the
# curriculum gate needs every family at the completion bar, so the two would
# fight each other without this floor.
MIN_FAMILY_SHARE = 0.06
# Symmetric ceiling: progress sampling will happily give one improving family a
# large share, but the gate needs every family at the bar, so no single family
# may take most of a stage.
MAX_FAMILY_SHARE = 0.25

# Outer difficulty-band rule (user requirement): the band rises only when a stage
# clears the completion gate, and each stage keeps a fixed share of the lower
# levels so raising the ceiling never erases what already works. The three-tuples
# are shares for the target level, the level below it and everything deeper.
EARLY_BAND_MIX = (0.70, 0.20, 0.10)
LATE_BAND_MIX = (0.80, 0.15, 0.05)
MIN_REPLAY_SHARE = 0.15


def difficulty_value(level: int, max_difficulty: float, levels: int = DIFFICULTY_LEVELS) -> float:
    """Map a discrete level to its geometric difficulty (paper App. C)."""
    if not 0 <= level < levels:
        raise ValueError(f"difficulty level must be in [0, {levels - 1}]")
    return float(max_difficulty) * level / max(1, levels - 1)


def speed_interval(level: int, speed_min: float, speed_max: float,
                   levels: int = SPEED_LEVELS) -> tuple[float, float]:
    """Map a discrete level to its sub-interval of the commanded speed band."""
    if not 0 <= level < levels:
        raise ValueError(f"speed level must be in [0, {levels - 1}]")
    span = (float(speed_max) - float(speed_min)) / levels
    low = float(speed_min) + span * level
    return low, low + span


class BandCurriculum:
    """Outer difficulty-band schedule layered on top of the progress sampler.

    Difficulty advances a band at a time once a stage clears the completion gate,
    and every band keeps the lower levels in the mixture (the user's
    anti-forgetting rule). Within the allowed band, the LP sampler decides which
    family/instance deserves the samples.
    """

    def __init__(self, difficulty_levels: int = DIFFICULTY_LEVELS,
                 min_replay_share: float = MIN_REPLAY_SHARE):
        if difficulty_levels < 1:
            raise ValueError("the band curriculum needs at least one difficulty level")
        self.difficulty_levels = int(difficulty_levels)
        self.min_replay_share = float(min_replay_share)

    def level_shares(self, target_level: int) -> dict[int, float]:
        """Difficulty-level mixture for one stage, normalised to sum to one."""
        if not 0 <= target_level < self.difficulty_levels:
            raise ValueError(f"target level must be in [0, {self.difficulty_levels - 1}]")
        template = LATE_BAND_MIX if target_level >= 3 else EARLY_BAND_MIX
        mixture: dict[int, float] = {}
        for offset, weight in enumerate(template):
            level = target_level - offset
            if offset == len(template) - 1:
                # Tail bucket: every remaining lower level shares it uniformly,
                # and when no lower level remains at all the weight merges into
                # level 0 instead of being dropped, so the target level keeps its
                # documented share (0.70 early, 0.80 late).
                if level > 0:
                    per_level = weight / (level + 1)
                    for deeper in range(level + 1):
                        mixture[deeper] = mixture.get(deeper, 0.0) + per_level
                else:
                    mixture[0] = mixture.get(0, 0.0) + weight
                break
            if level < 0:
                break
            mixture[level] = mixture.get(level, 0.0) + weight
        total = sum(mixture.values())
        if total <= 0.0:
            raise ValueError("the band mixture has no mass")
        return {level: weight / total for level, weight in mixture.items()}

    def replay_share(self, target_level: int) -> float:
        """Fraction of the stage spent on levels below the target (never zero for L>0)."""
        return 1.0 - self.level_shares(target_level).get(target_level, 0.0)

    def instance_weights(self, instances, target_level: int,
                         allowed_speed_levels=None) -> np.ndarray:
        """Per-instance multiplier for the sampler, normalised to sum to one."""
        mixture = self.level_shares(target_level)
        allowed_speeds = None if allowed_speed_levels is None else {
            int(level) for level in allowed_speed_levels
        }
        weights = np.zeros(len(instances), dtype=np.float64)
        for index, (_, difficulty_level, speed_level) in enumerate(instances):
            if allowed_speeds is not None and int(speed_level) not in allowed_speeds:
                continue
            weights[index] = mixture.get(int(difficulty_level), 0.0)
        if weights.sum() <= 0.0:
            raise ValueError("the band curriculum excluded every instance; widen the band")
        return weights / weights.sum()


class LPSampler:
    """Learning-progress sampling over (family, difficulty level, speed level)."""

    def __init__(self, families, difficulty_levels: int = DIFFICULTY_LEVELS,
                 speed_levels: int = SPEED_LEVELS, temperature: float = 1.0,
                 uniform_floor: float = UNIFORM_FLOOR, reward_smoothing: float = 0.5,
                 max_instance_share: float = MAX_INSTANCE_SHARE,
                 min_family_share: float = MIN_FAMILY_SHARE,
                 max_family_share: float = MAX_FAMILY_SHARE,
                 state_path: str | Path | None = None):
        if not families:
            raise ValueError("the task space needs at least one terrain family")
        self.families = tuple(families)
        self.difficulty_levels = int(difficulty_levels)
        self.speed_levels = int(speed_levels)
        self.temperature = float(temperature)
        self.uniform_floor = float(uniform_floor)
        self.reward_smoothing = float(reward_smoothing)
        self.max_instance_share = float(max_instance_share)
        self.min_family_share = float(min_family_share)
        self.max_family_share = float(max_family_share)
        self.state_path = None if state_path is None else Path(state_path)
        self.instances = [
            (family, difficulty, speed)
            for family in self.families
            for difficulty in range(self.difficulty_levels)
            for speed in range(self.speed_levels)
        ]
        self.reward: dict[tuple, float] = {}
        self.lp: dict[tuple, float] = {}
        self.stage = 0
        if self.state_path is not None and self.state_path.is_file():
            self.load(self.state_path)

    # -- estimation ---------------------------------------------------------
    def observe(self, instance, reward: float) -> None:
        """Record a stage's mean episodic reward for one instance."""
        instance = tuple(instance)
        if instance not in self.reward:
            self.reward[instance] = float(reward)
            self.lp[instance] = 0.0
            return
        previous = self.reward[instance]
        smoothed = (1.0 - self.reward_smoothing) * previous + self.reward_smoothing * float(reward)
        self.lp[instance] = smoothed - previous
        self.reward[instance] = smoothed

    def update(self) -> None:
        """Advance one curriculum stage and refresh the sampling distribution."""
        self.stage += 1
        if self.state_path is not None:
            self.save(self.state_path)

    def probabilities(self) -> dict[tuple, float]:
        """Softmax over the learning progress, mixed with a uniform floor."""
        if not self.lp:
            uniform = 1.0 / len(self.instances)
            return {instance: uniform for instance in self.instances}
        # Temperature is scaled by the observed progress spread so one unit of
        # temperature means "one typical learning-progress step", which keeps
        # the default usable across reward scales.
        spread = float(np.std([self.lp.get(i, 0.0) for i in self.instances]))
        scale = self.temperature * max(spread, 1e-6)
        logits = {instance: self.lp.get(instance, 0.0) / scale for instance in self.instances}
        peak = max(logits.values())
        weights = {instance: math.exp(value - peak) for instance, value in logits.items()}
        total = sum(weights.values())
        softmax = {instance: weight / total for instance, weight in weights.items()}
        floor = self.uniform_floor
        uniform = 1.0 / len(self.instances)
        probabilities = {
            instance: (1.0 - floor) * softmax[instance] + floor * uniform
            for instance in self.instances
        }
        if 0.0 < self.max_instance_share < 1.0:
            # Water-filling: clip the peaks and redistribute the excess over the
            # rest until no instance exceeds the cap. Plain clip-and-renormalise
            # cannot guarantee the cap when the distribution is very peaked, and
            # the cap is what keeps every family present in every stage.
            cap = max(self.max_instance_share, 1.0 / len(probabilities))
            for _ in range(64):
                excess = sum(max(0.0, value - cap) for value in probabilities.values())
                if excess <= 1e-12:
                    break
                over = {key for key, value in probabilities.items() if value > cap}
                for key in over:
                    probabilities[key] = cap
                rest = [key for key in probabilities if key not in over]
                base = sum(probabilities[key] for key in rest)
                if base <= 0.0:
                    break
                for key in rest:
                    probabilities[key] += excess * probabilities[key] / base
            total = sum(probabilities.values())
            probabilities = {key: value / total for key, value in probabilities.items()}
        return probabilities

    # -- sampling -----------------------------------------------------------
    @staticmethod
    def _balance_family_mass(weights: np.ndarray, instances, floor: float,
                             ceiling: float) -> np.ndarray:
        """Hold every terrain family between `floor` and `ceiling` of the mass.

        Learning progress concentrates on the instances that are improving and
        starves the ones that are not, but the curriculum gate requires every
        family to reach the completion bar. The floor keeps all terrain types in
        every stage, and the ceiling stops one improving family from crowding
        the others out.

        Targets are the clipped proportional (I-projection) solution
        ``t_f = clip(p_f + lam, floor, ceiling)`` with ``lam`` bisected so the
        targets sum to one. Scaling the over-ceiling families and renormalising
        does not converge: renormalisation pushes them back above the ceiling,
        so the floor branch never runs and the starved families are squeezed a
        little further on every iteration. That is what pinned
        ``stairs_up_long`` to 21 of 256 environments while the configured floor
        was 26, and let ``gaps_only`` sit at 67 against a ceiling of 64.
        """
        families = sorted({instance[0] for instance in instances})
        if not families:
            return weights
        count = len(families)
        floor = min(max(float(floor), 0.0), 1.0 / count)
        ceiling = max(float(ceiling), 1.0 / count)
        if floor * count > 1.0 + 1e-9:
            floor = 1.0 / count
        if ceiling * count < 1.0 - 1e-9:
            ceiling = 1.0 / count
        masks = {family: np.asarray([i[0] == family for i in instances]) for family in families}
        total = float(weights.sum())
        if total <= 0.0:
            return weights
        current = {family: float(weights[masks[family]].sum()) / total for family in families}
        if floor >= ceiling:
            targets = {family: 1.0 / count for family in families}
        else:
            low, high = -1.0, 1.0
            for _ in range(80):
                middle = 0.5 * (low + high)
                shifted = sum(
                    min(max(current[family] + middle, floor), ceiling) for family in families
                )
                if shifted > 1.0:
                    high = middle
                else:
                    low = middle
            offset = 0.5 * (low + high)
            targets = {
                family: min(max(current[family] + offset, floor), ceiling) for family in families
            }
        balanced = weights.copy()
        for family in families:
            if current[family] <= 0.0:
                balanced[masks[family]] = targets[family] / max(int(masks[family].sum()), 1)
                continue
            balanced[masks[family]] *= targets[family] / current[family]
        return balanced

    def sample(self, count: int, rng: np.random.Generator,
               instance_weights=None, allowed_speed_levels=None,
               min_family_share=None, max_family_share=None) -> list[tuple]:
        if count < 1:
            raise ValueError("sample count must be positive")
        probabilities = self.probabilities()
        instances = list(probabilities)
        weights = np.asarray([probabilities[i] for i in instances], dtype=np.float64)
        if allowed_speed_levels is not None:
            # The speed band is part of the outer schedule, so it is enforced here
            # as well as in the band weights: a speed level outside the current
            # band must never be sampled, whatever weights the caller passed.
            allowed = {int(level) for level in allowed_speed_levels}
            for index, instance in enumerate(instances):
                if int(instance[2]) not in allowed:
                    weights[index] = 0.0
        if instance_weights is not None:
            # Band multipliers (difficulty mixture, allowed speed levels) are
            # applied on top of the progress distribution before normalising.
            band = np.asarray(instance_weights, dtype=np.float64)
            if band.shape != weights.shape:
                raise ValueError(
                    f"instance weights have shape {band.shape}, expected {weights.shape}"
                )
            weights = weights * band
        if weights.sum() <= 0.0:
            raise ValueError("no instance is eligible for this stage; widen the band")
        weights = weights / weights.sum()
        floor = self.min_family_share if min_family_share is None else float(min_family_share)
        ceiling = self.max_family_share if max_family_share is None else float(max_family_share)
        weights = self._balance_family_mass(weights, instances, floor, ceiling)
        picked = rng.choice(len(instances), size=count, p=weights)
        return [instances[int(index)] for index in picked]

    def stage_plan(self, count: int, rng: np.random.Generator,
                   max_difficulty: float, speed_min: float, speed_max: float,
                   instance_weights=None, target_level: int | None = None,
                   allowed_speed_levels=None, min_family_share=None,
                   max_family_share=None) -> dict:
        """Per-environment assignment for one training stage."""
        sample = self.sample(count, rng, instance_weights=instance_weights,
                             allowed_speed_levels=allowed_speed_levels,
                             min_family_share=min_family_share,
                             max_family_share=max_family_share)
        return {
            "stage": self.stage,
            "temperature": self.temperature,
            "target_level": target_level,
            "allowed_speed_levels": None if allowed_speed_levels is None else sorted(
                int(level) for level in allowed_speed_levels
            ),
            "families": [item[0] for item in sample],
            "difficulty": [difficulty_value(item[1], max_difficulty) for item in sample],
            "difficulty_level": [item[1] for item in sample],
            "speed_level": [item[2] for item in sample],
            "speed_min": float(speed_min),
            "speed_max": float(speed_max),
            "instances": [list(item) for item in sample],
        }

    def observation_from_report(self, plan: dict, per_env_reward: list[float]) -> None:
        """Fold one stage report back into the estimates, per instance.

        Environments sharing an instance are averaged, which is exactly the
        paper's expected episodic reward R_cj(zeta) for that task instance.
        """
        if len(per_env_reward) != len(plan["instances"]):
            raise ValueError("stage report does not match the plan length")
        buckets: dict[tuple, list[float]] = {}
        for instance, reward in zip(plan["instances"], per_env_reward):
            buckets.setdefault(tuple(instance), []).append(float(reward))
        for instance, values in buckets.items():
            self.observe(instance, float(np.mean(values)))

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path | None = None) -> None:
        path = Path(path or self.state_path)
        payload = {
            "families": list(self.families),
            "difficulty_levels": self.difficulty_levels,
            "speed_levels": self.speed_levels,
            "temperature": self.temperature,
            "uniform_floor": self.uniform_floor,
            "reward_smoothing": self.reward_smoothing,
            "max_instance_share": self.max_instance_share,
            "min_family_share": self.min_family_share,
            "max_family_share": self.max_family_share,
            "stage": self.stage,
            "reward": {",".join(map(str, k)): v for k, v in self.reward.items()},
            "lp": {",".join(map(str, k)): v for k, v in self.lp.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload["families"]) != self.families:
            raise ValueError("curriculum state was built for a different terrain family list")
        self.stage = int(payload.get("stage", 0))
        self.reward = {}
        self.lp = {}
        for key, value in payload.get("reward", {}).items():
            parts = key.split(",")
            self.reward[(parts[0], int(parts[1]), int(parts[2]))] = float(value)
        for key, value in payload.get("lp", {}).items():
            parts = key.split(",")
            self.lp[(parts[0], int(parts[1]), int(parts[2]))] = float(value)
