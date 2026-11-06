"""Opt-in per-episode reward coverage and strict terminal-record correlation."""
from __future__ import annotations

import math

import torch


REWARD_EVENTS = ("supported_short", "safe_segment", "tire_progress", "four_feet_jump_air_time")
TERMINAL_FLAGS = ("success", "fallen", "collision", "out_of_bounds", "fell_into_gap")


class EpisodeRewardTracker:
    def __init__(self, num_envs: int, device: str):
        self.counts = torch.zeros(num_envs, len(REWARD_EVENTS), dtype=torch.long, device=device)
        self.first_x = torch.full((num_envs, len(REWARD_EVENTS)), float("nan"), device=device)
        self.speed_min = torch.full((num_envs,), float("inf"), device=device)
        self.speed_max = torch.full((num_envs,), float("-inf"), device=device)
        self._last_tick = -1

    def update(self, tick: int, terms: dict[str, torch.Tensor], course_x: torch.Tensor,
               command_speed: torch.Tensor) -> None:
        if tick == self._last_tick:
            return
        self._last_tick = tick
        for index, name in enumerate(REWARD_EVENTS):
            triggered = terms[name] > 0
            first = triggered & (self.counts[:, index] == 0)
            self.first_x[:, index] = torch.where(first, course_x, self.first_x[:, index])
            self.counts[:, index] += triggered.to(self.counts.dtype)
        self.speed_min = torch.minimum(self.speed_min, command_speed)
        self.speed_max = torch.maximum(self.speed_max, command_speed)

    def snapshot(self, env_id: int) -> dict:
        counts = self.counts[env_id].tolist()
        x = self.first_x[env_id].tolist()
        low = float(self.speed_min[env_id].item())
        high = float(self.speed_max[env_id].item())
        return {
            "reward_events": dict(zip(REWARD_EVENTS, counts)),
            "first_reward_x": {
                name: (float(x[index]) if counts[index] else None)
                for index, name in enumerate(REWARD_EVENTS)
            },
            "command_speed_min": low if math.isfinite(low) else None,
            "command_speed_max": high if math.isfinite(high) else None,
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        self.counts[env_ids] = 0
        self.first_x[env_ids] = float("nan")
        self.speed_min[env_ids] = float("inf")
        self.speed_max[env_ids] = float("-inf")


def merge_episode_trace(done_ids: list[int], episode_rows: list[dict],
                        stage_records: list[dict]) -> list[dict]:
    """Pair only matching completed episodes, never a later auto-reset event."""
    actual = [int(row.get("env_id", -1)) for row in episode_rows]
    stages = [int(row.get("env", -1)) for row in stage_records]
    if (len(set(done_ids)) != len(done_ids) or len(actual) != len(done_ids)
            or sorted(actual) != sorted(done_ids) or sorted(stages) != sorted(done_ids)):
        raise ValueError("episode trace missing, duplicate or mismatched env ids")
    by_env = {int(row["env"]): row for row in stage_records}
    result = []
    for episode in episode_rows:
        env_id = int(episode["env_id"])
        stage = by_env[env_id]
        try:
            # Isaac's episode_metrics has no step count; the runtime's
            # pre-reset stage record is the only truthful completed count.
            if int(stage["steps"]) <= 0:
                raise ValueError("episode trace has no completed control steps")
            if "steps" in episode and int(stage["steps"]) != int(episode["steps"]):
                raise ValueError("episode trace step count differs from terminal record")
            for name in TERMINAL_FLAGS:
                if bool(stage[name]) != bool(episode[name]):
                    raise ValueError(f"episode trace {name} differs from terminal record")
            for key in ("course_x", "course_y", "command_speed"):
                if not math.isfinite(float(stage[key])):
                    raise ValueError(f"episode trace {key} is nonfinite")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("episode trace missing terminal data") from error
        merged = dict(stage)
        merged.pop("env", None)
        merged["env_id"] = env_id
        result.append(merged)
    return result
