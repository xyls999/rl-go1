"""Task-completion contract for V1333 long-gap traces."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


def latch_finish_reached(
    already_reached: torch.Tensor,
    reached_this_step: torch.Tensor,
) -> torch.Tensor:
    """Keep finish entry latched until the environment resets the episode."""
    if not isinstance(already_reached, torch.Tensor) or not isinstance(reached_this_step, torch.Tensor):
        raise TypeError("finish-latch inputs must be tensors")
    if already_reached.shape != reached_this_step.shape:
        raise ValueError("finish-latch tensor shapes must match")
    return already_reached.bool() | reached_this_step.bool()


def task_completion_mask(
    finish_reached: torch.Tensor,
    fallen: torch.Tensor,
    fell_into_gap: torch.Tensor,
) -> torch.Tensor:
    """Complete only after the finish was reached without a fall or gap fall."""
    tensors = (finish_reached, fallen, fell_into_gap)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("completion flags must be tensors")
    if not (finish_reached.shape == fallen.shape == fell_into_gap.shape):
        raise ValueError("completion flag tensor shapes must match")
    return finish_reached.bool() & ~fallen.bool() & ~fell_into_gap.bool()


def strict_completion_termination_mask(
    finish_reached: torch.Tensor,
    fallen: torch.Tensor,
    fell_into_gap: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Return strict completion only for profiles that opt into the term."""
    if type(enabled) is not bool:
        raise TypeError("strict-completion termination enablement must be a boolean")
    completion = task_completion_mask(finish_reached, fallen, fell_into_gap)
    return completion if enabled else torch.zeros_like(completion, dtype=torch.bool)


def stage1_task_termination_mask(
    fallen: torch.Tensor,
    out_of_bounds: torch.Tensor,
    success: torch.Tensor,
    fell_into_gap: torch.Tensor,
    finish_reached: torch.Tensor,
    *,
    strict_completion_enabled: bool,
) -> torch.Tensor:
    """Keep failure termination and select the opt-in or legacy endpoint."""
    if type(strict_completion_enabled) is not bool:
        raise TypeError("strict-completion mode must be a boolean")
    tensors = (fallen, out_of_bounds, success, fell_into_gap, finish_reached)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("Stage-1 termination flags must be tensors")
    if any(value.shape != fallen.shape for value in tensors[1:]):
        raise ValueError("Stage-1 termination flag tensor shapes must match")
    endpoint = (
        strict_completion_termination_mask(
            finish_reached, fallen, fell_into_gap, enabled=True
        )
        if strict_completion_enabled else success.bool()
    )
    return fallen.bool() | out_of_bounds.bool() | fell_into_gap.bool() | endpoint


def _flag(row: Mapping, name: str) -> bool:
    value = row.get(name)
    if type(value) is not bool:
        raise ValueError(f"episode trace {name} must be a boolean")
    return value


def score_first_episode_rows(
    rows: Iterable[Mapping], expected_env_count: int = 512,
) -> dict:
    """Score exactly the earliest trace episode for every expected environment."""
    if type(expected_env_count) is not int or expected_env_count < 1:
        raise ValueError("expected_env_count must be a positive integer")

    first: dict[int, tuple[int, Mapping]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("episode trace rows must be objects")
        env_id, episode_index = row.get("env_id"), row.get("episode_index")
        if type(env_id) is not int or type(episode_index) is not int or episode_index < 0:
            raise ValueError("episode trace env_id and episode_index must be nonnegative integers")
        if not 0 <= env_id < expected_env_count:
            raise ValueError(f"episode trace environment id out of range: {env_id}")
        key = (env_id, episode_index)
        if key in seen:
            raise ValueError(f"duplicate episode trace record for env {env_id} episode {episode_index}")
        seen.add(key)
        prior = first.get(env_id)
        if prior is None or episode_index < prior[0]:
            first[env_id] = (episode_index, row)

    expected_ids = set(range(expected_env_count))
    if set(first) != expected_ids:
        raise ValueError("episode trace environment ids do not match the expected roster")

    selected = [first[env_id][1] for env_id in range(expected_env_count)]
    counts = {
        "task_finish_reached": 0,
        "task_completed": 0,
        "no_fall_no_gap": 0,
        "fallen": 0,
        "fell_into_gap": 0,
        "out_of_bounds": 0,
        "collision": 0,
        "legacy_success": 0,
    }
    for row in selected:
        finish_reached = _flag(row, "task_finish_reached")
        fallen = _flag(row, "fallen")
        fell_into_gap = _flag(row, "fell_into_gap")
        completed = _flag(row, "task_completed")
        expected_completed = finish_reached and not fallen and not fell_into_gap
        if completed != expected_completed:
            raise ValueError("episode trace task_completed disagrees with the completion contract")
        counts["task_finish_reached"] += int(finish_reached)
        counts["task_completed"] += int(expected_completed)
        counts["no_fall_no_gap"] += int(not fallen and not fell_into_gap)
        counts["fallen"] += int(fallen)
        counts["fell_into_gap"] += int(fell_into_gap)
        counts["out_of_bounds"] += int(_flag(row, "out_of_bounds"))
        counts["collision"] += int(_flag(row, "collision"))
        counts["legacy_success"] += int(_flag(row, "success"))

    result = {
        "status": "completion-score-complete",
        "metric_contract": "finish_reached and not fallen and not fell_into_gap",
        "episode_selection": "lowest episode_index per env_id",
        "denominator": expected_env_count,
        **counts,
    }
    result["task_completion_rate"] = counts["task_completed"] / expected_env_count
    result["no_fall_no_gap_rate"] = counts["no_fall_no_gap"] / expected_env_count
    return result
