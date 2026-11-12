from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/go1_hier_sim"))

from go1_hier_sim.envs import long_gap_completion_v1333 as completion  # noqa: E402
from go1_hier_sim.envs.long_gap_completion_v1333 import (  # noqa: E402
    latch_finish_reached,
    score_first_episode_rows,
    task_completion_mask,
)


def row(env_id=0, episode_index=0, **overrides):
    value = {
        "env_id": env_id,
        "episode_index": episode_index,
        "task_finish_reached": True,
        "task_completed": True,
        "fallen": False,
        "fell_into_gap": False,
        "out_of_bounds": False,
        "collision": False,
        "success": False,
    }
    value.update(overrides)
    return value


def test_finish_region_latch_is_monotonic_until_episode_reset():
    reached = latch_finish_reached(
        torch.tensor([False, True]), torch.tensor([True, False])
    )
    assert reached.tolist() == [True, True]


def test_completion_requires_finish_and_both_no_fall_flags():
    result = task_completion_mask(
        torch.tensor([True, False, True, True]),
        torch.tensor([False, False, True, False]),
        torch.tensor([False, False, False, True]),
    )
    assert result.tolist() == [True, False, False, False]


def test_opt_in_strict_completion_termination_matches_the_completion_contract():
    terminate = getattr(completion, "strict_completion_termination_mask", None)
    assert callable(terminate), "missing opt-in strict-completion termination mask"
    result = terminate(
        torch.tensor([True, True, True, False]),
        torch.tensor([False, True, False, False]),
        torch.tensor([False, False, True, False]),
        enabled=True,
    )
    assert result.tolist() == [True, False, False, False]

    disabled = terminate(
        torch.tensor([True]), torch.tensor([False]), torch.tensor([False]), enabled=False,
    )
    assert disabled.tolist() == [False]


def test_stage1_termination_uses_strict_completion_in_opt_in_profile_only():
    terminate = getattr(completion, "stage1_task_termination_mask", None)
    assert callable(terminate), "missing stage-1 objective-aligned termination mask"
    fallen = torch.tensor([False, False, True, False])
    out_of_bounds = torch.tensor([False, False, False, True])
    fell_into_gap = torch.tensor([False, False, False, False])
    success = torch.tensor([True, True, False, False])
    finish = torch.tensor([False, True, True, True])

    strict = terminate(
        fallen, out_of_bounds, success, fell_into_gap, finish,
        strict_completion_enabled=True,
    )
    assert strict.tolist() == [False, True, True, True]

    legacy = terminate(
        fallen, out_of_bounds, success, fell_into_gap, finish,
        strict_completion_enabled=False,
    )
    assert legacy.tolist() == [True, True, True, True]


def test_finished_episode_counts_even_if_out_of_bounds_or_collision_afterward():
    report = score_first_episode_rows(
        [row(out_of_bounds=True, collision=True, success=False)], expected_env_count=1
    )
    assert report["denominator"] == 1
    assert report["task_completed"] == 1
    assert report["task_completion_rate"] == 1.0
    assert report["out_of_bounds"] == 1
    assert report["collision"] == 1
    assert report["legacy_success"] == 0


def test_upright_episode_without_reaching_finish_is_not_complete():
    report = score_first_episode_rows(
        [row(task_finish_reached=False, task_completed=False)], expected_env_count=1
    )
    assert report["task_completed"] == 0
    assert report["task_completion_rate"] == 0.0
    assert report["no_fall_no_gap"] == 1


@pytest.mark.parametrize(
    "flags",
    [
        {"fallen": True, "task_completed": False},
        {"fell_into_gap": True, "task_completed": False},
    ],
)
def test_fall_or_gap_fall_prevents_completion_even_after_finish(flags):
    report = score_first_episode_rows([row(**flags)], expected_env_count=1)
    assert report["task_completed"] == 0


def test_scorer_selects_lowest_episode_index_per_environment():
    report = score_first_episode_rows(
        [
            row(env_id=0, episode_index=1, task_finish_reached=True, task_completed=True),
            row(env_id=0, episode_index=0, task_finish_reached=False, task_completed=False),
        ],
        expected_env_count=1,
    )
    assert report["denominator"] == 1
    assert report["task_completed"] == 0
    assert report["no_fall_no_gap"] == 1


def test_scorer_rejects_duplicate_first_episode_records():
    with pytest.raises(ValueError, match="duplicate"):
        score_first_episode_rows(
            [row(env_id=0, episode_index=0), row(env_id=0, episode_index=0)],
            expected_env_count=1,
        )


def test_scorer_rejects_missing_environments_and_inconsistent_recorded_completion():
    with pytest.raises(ValueError, match="environment ids"):
        score_first_episode_rows([row(env_id=1)], expected_env_count=2)
    with pytest.raises(ValueError, match="task_completed"):
        score_first_episode_rows(
            [row(task_finish_reached=False, task_completed=True)], expected_env_count=1
        )


def test_runtime_and_training_report_use_the_separate_completion_contract():
    runtime_path = ROOT / "source/go1_hier_sim/go1_hier_sim/envs/runtime.py"
    config_path = ROOT / "source/go1_hier_sim/go1_hier_sim/envs/config.py"
    cli_path = ROOT / "source/go1_hier_sim/go1_hier_sim/cli.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    config = config_path.read_text(encoding="utf-8")
    cli = cli_path.read_text(encoding="utf-8")
    assert "latch_finish_reached(self._task_finish_reached, reached)" in runtime
    assert '"task_completed": task_completion_mask(' in runtime
    assert '"task_completion_contract": "task_finish_reached and not fallen and not fell_into_gap"' in cli
    assert '"completion_rate": all_success / all_episodes' in cli
    assert "stage1_task_termination_mask(" in config
    assert '"task_completed" in env.cfg.hier_stage1_rewards' in config


def test_episode_trace_requires_completion_flags_to_match_terminal_metrics():
    from go1_hier_sim.envs.episode_trace_v12 import merge_episode_trace

    terminal = row()
    terminal["steps"] = 1
    terminal["command_speed"] = 0.5
    terminal["course_x"] = 0.5
    terminal["course_y"] = 0.0
    terminal["env"] = 0
    episode = {key: value for key, value in terminal.items() if key != "env"}
    assert merge_episode_trace([0], [episode], [terminal])[0]["task_completed"] is True

    missing_metric = dict(episode)
    del missing_metric["task_completed"]
    with pytest.raises(ValueError, match="episode trace missing terminal data"):
        merge_episode_trace([0], [missing_metric], [terminal])
