from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

from go1_hier_sim.reward_config import (  # noqa: E402
    OPTIONAL_REWARD_FIELDS,
    load_stage1_rewards,
)

A03_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a03.toml"
A04_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a04.toml"
A05_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a05.toml"


def test_a05_preserves_a03_reward_tables_and_adds_only_objective_aligned_terms():
    previous = load_stage1_rewards(A03_CONFIG)
    a04 = load_stage1_rewards(A04_CONFIG)
    updated = load_stage1_rewards(A05_CONFIG)

    assert set(updated) == set(previous) | {
        "airborne_forward_progress_confirmed", "task_completed",
    }
    for name, table in previous.items():
        assert updated[name] == table
    assert "airborne_forward_progress" in a04
    assert "airborne_forward_progress" not in updated
    assert updated["airborne_forward_progress_confirmed"] == {
        "weight": 20.0,
        "max_credit_m": 0.45,
    }
    assert updated["task_completed"] == {"weight": 200.0}
    assert OPTIONAL_REWARD_FIELDS["airborne_forward_progress_confirmed"] == {
        "weight", "max_credit_m",
    }
    assert OPTIONAL_REWARD_FIELDS["task_completed"] == {"weight"}


def test_strict_completion_term_cannot_be_enabled_without_positive_reward(tmp_path):
    content = A05_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "disabled_task_completed.toml"
    path.write_text(content.replace("[task_completed]\nweight = 200.0", "[task_completed]\nweight = 0.0"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="task_completed.weight must be positive"):
        load_stage1_rewards(path)


def test_confirmed_airborne_credit_requires_positive_cap(tmp_path):
    content = A05_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "zero_credit_cap.toml"
    path.write_text(
        content.replace("[airborne_forward_progress_confirmed]\nweight = 20.0\nmax_credit_m = 0.45",
                        "[airborne_forward_progress_confirmed]\nweight = 20.0\nmax_credit_m = 0.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="airborne_forward_progress_confirmed.max_credit_m"):
        load_stage1_rewards(path)
