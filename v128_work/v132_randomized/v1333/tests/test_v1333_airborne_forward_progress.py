from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

from go1_hier_sim.envs.airborne_forward_progress_v1333 import (  # noqa: E402
    airborne_forward_distance_reward,
)
from go1_hier_sim.reward_config import (  # noqa: E402
    OPTIONAL_REWARD_FIELDS,
    REWARD_FIELDS,
    load_stage1_rewards,
)

A03_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a03.toml"
A04_CONFIG = ROOT / "configs" / "rewards_stage1_gaps_v1333_longgap_a04.toml"


def _step(
    *,
    x: float,
    dx: float,
    credit: torch.Tensor | None = None,
    air_time: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02),
    window: bool = True,
    gap: bool = True,
    intrusion: float = 0.0,
    failed: bool = False,
    cleared: bool = False,
    bounds: torch.Tensor | None = None,
    max_credit_m: float = 0.45,
):
    if bounds is None:
        bounds = torch.tensor([[[0.0, 0.34, 0.80]]], dtype=torch.float32)
    if credit is None:
        credit = torch.zeros(bounds.shape[:2], dtype=torch.float32)
    return airborne_forward_distance_reward(
        current_air_time=torch.tensor([air_time], dtype=torch.float32),
        forward_delta=torch.tensor([dx], dtype=torch.float32),
        local_x=torch.tensor([x], dtype=torch.float32),
        gap_bounds=bounds,
        window_open=torch.tensor([window]),
        gap_mask=torch.tensor([gap]),
        intrusion=torch.tensor([intrusion], dtype=torch.float32),
        hard_failure=torch.tensor([failed]),
        cleared_gaps=torch.full(bounds.shape[:2], cleared, dtype=torch.bool),
        credited_distance=credit,
        approach_distance=0.35,
        landing_distance=0.15,
        short_width_max=0.265,
        max_gap_width=0.48,
        max_credit_m=max_credit_m,
        control_dt=0.02,
    )


def test_airborne_forward_reward_is_distance_scaled_and_requires_all_gates():
    reward, credit = _step(x=0.05, dx=0.04)
    assert reward.item() == pytest.approx(2.0)  # 0.04 m / 0.02 s
    assert credit[0, 0].item() == pytest.approx(0.04)

    for kwargs in (
        {"air_time": (0.02, 0.02, 0.02, 0.0)},
        {"window": False},
        {"gap": False},
        {"intrusion": 0.01},
        {"failed": True},
        {"dx": -0.04},
        {"x": 0.35},  # beyond the far lip: landing flight is not distance credit
    ):
        sample = {"x": 0.05, "dx": 0.04}
        sample.update(kwargs)
        reward, _ = _step(**sample)
        assert reward.item() == pytest.approx(0.0)


def test_airborne_forward_reward_is_capped_per_gap_and_can_pay_a_second_gap():
    bounds = torch.tensor(
        [[[0.0, 0.30, 0.60], [0.80, 1.10, 1.40]]], dtype=torch.float32
    )
    credit = torch.zeros((1, 2), dtype=torch.float32)
    total_distance_credit = 0.0

    for x in (0.05, 0.15, 0.25, 0.30):
        reward, credit = _step(x=x, dx=0.10, credit=credit, bounds=bounds)
        total_distance_credit += reward.item() * 0.02
    assert total_distance_credit == pytest.approx(0.30)
    assert credit[0, 0].item() == pytest.approx(0.30)

    # Repeated forward motion in the same window cannot exceed its physical width.
    reward, credit = _step(x=0.10, dx=0.10, credit=credit, bounds=bounds)
    assert reward.item() == pytest.approx(0.0)

    reward, credit = _step(x=0.90, dx=0.10, credit=credit, bounds=bounds)
    assert reward.item() == pytest.approx(5.0)
    assert credit[0, 1].item() == pytest.approx(0.10)


def test_intrusion_or_hard_failure_refunds_unconfirmed_credit_once():
    prior = torch.tensor([[0.12]], dtype=torch.float32)
    reward, credit = _step(x=0.10, dx=0.02, credit=prior, intrusion=0.01)
    assert reward.item() == pytest.approx(-6.0)  # -0.12 m / 0.02 s
    assert credit[0, 0].item() == pytest.approx(0.0)

    reward, credit = _step(x=0.10, dx=0.02, credit=credit, failed=True)
    assert reward.item() == pytest.approx(0.0)
    assert credit[0, 0].item() == pytest.approx(0.0)


def test_stable_clearance_keeps_earned_distance_if_failure_flag_is_also_set():
    prior = torch.tensor([[0.12]], dtype=torch.float32)
    reward, credit = _step(x=0.34, dx=0.02, credit=prior, failed=True, cleared=True)
    assert reward.item() == pytest.approx(0.0)
    assert credit[0, 0].item() == pytest.approx(0.12)


def test_a04_adds_only_the_bounded_airborne_distance_term_to_a03():
    assert A04_CONFIG.is_file(), "missing isolated long-gap a04 reward profile"
    assert "airborne_forward_progress" not in REWARD_FIELDS
    assert OPTIONAL_REWARD_FIELDS["airborne_forward_progress"] == {
        "weight", "max_credit_m"
    }

    previous = load_stage1_rewards(A03_CONFIG)
    updated = load_stage1_rewards(A04_CONFIG)
    assert set(updated) == set(previous) | {"airborne_forward_progress"}
    for term in previous:
        assert updated[term] == previous[term]
    assert updated["airborne_forward_progress"] == {
        "weight": 20.0,
        "max_credit_m": 0.45,
    }
