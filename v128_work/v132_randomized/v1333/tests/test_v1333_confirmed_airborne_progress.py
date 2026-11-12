from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "go1_hier_sim"))

MODULE_NAME = "go1_hier_sim.envs.confirmed_airborne_progress_v1333"


def _reward_fn():
    assert importlib.util.find_spec(MODULE_NAME) is not None, (
        "missing confirmed per-gap airborne-progress helper"
    )
    module = importlib.import_module(MODULE_NAME)
    function = getattr(module, "confirmed_airborne_forward_distance_reward", None)
    assert callable(function), "missing confirmed_airborne_forward_distance_reward API"
    return function


def _step(function, *, pending, paid, x=0.05, dx=0.04, clear=False,
          air_time=(0.02, 0.02, 0.02, 0.02), intrusion=0.0, failed=False,
          bounds=None, max_credit_m=0.45):
    if bounds is None:
        bounds = torch.tensor([[[0.0, 0.34, 0.80]]], dtype=torch.float32)
    return function(
        current_air_time=torch.tensor([air_time], dtype=torch.float32),
        forward_delta=torch.tensor([dx], dtype=torch.float32),
        local_x=torch.tensor([x], dtype=torch.float32),
        gap_bounds=bounds,
        window_open=torch.tensor([True]),
        gap_mask=torch.tensor([True]),
        intrusion=torch.tensor([intrusion], dtype=torch.float32),
        hard_failure=torch.tensor([failed]),
        cleared_gaps=torch.full(bounds.shape[:2], clear, dtype=torch.bool),
        pending_distance=pending,
        paid_distance=paid,
        approach_distance=0.35,
        landing_distance=0.15,
        short_width_max=0.265,
        max_gap_width=0.48,
        max_credit_m=max_credit_m,
        control_dt=0.02,
    )


def test_airborne_distance_is_escrowed_then_committed_once_after_clearance():
    reward_fn = _reward_fn()
    pending = torch.zeros((1, 1), dtype=torch.float32)
    paid = torch.zeros_like(pending)

    reward, pending, paid = _step(reward_fn, pending=pending, paid=paid)
    assert reward.item() == pytest.approx(0.0)
    assert pending[0, 0].item() == pytest.approx(0.04)
    assert paid[0, 0].item() == pytest.approx(0.0)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.50, dx=0.0, clear=True,
    )
    assert reward.item() == pytest.approx(2.0)  # 0.04 m / 0.02 s
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.04)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.50, dx=0.10, clear=True,
    )
    assert reward.item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.04)


def test_stable_clearance_confirms_credit_even_if_failure_is_also_latched():
    reward_fn = _reward_fn()
    pending = torch.tensor([[0.04]], dtype=torch.float32)
    paid = torch.zeros_like(pending)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.50, dx=0.0,
        clear=True, failed=True,
    )
    assert reward.item() == pytest.approx(2.0)
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.04)


def test_near_side_grounded_touchdown_cancels_pending_credit():
    reward_fn = _reward_fn()
    pending = torch.zeros((1, 1), dtype=torch.float32)
    paid = torch.zeros_like(pending)
    _, pending, paid = _step(reward_fn, pending=pending, paid=paid)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.10, dx=0.0,
        air_time=(0.0, 0.0, 0.0, 0.0),
    )
    assert reward.item() == pytest.approx(0.0)
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.0)


def test_hard_failure_and_intrusion_cancel_unconfirmed_credit():
    reward_fn = _reward_fn()
    pending = torch.zeros((1, 1), dtype=torch.float32)
    paid = torch.zeros_like(pending)
    _, pending, paid = _step(reward_fn, pending=pending, paid=paid)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.10, dx=0.0, failed=True,
    )
    assert reward.item() == pytest.approx(0.0)
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.0)

    pending = torch.tensor([[0.04]], dtype=torch.float32)
    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.10, dx=0.0, intrusion=0.01,
    )
    assert reward.item() == pytest.approx(0.0)
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.0)


def test_credit_is_capped_by_the_smaller_of_gap_width_and_configured_cap():
    reward_fn = _reward_fn()
    pending = torch.zeros((1, 1), dtype=torch.float32)
    paid = torch.zeros_like(pending)
    bounds = torch.tensor([[[0.0, 0.34, 0.80]]], dtype=torch.float32)

    for x in (0.04, 0.08, 0.12):
        _, pending, paid = _step(
            reward_fn, pending=pending, paid=paid, x=x, dx=0.30,
            bounds=bounds, max_credit_m=0.45,
        )
    assert pending[0, 0].item() == pytest.approx(0.34)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, x=0.50, dx=0.0,
        clear=True, bounds=bounds, max_credit_m=0.45,
    )
    assert reward.item() == pytest.approx(17.0)  # 0.34 m / 0.02 s
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.34)


def test_unused_nonfinite_gap_slots_are_ignored_without_nan_credit():
    reward_fn = _reward_fn()
    pending = torch.zeros((1, 1), dtype=torch.float32)
    paid = torch.zeros_like(pending)
    bounds = torch.full((1, 1, 3), float("inf"), dtype=torch.float32)

    reward, pending, paid = _step(
        reward_fn, pending=pending, paid=paid, bounds=bounds,
    )
    assert reward.item() == pytest.approx(0.0)
    assert torch.isfinite(pending).all()
    assert pending[0, 0].item() == pytest.approx(0.0)
    assert paid[0, 0].item() == pytest.approx(0.0)


def test_runtime_wires_a05_escrow_to_stable_clearance_and_resets_per_env_state():
    runtime_path = ROOT / "source/go1_hier_sim/go1_hier_sim/envs/runtime.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    assert "from .confirmed_airborne_progress_v1333 import" in runtime
    assert 'confirmed_cfg = cfg.get("airborne_forward_progress_confirmed")' in runtime
    assert "cleared_gaps=self._v132_paid" in runtime
    assert "pending_distance=self._airborne_forward_progress_pending" in runtime
    assert 'terms["airborne_forward_progress_confirmed"] = confirmed_reward' in runtime
    assert "self._airborne_forward_progress_pending[ids] = 0.0" in runtime
    assert "self._airborne_forward_progress_paid[ids] = 0.0" in runtime
