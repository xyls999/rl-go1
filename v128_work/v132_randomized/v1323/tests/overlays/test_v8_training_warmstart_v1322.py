"""V8 warm-start contracts under explicit legacy and production PPO configs."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go1_hier_sim.cli import _ppo_config, _ppo_values, _load_warm_start_weights, _warm_start_optimizer
from go1_hier_sim.models import HierActorCritic
from go1_hier_sim.hier_v2.map_attention_v3 import HierActorCriticV3
from go1_hier_sim.contracts import LOCO_OBS_DIM


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _policy(cls):
    obs = {"policy": torch.zeros(2, LOCO_OBS_DIM),
           "critic": torch.zeros(2, LOCO_OBS_DIM + 7)}
    return cls(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, "locomotion")


def test_policy_selection_is_versioned_and_refuses_stage2(monkeypatch):
    args = SimpleNamespace(initialize_from="v7b.pt", leap_base_checkpoint=None, seed=1,
                           device="cpu", iterations=2, run_name="v8", reward_contract="stage1")
    monkeypatch.delenv("GO1_POLICY_VARIANT", raising=False)
    assert _ppo_config("stage1", args)["policy"]["class_name"] == "HierActorCritic"
    monkeypatch.setenv("GO1_POLICY_VARIANT", "map_attn_v3")
    assert _ppo_config("stage1", args)["policy"]["class_name"] == "HierActorCriticV3"
    with pytest.raises(ValueError, match="stage1"):
        _ppo_config("stage2", args)
    monkeypatch.setenv("GO1_POLICY_VARIANT", "typo")
    with pytest.raises(ValueError, match="variant"):
        _ppo_config("stage1", args)


def test_exact_v7b_structure_loads_except_new_adapter_and_exploration():
    torch.manual_seed(23)
    legacy = _policy(HierActorCritic)
    updated = _policy(HierActorCriticV3)
    saved = legacy.state_dict()
    _load_warm_start_weights(updated, dict(saved), "map_attn_v3")
    after = updated.state_dict()
    for name, value in saved.items():
        if name != "log_std":
            torch.testing.assert_close(after[name], value)
    broken = dict(saved)
    broken["actor.gru.weight_ih_l0"] = broken["actor.gru.weight_ih_l0"][:, :-1]
    with pytest.raises((RuntimeError, ValueError)):
        _load_warm_start_weights(_policy(HierActorCriticV3), broken, "map_attn_v3")
    broken = dict(saved)
    broken["unexpected.privileged_input"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        _load_warm_start_weights(_policy(HierActorCriticV3), broken, "map_attn_v3")


def test_new_adapter_gets_independent_bounded_rate(monkeypatch):
    monkeypatch.setenv("GO1_PPO_CONFIG_PATH", str(CONFIGS / "ppo.json"))
    policy = _policy(HierActorCriticV3)
    optimizer = _warm_start_optimizer(policy, _ppo_values(), "map_attn_v3")
    groups = optimizer.param_groups
    assert [group["lr"] for group in groups] == [1e-7, 3e-5, 3e-6]
    assigned = [id(p) for group in groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(policy.parameters()))


def test_production_gap_ppo_adapter_rate_and_assignment(monkeypatch):
    monkeypatch.setenv("GO1_PPO_CONFIG_PATH", str(CONFIGS / "ppo_gaps_long.json"))
    policy = _policy(HierActorCriticV3)
    optimizer = _warm_start_optimizer(policy, _ppo_values(), "map_attn_v3")
    groups = optimizer.param_groups
    assert [group["lr"] for group in groups] == [2e-6, 3e-5, 2e-5]
    assigned = [id(p) for group in groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(policy.parameters()))
