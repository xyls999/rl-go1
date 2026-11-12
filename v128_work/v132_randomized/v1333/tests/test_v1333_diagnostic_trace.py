import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from go1_hier_sim.envs.diagnostic_trace_v1333 import format_gap_actor_trace


FEATURE_POSITIONS = (49, 64, 161, 176)


def test_trace_extracts_the_four_actor_visible_gap_features_and_action():
    observation = torch.zeros((2, 1073))
    observation[1, list(FEATURE_POSITIONS)] = torch.tensor([1.0, 0.25, 0.5, 0.75])
    action = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    trace = format_gap_actor_trace(observation, action, env_id=1, step=25)
    assert trace == {
        "env_id": 1,
        "step": 25,
        "gap_actor_features": {
            "visible": 1.0,
            "near_lip_normalized": 0.25,
            "width_normalized": 0.5,
            "far_edge_normalized": 0.75,
        },
        "action": [float(value) for value in range(12, 24)],
    }


def test_trace_rejects_wrong_interface_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="1073"):
        format_gap_actor_trace(torch.zeros((1, 1072)), torch.zeros((1, 12)))
    with pytest.raises(ValueError, match="12"):
        format_gap_actor_trace(torch.zeros((1, 1073)), torch.zeros((1, 11)))
    bad = torch.zeros((1, 1073))
    bad[0, FEATURE_POSITIONS[0]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        format_gap_actor_trace(bad, torch.zeros((1, 12)))


def test_evaluation_debug_motion_wires_the_policy_trace():
    cli = (ROOT / "source" / "go1_hier_sim" / "go1_hier_sim" / "cli.py").read_text()
    assert "format_gap_actor_trace" in cli
    assert "GAP_POLICY_TRACE" in cli
