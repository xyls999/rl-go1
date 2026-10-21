"""v38 credit: narrow opening voids become creditable only under the override."""
import torch

from go1_hier_sim.envs.prelaunch_v19 import PrelaunchGradientState


def _state():
    bounds = torch.full((1, 24, 3), float('nan'), dtype=torch.float32)
    bounds[0, 0] = torch.tensor([1.000, 1.082, 1.130])
    bounds[0, 1] = torch.tensor([2.000, 2.400, 2.450])
    mask = torch.tensor([True])
    return bounds, mask


def test_default_excludes_narrow_void(monkeypatch):
    monkeypatch.delenv('GO1_V19_MIN_WIDTH', raising=False)
    bounds, mask = _state()
    st = PrelaunchGradientState(bounds, mask, scale=1.0)
    assert bool(st.valid[0, 0]) is False
    assert bool(st.valid[0, 1]) is True


def test_override_credits_narrow_void(monkeypatch):
    monkeypatch.setenv('GO1_V19_MIN_WIDTH', '0.05')
    bounds, mask = _state()
    st = PrelaunchGradientState(bounds, mask, scale=1.0)
    assert bool(st.valid[0, 0]) is True
    assert bool(st.valid[0, 1]) is True
