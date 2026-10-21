"""v47 privileged actor: the actor sees exact terrain features under the flag."""
import ast
import os
import types
from pathlib import Path

import torch

_CFG = Path('/config/Desktop/sim/source/go1_hier_sim/go1_hier_sim/envs/config.py')


def _fn():
    tree = ast.parse(_CFG.read_text(encoding='utf-8-sig'))
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'actor_observation'][0]
    ns = {}
    # the function references module-level _SENSOR_REALITY and SensorReality
    src = ast.get_source_segment(_CFG.read_text(encoding='utf-8-sig'), node)
    return src


def _env():
    hier = types.SimpleNamespace()
    hier.refresh = lambda: None
    hier.locomotion_obs = torch.full((2, 8), 1.0)
    hier.teacher_locomotion_obs = torch.full((2, 8), 7.0)
    hier._episode_id = torch.zeros(2, dtype=torch.long)
    return types.SimpleNamespace(hier=hier, cfg=types.SimpleNamespace(hier_stage='locomotion'),
                                 num_envs=2, device='cpu')


def test_privileged_actor_returns_teacher_observations(monkeypatch):
    src = _fn()
    ns = {'torch': torch}
    # stub the reality-gap helper used by the camera path
    class _S:
        def __init__(self, *a, **k):
            pass

        def degrade(self, obs, ep):
            return obs

    ns['SensorReality'] = _S
    ns['_SENSOR_REALITY'] = {}
    exec(src, ns)
    fn = ns['actor_observation']
    monkeypatch.setenv('GO1_PRIVILEGED_ACTOR', '1')
    out = fn(_env())
    assert float(out[0, 0]) == 7.0
    monkeypatch.delenv('GO1_PRIVILEGED_ACTOR', raising=False)
    out2 = fn(_env())
    assert float(out2[0, 0]) == 1.0
