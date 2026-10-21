"""v36 soft-dynamics approximation: soft8 = exact geometry; fall termination suppressed."""
import ast
import types
from pathlib import Path

import numpy as np
import torch

from go1_hier_sim.envs.geometry import build_tile

_CFG = Path('/config/Desktop/sim/source/go1_hier_sim/go1_hier_sim/envs/config.py')


def _task_termination():
    tree = ast.parse(_CFG.read_text(encoding='utf-8-sig'))
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == 'task_termination'][0]
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<task_termination>', 'exec'), ns)
    return ns['task_termination']


def test_combined_soft8_is_byte_identical_to_combined():
    a = build_tile('combined', 0.1, np.random.default_rng(2026), [])
    b = build_tile('combined_soft8', 0.1, np.random.default_rng(2026), [])
    assert np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    assert a.metadata['course_profile']['blocks'] == b.metadata['course_profile']['blocks']


def _env_with(mask):
    hier = types.SimpleNamespace()
    hier._soft_termination_mask = mask
    hier.get_metrics = lambda: {
        'fallen': torch.tensor([True, True]),
        'out_of_bounds': torch.tensor([False, False]),
        'success': torch.tensor([False, False]),
        'fell_into_gap': torch.tensor([True, False]),
    }
    return types.SimpleNamespace(hier=hier)


def test_fall_termination_suppressed_only_on_masked_envs():
    task_termination = _task_termination()
    out = task_termination(_env_with(torch.tensor([True, False])))
    assert bool(out[0]) is False and bool(out[1]) is True
    out2 = task_termination(_env_with(torch.tensor([False, False])))
    assert bool(out2[0]) is True and bool(out2[1]) is True


def test_no_mask_attribute_keeps_original_semantics():
    task_termination = _task_termination()
    env = _env_with(torch.tensor([False, False]))
    del env.hier._soft_termination_mask
    env.hier.__dict__.pop('_soft_termination_mask', None)
    out = task_termination(env)
    assert bool(out[0]) is True
