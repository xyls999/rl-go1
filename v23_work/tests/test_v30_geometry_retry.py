"""Reproduce the exact v30 crash: pure-combined 512-slot plan at seed 2026/2028 must build."""
import json
from pathlib import Path

import numpy as np
import pytest

from go1_hier_sim.envs.geometry import build_scene

PLAN = Path('/config/Desktop/sim/reports/lp_plan_2026-10-19_go1-v30-combined-teacher-s01-p1-d1.json')


@pytest.mark.parametrize('seed', [2026, 2028])
def test_pure_combined_plan_builds_and_is_reproducible(seed):
    tasks = json.loads(PLAN.read_text())
    pairs = [(f, d) for f, d in zip(tasks['families'], tasks['difficulty'])]
    assert len(pairs) == 512 and all(f == 'combined' for f, _ in pairs)
    first_mesh, first_origins = build_scene(512, 'lp_acrl', True, seed, 0.0, 0.0, 0.6, tasks=pairs)
    second_mesh, second_origins = build_scene(512, 'lp_acrl', True, seed, 0.0, 0.0, 0.6, tasks=pairs)
    assert first_mesh.vertices.shape == second_mesh.vertices.shape
    assert np.allclose(first_mesh.vertices, second_mesh.vertices)
    assert np.allclose(first_origins, second_origins)
