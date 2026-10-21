"""v32 soft-curriculum family: default terrain byte-identical, combined_soft widens pads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from go1_hier_sim.envs.geometry import build_tile

OLD = Path('/config/Desktop/sim/reports/v29_backup_20261018/geometry.py')


def _old_module():
    spec = importlib.util.spec_from_file_location('geometry_old_v32', OLD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gaps(mesh):
    return [b for b in mesh.metadata['course_profile']['blocks'] if b['kind'] == 'gap']


@pytest.mark.skipif(not OLD.exists(), reason='v30b geometry backup missing')
def test_default_geometry_is_byte_identical():
    old = _old_module()
    new_mesh = build_tile('combined', 0.1, np.random.default_rng(2026), [])
    old_mesh = old.build_tile('combined', 0.1, np.random.default_rng(2026), [])
    assert np.array_equal(np.asarray(new_mesh.vertices), np.asarray(old_mesh.vertices))
    assert _gaps(new_mesh) == _gaps(old_mesh)


def test_combined_soft_widens_only_the_three_opening_pads():
    base = _gaps(build_tile('combined', 0.1, np.random.default_rng(2026), []))
    soft = _gaps(build_tile('combined_soft', 0.1, np.random.default_rng(2026), []))
    for i in range(3):
        pad_b = base[i + 1]['start'] - base[i]['void_end']
        pad_s = soft[i + 1]['start'] - soft[i]['void_end']
        assert pad_s - pad_b == pytest.approx(0.15)
    late_b = base[4]['start'] - base[3]['void_end']
    late_s = soft[4]['start'] - soft[3]['void_end']
    assert late_s - late_b == pytest.approx(0.0, abs=1e-9)


def test_soft2_and_soft3_shrink_only_the_opening_gap_widths():
    base = _gaps(build_tile('combined', 0.1, np.random.default_rng(2026), []))
    w2 = _gaps(build_tile('combined_soft2', 0.1, np.random.default_rng(2026), []))
    w3 = _gaps(build_tile('combined_soft3', 0.1, np.random.default_rng(2026), []))
    for i in range(3):
        assert w2[i]['width'] == pytest.approx(base[i]['width'] * 0.25, rel=0.03)
        assert w3[i]['width'] == pytest.approx(base[i]['width'] * 0.6, rel=0.03)
    assert w2[3]['width'] == pytest.approx(base[3]['width'], rel=0.03)
    assert w3[3]['width'] == pytest.approx(base[3]['width'], rel=0.03)
    for i in range(4):
        assert w2[i]['void_end'] - w2[i]['start'] == pytest.approx(w2[i]['width'] + 0.05, abs=1e-6)
