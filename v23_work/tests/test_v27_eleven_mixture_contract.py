"""Mutation caught: v27 must train the full eleven families including combined from the v26 student."""
from pathlib import Path
import importlib.util
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from go1_hier_sim.envs.task_curriculum import TERRAIN_FAMILIES
from go1_hier_sim.envs.eleven_roster_v27 import select_training_roster

ROOT = Path(__file__).resolve().parents[1]


def test_roster_is_all_eleven_and_rejects_narrower_plans():
    chosen = select_training_roster({'training_roster': 'eleven_with_combined'}, TERRAIN_FAMILIES)
    assert set(chosen) == set(TERRAIN_FAMILIES) and 'combined' in chosen
    for wrong in ('ten_except_combined', 'triple_refinement', None):
        try:
            select_training_roster({'training_roster': wrong}, TERRAIN_FAMILIES)
            raise AssertionError('accepted')
        except ValueError:
            pass
    try:
        select_training_roster({'training_roster': 'eleven_with_combined'}, TERRAIN_FAMILIES[:10])
        raise AssertionError('accepted registry without combined')
    except ValueError:
        pass


def test_config_driver_and_launcher_lineage():
    spec = importlib.util.spec_from_file_location(
        'v27_driver', ROOT / 'scripts/run_student_v27.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.select_training_roster is select_training_roster
    config = json.loads((ROOT / 'configs/curriculum_phases_eleven_mixture_v27.json').read_text())
    assert config['training_roster'] == 'eleven_with_combined'
    assert config['min_family_share'] == 0.06 and config['max_family_share'] == 0.1
    assert (config['num_envs'], config['max_stages'], config['warmup_first']) == (512, 1, 150)
    assert config['phases'][0]['iterations'] == 900
    assert config['init_checkpoint'].endswith(
        '2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt')
    assert config['gate']['completion_gate'] == 0.9 and config['gate']['min_family_episodes'] == 40
    assert config['require_deterministic_gate'] is True
    source = (ROOT / 'scripts/start_v27_eleven_mixture.sh').read_text()
    for expected in ('NPROC_PER_NODE=4', 'GO1_DDP_BACKEND=gloo',
                     'exec 9>reports/v7_sim.lock', 'flock -n 9', 'lp_phased_stop.flag',
                     'run_student_v27.py', '--max-stages 1', '2026-10-18_go1-v27-eleven-mixture',
                     'GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_CAP=.60 GO1_V23_TIRE_PEN=.5',
                     '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
                     '5da04ecda118796872f40d91e116f5b24499d05685af54355e51e8eea9c8b7dc'):
        assert expected in source
    assert 'GO1_V9_GAP_CONTACT=1' not in source
