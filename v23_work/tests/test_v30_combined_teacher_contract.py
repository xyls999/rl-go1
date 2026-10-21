"""Mutation caught: v30 must be a pure combined specialization warm-started from the v26 student."""
from pathlib import Path
import importlib.util
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from go1_hier_sim.envs.task_curriculum import TERRAIN_FAMILIES
from go1_hier_sim.envs.combined_roster_v30 import select_training_roster

ROOT = Path(__file__).resolve().parents[1]


def test_roster_is_combined_only_and_rejects_other_plans():
    assert select_training_roster({'training_roster': 'single:combined'}, TERRAIN_FAMILIES) == ('combined',)
    for wrong in ('ten_except_combined', 'triple_refinement', 'eleven_with_combined', None):
        try:
            select_training_roster({'training_roster': wrong}, TERRAIN_FAMILIES)
            raise AssertionError('accepted')
        except ValueError:
            pass


def test_config_and_launcher_lineage():
    spec = importlib.util.spec_from_file_location(
        'v30_driver', ROOT / 'scripts/run_student_v30.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.select_training_roster is select_training_roster
    config = json.loads((ROOT / 'configs/curriculum_phases_combined_v30.json').read_text())
    assert config['training_roster'] == 'single:combined'
    assert (config['num_envs'], config['max_stages'], config['warmup_first']) == (512, 1, 150)
    assert config['phases'][0]['iterations'] == 900
    assert config['init_checkpoint'].endswith(
        '2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt')
    assert config['require_deterministic_gate'] is True
    source = (ROOT / 'scripts/start_v30_combined_teacher.sh').read_text()
    for expected in ('NPROC_PER_NODE=4', 'GO1_DDP_BACKEND=gloo',
                     'exec 9>reports/v7_sim.lock', 'flock -n 9',
                     'run_student_v30.py', '2026-10-19_go1-v30-combined-teacher',
                     'GO1_V23_TIRE_BRAKE_FAMILIES=combined',
                     '6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674'):
        assert expected in source
    assert 'GO1_V29_TEACHER_DISTILL=1' not in source
    assert 'GO1_V9_GAP_CONTACT=1' not in source
