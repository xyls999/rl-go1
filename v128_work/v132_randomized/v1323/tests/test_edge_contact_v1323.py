"""CPU-only V132.3 real-gap edge contact penalty contracts."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/go1_hier_sim/go1_hier_sim'
BASE = Path(__file__).resolve().parents[2] / 'v1322'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EdgeContactReward(unittest.TestCase):
    def setUp(self):
        self.step = load_file('edge_contact_v1323', SOURCE / 'envs/edge_contact_v1323.py').edge_contact_step
        self.bounds = torch.full((2, 24, 3), float('nan'))
        self.bounds[:, 0] = torch.tensor([1.50, 1.67, 2.42])
        self.foot_x = torch.tensor([[1.46, 1.71, 1.30, 2.00], [1.46, 1.71, 1.30, 2.00]])
        self.foot_z = torch.zeros((2, 4))
        self.contacts = torch.ones((2, 4), dtype=torch.bool)
        self.failed = torch.zeros(2, dtype=torch.bool)
        self.entrance = torch.tensor([True, False])

    def score(self):
        return self.step(self.bounds, self.foot_x, self.foot_z, self.contacts,
                         self.failed, self.entrance)

    def test_counts_only_supported_pads_within_five_centimeters_of_real_edges(self):
        torch.testing.assert_close(self.score(), torch.tensor([2., 0.]))
        self.foot_x[0] = torch.tensor([1.45, 1.72, 1.30, 2.00])
        torch.testing.assert_close(self.score(), torch.tensor([2., 0.]))
        self.foot_x[0] = torch.tensor([1.44, 1.73, 1.30, 2.00])
        torch.testing.assert_close(self.score(), torch.tensor([0., 0.]))

    def test_never_penalizes_airborne_floor_invalid_or_failed_contacts(self):
        for change in (
            ('contacts', torch.zeros((2, 4), dtype=torch.bool)),
            ('foot_z', torch.full((2, 4), -.10)),
            ('failed', torch.ones(2, dtype=torch.bool)),
            ('bounds', torch.full((2, 24, 3), float('nan'))),
            ('foot_x', torch.tensor([[1.55, 1.65, 1.30, 2.00]] * 2)),
        ):
            with self.subTest(change=change[0]):
                original = getattr(self, change[0]); setattr(self, change[0], change[1])
                torch.testing.assert_close(self.score(), torch.zeros(2))
                setattr(self, change[0], original)
        self.contacts[0, :2] = False
        torch.testing.assert_close(self.score(), torch.zeros(2))
        self.entrance[0] = False
        torch.testing.assert_close(self.score(), torch.zeros(2))

    def test_four_contacts_never_count_more_than_four_near_adjacent_gaps(self):
        self.bounds[0, 1] = torch.tensor([2.42, 2.59, 3.24])
        self.foot_x[0] = torch.tensor([1.46, 1.71, 2.38, 2.63])
        torch.testing.assert_close(self.score(), torch.tensor([4., 0.]))
        self.assertTrue(torch.isfinite(self.score()).all())

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            self.step(self.bounds, self.foot_x[:, :3], self.foot_z, self.contacts,
                      self.failed, self.entrance)


class EdgeContactIntegration(unittest.TestCase):
    def test_reward_loader_accepts_negative_optional_term_and_rejects_positive_weight(self):
        loader = load_file('reward_config_v1323', SOURCE / 'reward_config.py')
        with tempfile.TemporaryDirectory() as directory:
            text = (BASE / 'configs/rewards_stage1_gaps_v131_landing_stability.toml').read_text()
            for weight in (-1.0, 0.0, 1.0):
                path = Path(directory) / 'rewards.toml'
                path.write_text(text + f'\n[edge_contact]\nweight = {weight}\n')
                if weight > 0:
                    with self.assertRaisesRegex(ValueError, 'edge_contact'):
                        loader.load_stage1_rewards(path)
                else:
                    self.assertEqual(loader.load_stage1_rewards(path)['edge_contact']['weight'], weight)

    def test_runtime_edge_term_consumes_real_bounds_and_true_contact_only(self):
        helper = load_file('edge_contact_v1323', SOURCE / 'envs/edge_contact_v1323.py')
        text = (SOURCE / 'envs/runtime.py').read_text(encoding='utf-8')
        tree = ast.parse(text)
        reward = next(child for node in tree.body if isinstance(node, ast.ClassDef)
                      for child in node.body if isinstance(child, ast.FunctionDef)
                      and child.name == '_build_locomotion_reward_terms')
        calls = [node for node in ast.walk(reward) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'edge_contact_step']
        self.assertEqual(len(calls), 1)
        bounds = torch.full((2, 24, 3), float('nan'))
        bounds[:, 0] = torch.tensor([1.50, 1.67, 2.42])
        context = {'edge_contact_step': helper.edge_contact_step,
                   'self': SimpleNamespace(_v132_gap_bounds=bounds,
                       _v132_entrance_mask=torch.tensor([True, False]),
                       _fallen=torch.zeros(2, dtype=torch.bool),
                       _collision=torch.zeros(2, dtype=torch.bool),
                       _fell_into_gap=torch.zeros(2, dtype=torch.bool),
                       _out_of_bounds=torch.zeros(2, dtype=torch.bool)),
                   'entrance_foot_x': torch.tensor([[1.46, 1.71, 1.30, 2.00]] * 2),
                   'entrance_foot_z': torch.zeros((2, 4)),
                   'current_contacts': torch.ones((2, 4), dtype=torch.bool)}
        actual = eval(compile(ast.fix_missing_locations(ast.Expression(calls[0])), '<reward-call>', 'eval'), context)
        torch.testing.assert_close(actual, torch.tensor([2., 0.]))
        context['self']._out_of_bounds[0] = True
        zero = eval(compile(ast.fix_missing_locations(ast.Expression(calls[0])), '<reward-call>', 'eval'), context)
        torch.testing.assert_close(zero, torch.zeros(2))
        self.assertIn('edge_contact', {name for node in ast.walk(reward) if isinstance(node, ast.Dict)
                                       for key in node.keys if isinstance(key, ast.Constant)
                                       for name in [key.value] if isinstance(name, str)})
        config = (SOURCE / 'envs/config.py').read_text(encoding='utf-8')
        self.assertIn('edge_contact = RewardTermCfg(', config)
        self.assertIn('"edge_contact"', config)
        self.assertEqual((SOURCE / 'envs/geometry.py').read_bytes(),
                         (BASE / 'source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
