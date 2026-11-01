"""Source-level reward wiring contracts without importing Isaac Lab."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = (Path(__file__).resolve().parents[1] /
          'source/go1_hier_sim/go1_hier_sim/envs/runtime.py')


class RuntimeEntranceContract(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text(encoding='utf-8')
        self.tree = ast.parse(self.source)
        self.runtime = next(node for node in self.tree.body
                            if isinstance(node, ast.ClassDef) and
                            any(isinstance(child, ast.FunctionDef) and child.name == 'reset'
                                for child in node.body))

    def method(self, name):
        return next(node for node in self.runtime.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def test_entrance_mask_reward_only_not_combined_mask(self):
        init = ast.get_source_segment(self.source, self.method('__init__'))
        self.assertIn('self._v132_entrance_mask', init)
        self.assertIn('str(name) == "combined_part_entrance"', init)
        self.assertIn('self._v132_support_watermark', init)
        self.assertIn('self._v132_gap_bounds.shape[:2]', init)
        self.assertIn('from .entrance_support_v1322 import entrance_support_step', self.source)

    def test_real_paid_event_is_sole_entrance_safe_segment(self):
        reward = ast.get_source_segment(self.source, self.method('_build_locomotion_reward_terms'))
        self.assertIn('entrance_support_step(', reward)
        self.assertIn('self._v132_gap_bounds', reward)
        self.assertIn('self._v132_paid', reward)
        self.assertIn('self._fallen | self._collision | self._fell_into_gap | self._out_of_bounds', reward)
        self.assertIn('safe_segment = safe_segment + v132_event * (self._v132_entrance_mask & ~self._out_of_bounds).to(v132_event.dtype)', reward)
        self.assertLess(reward.index('distance = self._update_failures()'),
                        reward.index('safe_segment = safe_segment + v132_event'))
        self.assertLess(reward.index('v132_event, self._v132_active, self._v132_paid = gap_clearance_step('),
                        reward.index('safe_segment = safe_segment + v132_event'))
        self.assertIn('safe_segment = v7["safe_segment"]', reward)
        self.assertIn('terms.update({"supported_short": supported_short, "safe_segment": safe_segment,', reward)

    def test_out_of_bounds_veto_only_new_entrance_bonus(self):
        reward = self.method('_build_locomotion_reward_terms')
        assignments = [node for node in ast.walk(reward) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'safe_segment'
                               for target in node.targets)
                       and 'v132_event' in ast.unparse(node.value)]
        self.assertEqual(len(assignments), 1)
        expression = ast.Expression(assignments[0].value)
        ast.fix_missing_locations(expression)
        context = {'torch': torch,
                   'self': SimpleNamespace(
                       _v132_entrance_mask=torch.tensor([True, True, False, False]),
                       _out_of_bounds=torch.tensor([True, False, False, True])),
                   'safe_segment': torch.tensor([0., 0., .25, .5]),
                   'v132_event': torch.ones(4)}
        actual = eval(compile(expression, '<safe_segment>', 'eval'), context)
        torch.testing.assert_close(actual, torch.tensor([0., 1., .25, .5]))

    def test_partial_reset_and_actor_contract(self):
        reset = ast.get_source_segment(self.source, self.method('reset'))
        self.assertIn('self._v132_support_watermark[ids] = float("-inf")', reset)
        self.assertIn('self._v132_paid[ids] = False', reset)
        for name in ('_refresh_observations', '_call_locomotion_policy',
                     'validate_observations'):
            actor_path = ast.get_source_segment(self.source, self.method(name))
            self.assertNotIn('entrance_support_step', actor_path)
            self.assertNotIn('_v132_support_watermark', actor_path)
            self.assertNotIn('_v132_gap_bounds', actor_path)


if __name__ == '__main__':
    unittest.main()
