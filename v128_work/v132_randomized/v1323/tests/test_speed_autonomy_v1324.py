"""Speed-autonomy preserves the observation contract and only removes speed targets."""
import ast
import importlib.util
import json
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/go1_hier_sim/go1_hier_sim'
BASE = ROOT.parent / 'v1322'


def load_helper():
    path = SOURCE / 'envs/speed_autonomy_v1324.py'
    spec = importlib.util.spec_from_file_location('speed_autonomy_v1324', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpeedAutonomyTests(unittest.TestCase):
    def test_forward_direction_command_discards_speed_magnitude(self):
        import torch
        helper = load_helper()
        command = torch.tensor([[0.35, 0.2, 0.5], [0.8, -0.2, -0.5],
                                [0.0, 0.3, 0.7], [-0.4, 0.1, -0.1]])
        expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        torch.testing.assert_close(helper.direction_only_command(command), expected)

    def test_autonomous_progress_reference_uses_saturating_speed_cap(self):
        import torch
        helper = load_helper()
        reference = helper.autonomous_progress_reference(
            torch.tensor([0.35, 0.8, 1.6]), 1.2)
        torch.testing.assert_close(reference, torch.tensor([1.2, 1.2, 1.2]))

    def test_direction_command_rejects_invalid_shape(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'shape'):
            load_helper().direction_only_command(torch.zeros((2, 2)))

    def test_runtime_gates_direction_only_observation_behind_explicit_flag(self):
        tree = ast.parse((SOURCE / 'envs/runtime.py').read_text(encoding='utf-8'))
        runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'HierarchicalRuntime')
        method = next(node for node in runtime.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_desired_stage1_command')
        strings = {node.value for node in ast.walk(method) if isinstance(node, ast.Constant)
                   and isinstance(node.value, str)}
        calls = {node.func.id for node in ast.walk(method) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        self.assertIn('GO1_STAGE1_DIRECTION_ONLY', strings)
        self.assertIn('direction_only_command', calls)
        reward_method = next(node for node in runtime.body if isinstance(node, ast.FunctionDef)
                             and node.name == '_build_locomotion_reward_terms')
        reward_strings = {node.value for node in ast.walk(reward_method)
                          if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        reward_calls = {node.func.id for node in ast.walk(reward_method) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)}
        self.assertIn('GO1_STAGE1_DIRECTION_ONLY', reward_strings)
        self.assertIn('autonomous_progress_reference', reward_calls)

    def test_speed_reward_changes_only_drop_tracking_and_raise_progress_cap(self):
        old = tomllib.loads((BASE / 'configs/rewards_stage1_gaps_v131_landing_stability.toml')
                            .read_text(encoding='utf-8'))
        treatment = tomllib.loads((ROOT / 'configs/rewards_stage1_gaps_v131_landing_stability.toml')
                                  .read_text(encoding='utf-8'))
        self.assertEqual(treatment['track_linear_velocity']['weight'], 0.0)
        self.assertEqual(treatment['forward_velocity']['weight'], 0.0)
        self.assertEqual(treatment['forward_progress']['max_progress_speed'], 1.2)
        self.assertEqual(treatment['speed_error_l1']['weight'], 0.0)
        self.assertEqual(treatment['no_progress']['weight'], 0.0)
        self.assertEqual(treatment['fallen']['weight'], old['fallen']['weight'])
        self.assertEqual(treatment['edge_contact']['weight'], 0.0)
        treatment.pop('edge_contact')
        treatment['track_linear_velocity']['weight'] = old['track_linear_velocity']['weight']
        treatment['forward_velocity']['weight'] = old['forward_velocity']['weight']
        treatment['forward_progress']['max_progress_speed'] = old['forward_progress']['max_progress_speed']
        treatment['speed_error_l1']['weight'] = old['speed_error_l1']['weight']
        treatment['no_progress']['weight'] = old['no_progress']['weight']
        self.assertEqual(treatment, old)

    def test_pilot_budget_is_200_with_40_critic_only_and_unique_a02_outputs(self):
        spec = json.loads((ROOT / 'configs/pilot_execution.json').read_text(encoding='utf-8'))
        self.assertEqual(spec['iterations'], 200)
        self.assertEqual(spec['critic_warmup_iterations'], 40)
        self.assertEqual(spec['pilot_id'], 'go1-v1323-speed-autonomy-pilot-a02')
        self.assertEqual(spec['final_holdout'], False)
        self.assertEqual(spec['automatic_full_training_authorized'], False)


if __name__ == '__main__':
    unittest.main()
