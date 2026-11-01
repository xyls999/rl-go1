"""CPU-only real-gap support reward tests; no Isaac imports."""
import importlib.util
import unittest
from pathlib import Path

import torch

HELPER = (Path(__file__).resolve().parents[1] /
          'source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py')


def load_helper():
    spec = importlib.util.spec_from_file_location('entrance_support_v1322', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.entrance_support_step


class EntranceSupportReward(unittest.TestCase):
    def setUp(self):
        self.step = load_helper()
        self.bounds = torch.full((2, 24, 3), float('nan'))
        self.bounds[:, 0] = torch.tensor([1.50, 1.67, 2.42])  # 0.12 m opening + 0.05 m lip
        self.x = torch.tensor([1.35, 1.35])
        self.foot_x = torch.full((2, 4), 1.30)
        self.foot_z = torch.zeros(2, 4)
        self.contacts = torch.ones(2, 4, dtype=torch.bool)
        self.root_z = torch.full((2,), .32)
        self.upright = torch.full((2,), .96)
        self.velocity = torch.full((2,), .4)
        self.command = torch.full((2,), .5)
        self.failed = torch.zeros(2, dtype=torch.bool)
        self.paid = torch.zeros(2, 24, dtype=torch.bool)
        self.entrance = torch.tensor([True, False])
        self.watermark = torch.full((2, 24), float('-inf'))

    def score(self):
        return self.step(self.bounds, self.x, self.foot_x, self.foot_z,
                         self.contacts, self.root_z, self.upright, self.velocity,
                         self.command, self.failed, self.paid, self.entrance,
                         self.watermark)

    def test_supported_forward_approach_only_entrance(self):
        score, watermark = self.score()
        self.assertGreater(score[0].item(), 0)
        self.assertLessEqual(score[0].item(), 1)
        self.assertEqual(score[1].item(), 0)
        self.assertAlmostEqual(watermark[0, 0].item(), 1.35, places=5)
        self.assertTrue(torch.isneginf(watermark[1]).all())

    def test_no_support_or_movement_or_real_pad(self):
        for change in (
            {'velocity': torch.zeros(2)}, {'command': torch.zeros(2)},
            {'contacts': torch.zeros(2, 4, dtype=torch.bool)},
            {'contacts': torch.tensor([[True, False, False, False]] * 2)},
            {'foot_z': torch.full((2, 4), -.10)},
            {'foot_x': torch.full((2, 4), 1.55)},
            {'root_z': torch.full((2,), .22)},
            {'upright': torch.full((2,), .8)},
            {'failed': torch.ones(2, dtype=torch.bool)},
            {'paid': torch.ones(2, 24, dtype=torch.bool)},
            {'bounds': torch.full((2, 24, 3), float('nan'))},
            {'x': torch.full((2,), .4)},
        ):
            with self.subTest(change=list(change)):
                name, value = next(iter(change.items()))
                setattr(self, name, value)
                score, _ = self.score()
                self.assertEqual(score[0].item(), 0)
                self.setUp()

    def test_oscillation_cannot_pay_twice_on_same_unpaid_gap(self):
        first, self.watermark = self.score()
        self.x[0] = 1.20
        self.velocity[0] = -.3
        backward, self.watermark = self.score()
        self.x[0] = 1.35
        self.velocity[0] = .4
        returned, self.watermark = self.score()
        self.assertGreater(first[0].item(), 0)
        self.assertEqual(backward[0].item(), 0)
        self.assertEqual(returned[0].item(), 0)
        self.x[0] = 1.38
        newly_reached, self.watermark = self.score()
        self.assertGreater(newly_reached[0].item(), 0)
        self.assertLess(newly_reached[0].item(), first[0].item())

    def test_partial_episode_reset_only_clears_selected_watermark(self):
        self.entrance[:] = True
        _, self.watermark = self.score()
        self.watermark[[0]] = float('-inf')  # runtime.reset(env_ids) performs this
        score, next_watermark = self.score()
        self.assertGreater(score[0].item(), 0)
        self.assertEqual(score[1].item(), 0)
        self.assertAlmostEqual(next_watermark[1, 0].item(), 1.35, places=5)

    def test_physical_entrance_width_includes_five_centimeter_metadata_lip(self):
        # Geometry records void_end = start + physical_width + 0.05 m lip.
        # In particular, the widest valid 0.24 m opening has a 0.29 m span.
        for physical_width, rewarded in ((.07, False), (.08, True), (.24, True), (.25, False)):
            with self.subTest(physical_width=physical_width):
                self.bounds[0, 0, 1] = 1.50 + physical_width + .05
                score, _ = self.score()
                if rewarded:
                    self.assertGreater(score[0].item(), 0)
                else:
                    self.assertEqual(score[0].item(), 0)

    def test_invalid_bounds_cannot_contaminate_score(self):
        self.bounds[0, 0] = torch.tensor([1.50, 1.50, 2.42])
        score, _ = self.score()
        self.assertEqual(score[0].item(), 0)
        self.assertTrue(torch.isfinite(score).all())


if __name__ == '__main__':
    unittest.main()
