import copy
import math

import unittest
import torch

from go1_hier_sim.contracts import (
    RELIEF_DIM,
    RELIEF_ROWS,
    CommandRouter,
    command_from_action,
    depth_to_input,
    fixed_heading_observation,
    fixed_world_velocity_to_body,
    gait_phase_observation,
    interface_manifest,
    pack_goal,
    rectangular_region_distance,
    soft_trot_schedule,
    world_goal_to_body,
)


class ContractTests(unittest.TestCase):
    def test_fixed_heading_exposes_yaw_error_without_absolute_pose(self):
        axis = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        yaw = torch.tensor([0.0, math.pi / 2, -math.pi / 2])
        heading = fixed_heading_observation(axis, yaw)
        torch.testing.assert_close(heading[0], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(heading[1], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(heading[2], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0)

    def test_finish_region_is_the_full_one_metre_strip(self):
        forward = torch.tensor([9.10, 9.60, 10.10, 8.90, 10.30, 9.60])
        lateral = torch.tensor([-4.0, 0.0, 4.0, 0.0, 0.0, 4.3])
        distance = rectangular_region_distance(forward, lateral, 9.10, 10.10, 4.0)
        torch.testing.assert_close(distance[:3], torch.zeros(3))
        torch.testing.assert_close(distance[3:], torch.tensor([0.2, 0.2, 0.3]), atol=1e-6, rtol=0)

    def test_soft_trot_clock_has_diagonal_pairs_and_continuous_observation(self):
        phase = torch.tensor([0.25, 0.75])
        duty = torch.tensor([0.56, 0.56])
        contact, swing = soft_trot_schedule(phase, duty, 0.08)
        torch.testing.assert_close(contact[:, 0], contact[:, 3])
        torch.testing.assert_close(contact[:, 1], contact[:, 2])
        self.assertGreater(contact[0, 0], contact[0, 1])
        self.assertLess(contact[1, 0], contact[1, 1])
        self.assertTrue(((swing >= 0) & (swing <= 1)).all())
        wrapped = gait_phase_observation(torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(wrapped[0], wrapped[1], atol=1e-6, rtol=0)

    def test_fixed_world_velocity_does_not_rotate_with_robot(self):
        axis = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        speed = torch.tensor([0.65, 0.65, 0.65])
        yaw = torch.tensor([0.0, math.pi / 2, math.pi])
        body = fixed_world_velocity_to_body(axis, speed, yaw)
        torch.testing.assert_close(body[0], torch.tensor([0.65, 0.0]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(body[1], torch.tensor([0.0, -0.65]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(body[2], torch.tensor([-0.65, 0.0]), atol=1e-6, rtol=1e-5)

    def test_goal_rotation_and_expiry(self):
        xy = world_goal_to_body(torch.zeros(1, 2), torch.tensor([[2., 0.]]), torch.tensor([math.pi / 2]))
        torch.testing.assert_close(xy, torch.tensor([[0., -2.]]), atol=1e-6, rtol=1e-5)
        goal = pack_goal(xy, torch.tensor([True]), torch.tensor([0.6]))
        self.assertEqual(goal.tolist(), [[0., 0., 0., 1.]])
        with self.assertRaisesRegex(ValueError, "relative XY"):
            pack_goal(torch.zeros(1, 1), torch.tensor([True]), torch.zeros(1))


    def test_command_mapping_ownership_and_stale_stop(self):
        a = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
        v = command_from_action(a)
        torch.testing.assert_close(v, torch.tensor([[0., -.3, -1.], [.8, .3, 1.]]))
        router = CommandRouter(2)
        valid = torch.ones(2, dtype=torch.bool)
        first = router.update(v, torch.zeros(2), valid, .1, "navigation")
        stopped = router.update(v, torch.tensor([1.0, 0.0]), valid, .1, "navigation")
        self.assertTrue(stopped.stopped[0])
        self.assertFalse(stopped.stopped[1])
        self.assertEqual(stopped.reason, "invalid_or_stale_input")
        self.assertEqual(stopped.velocity[0].abs().sum(), 0)
        with self.assertRaisesRegex(ValueError, "owner"):
            router.update(v, torch.zeros(2), valid, .1, "external")
        router.reset(torch.tensor([0]))
        self.assertEqual(router.velocity[0].count_nonzero(), 0)
        self.assertGreater(router.velocity[1].count_nonzero(), 0)


    def test_invalid_depth_remains_observable(self):
        raw = torch.full((1, 60, 80, 1), float("inf"))
        raw[0, 0, 1, 0] = 3.0
        raw[0, 1, 0, 0] = float("nan")
        raw[0, 1, 1, 0] = 0.1
        image = depth_to_input(raw)
        self.assertEqual(image.shape, (1, 2, 60, 80))
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(image[0, 1].sum(), 1)
        self.assertEqual(image[0, 0, 0, 1], .5)
        with self.assertRaisesRegex(ValueError, "60,80"):
            depth_to_input(torch.ones(1, 30, 40, 1))


    def test_navigation_geometry_is_the_terrain_relief_map(self):
        # The speed layer senses terrain relief, not obstacles: schema 5 kept the
        # navigation observation size while replacing the range scan.
        manifest = interface_manifest()
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["navigation"]["input"], 193)
        self.assertEqual(manifest["vision"]["terrain_relief"], 2 * RELIEF_DIM)
        self.assertNotIn("obstacle", manifest["vision"])
        self.assertEqual(manifest["vision"]["relief_rows"], RELIEF_ROWS)
        self.assertEqual(manifest["normalization"]["relief"], "ground_gradient_clamped_to_1")


    def test_schema4_relief_migration_only_accepts_the_renamed_key(self):
        from go1_hier_sim.checkpoints import _is_schema4_relief_migration

        legacy = copy.deepcopy(interface_manifest())
        legacy["schema_version"] = 4
        legacy["vision"].pop("relief_rows")
        legacy["vision"].pop("relief_cols")
        legacy["normalization"].pop("relief")
        legacy["vision"]["obstacle"] = legacy["vision"].pop("terrain_relief")
        self.assertTrue(_is_schema4_relief_migration(legacy))
        legacy["vision"]["obstacle"] = 143
        self.assertFalse(_is_schema4_relief_migration(legacy))
        self.assertFalse(_is_schema4_relief_migration(interface_manifest()))


if __name__ == "__main__":
    unittest.main()
