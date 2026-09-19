import math
import unittest

import numpy as np
import torch

from go1_hier_sim.envs.geometry import (
    CAMERA_PITCH,
    COURSE_FEATURE_END,
    COURSE_GOAL_OFFSET,
    COURSE_LEAD_IN_END,
    COURSE_TREAD_DEPTH,
    START_IN_TILE,
    TILE_SPACING,
    TILE_SIZE,
    VisibleSurfaceMemory,
    build_scene,
    build_tile,
    camera_coordinates,
    camera_frustum,
    quat_rotate,
    rotate_yaw,
    yaw_from_quat,
)


def _course_surface_top(tile, tile_x, local_y=0.0):
    """Return the walkable surface height at a tile-frame position.

    Course geometry is authored in tile coordinates (the spawn sits at
    START_IN_TILE), while reward and finish offsets are spawn-relative. The
    highest hit is the walking surface: a course patch floats above the base
    plate, and a stair tread is a solid box.
    """
    origin = np.array([[tile_x, START_IN_TILE[1] + local_y, 5.0]])
    hits, _, _ = tile.ray.intersects_location(
        origin, np.array([[0.0, 0.0, -1.0]]), multiple_hits=True
    )
    if len(hits) == 0:
        raise AssertionError(f"no ground at course x={tile_x}")
    return float(hits[:, 2].max())


class GeometryTests(unittest.TestCase):
    def test_quaternion_yaw_and_camera_frame(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        root = torch.zeros(1, 3)
        # The physical optical axis is pitched down twenty degrees from body +X.
        axis = torch.tensor([math.cos(CAMERA_PITCH), 0.0, -math.sin(CAMERA_PITCH)])
        point = root + torch.tensor([[0.25, 0.0, 0.12]]) + 2.0 * axis
        depth, lateral, up, position = camera_coordinates(point[:, None], root, identity)
        torch.testing.assert_close(depth, torch.tensor([[2.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(lateral, torch.zeros_like(lateral), atol=1e-6, rtol=0)
        torch.testing.assert_close(up, torch.zeros_like(up), atol=1e-6, rtol=0)
        visible, returned_position = camera_frustum(point[:, None], root, identity)
        self.assertTrue(visible.item())
        torch.testing.assert_close(position, returned_position)

        yaw = torch.tensor([math.pi / 2])
        quaternion = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
        torch.testing.assert_close(yaw_from_quat(quaternion), yaw, atol=1e-6, rtol=0)
        rotated = quat_rotate(quaternion, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            rotate_yaw(torch.tensor([[[1.0, 0.0, 2.0]]]), yaw),
            torch.tensor([[[0.0, 1.0, 2.0]]]), atol=1e-6, rtol=0,
        )

    def test_visibility_memory_is_world_fixed_height_aware_and_expires(self):
        origins = torch.tensor([START_IN_TILE, START_IN_TILE])
        memory = VisibleSurfaceMemory(origins, resolution=0.1, ttl=2.0)
        points = torch.tensor([
            [[3.2, 8.1, 0.0], [3.4, 8.1, 0.2]],
            [[3.2, 8.1, 0.0], [3.4, 8.1, 0.2]],
        ])
        memory.update(points, torch.tensor([[True, False], [False, True]]), timestamp=1.0)
        known = memory.query(points, timestamp=2.0)
        self.assertEqual(known.tolist(), [[True, False], [False, True]])
        raised_surface = points.clone()
        raised_surface[:, 0, 2] += 0.2
        self.assertFalse(memory.query(raised_surface, timestamp=2.0)[:, 0].any())
        self.assertFalse(memory.query(points, timestamp=3.01).any())
        memory.reset(torch.tensor([1]))
        self.assertFalse(memory.query(points, timestamp=1.0)[1].any())

    def test_each_tile_has_spawn_clearance_and_optional_obstacles_in_same_mesh(self):
        for kind in ("flat", "slopes", "rough", "stairs", "course"):
            plain = build_tile(kind, 0.5, np.random.default_rng(7), obstacles=False)
            occupied = build_tile(kind, 0.5, np.random.default_rng(7), obstacles=True)
            plain_mask = plain.metadata["obstacle_faces"]
            occupied_mask = occupied.metadata["obstacle_faces"]
            self.assertEqual(plain_mask.dtype, np.dtype(bool))
            self.assertFalse(plain_mask.any())
            self.assertTrue(occupied_mask.any())
            # Obstacles are faces of the one returned mesh, so the identical object
            # can be imported for rendering/collision and converted for ray queries.
            self.assertEqual(len(occupied_mask), len(occupied.faces))
            obstacle_vertices = occupied.vertices[np.unique(occupied.faces[occupied_mask])]
            self.assertGreater(obstacle_vertices[:, 2].max(), 0.9)
            spawn = np.asarray(START_IN_TILE[:2])
            obstacle_xy = obstacle_vertices[:, :2]
            self.assertGreater(np.linalg.norm(obstacle_xy - spawn, axis=1).min(), 1.0)

    def test_course_is_a_dense_mix_with_little_level_ground(self):
        # The route has to be a dense mix of different feature kinds rather than
        # one slope plus one staircase, the blocks have to tile the feature span
        # without level gaps, and only a short level approach may be left before
        # the finish line.
        easy = build_tile("course", 0.1, np.random.default_rng(17), obstacles=False)
        hard = build_tile("course", 0.8, np.random.default_rng(17), obstacles=False)
        for profile in (easy.metadata["course_profile"], hard.metadata["course_profile"]):
            kinds = set(profile["feature_kinds"])
            self.assertGreaterEqual(len(kinds), 3)
            self.assertTrue({"slope", "stairs"} <= kinds)
            self.assertGreater(profile["roughness_amplitude"], 0.0)
            self.assertLess(profile["roughness_amplitude"], 0.02)
            self.assertGreater(profile["slope_height"], 0.0)
            self.assertLess(profile["slope_height"], 0.90)
            rises = np.asarray(profile["step_rises"], dtype=float)
            self.assertGreaterEqual(rises.size, 6)
            self.assertTrue((rises > 0.010).all())
            blocks = profile["blocks"]
            self.assertGreater(len(blocks), 3)
            self.assertGreaterEqual(
                sum(1 for block in blocks if block["kind"] == "stairs"), 2
            )
            for left, right in zip(blocks, blocks[1:]):
                self.assertAlmostEqual(left["end"], right["start"], places=6)
            self.assertAlmostEqual(blocks[0]["start"], COURSE_LEAD_IN_END, places=6)
            self.assertGreaterEqual(blocks[-1]["end"], profile["feature_end"] - 0.16)
            self.assertLess(profile["flat_approach_m"], 3.5)
        easy_profile = easy.metadata["course_profile"]
        hard_profile = hard.metadata["course_profile"]
        self.assertGreater(hard_profile["slope_height"], easy_profile["slope_height"])
        self.assertGreater(hard_profile["roughness_amplitude"], easy_profile["roughness_amplitude"])
        self.assertEqual(hard_profile["feature_end"], COURSE_FEATURE_END)

    def test_course_step_rises_are_randomised_inside_the_kinematic_envelope(self):
        # A 0.25 + 0.25 m leg standing at 0.365 m can lift a foot and then push the
        # body up by roughly 0.16 m, so one stair rise may never exceed that cap,
        # and the curriculum has to actually reach real, uneven stairs rather than a
        # monotone ramp.
        hard = build_tile("course", 1.0, np.random.default_rng(5), obstacles=False)
        profile = hard.metadata["course_profile"]
        self.assertIn("step_rises", profile)
        rises = np.asarray(profile["step_rises"], dtype=float)
        self.assertGreaterEqual(rises.size, 6)
        self.assertGreaterEqual(rises.min(), 0.04 - 1e-9)
        self.assertLessEqual(rises.max(), 0.16 + 1e-9)
        self.assertGreater(rises.max(), 0.10)
        # Height is not fixed: the risers differ inside the same stairway.
        self.assertGreater(rises.max() - rises.min(), 0.02)
        easy = build_tile("course", 0.0, np.random.default_rng(5), obstacles=False)
        easy_rises = np.asarray(easy.metadata["course_profile"]["step_rises"], dtype=float)
        self.assertGreaterEqual(easy_rises.size, 6)
        self.assertAlmostEqual(float(easy_rises.max()), 0.04, places=9)

        # Treads must be long enough for the ~0.6 m trunk to bridge one without
        # jamming on the next riser, at every difficulty the curriculum reaches.
        for difficulty in (0.0, 0.5, 1.0):
            tile = build_tile("course", difficulty, np.random.default_rng(11), obstacles=False)
            tile_profile = tile.metadata["course_profile"]
            self.assertGreaterEqual(tile_profile["tread_depth_min"], 0.35)
            self.assertLessEqual(
                max(tile_profile["tread_depths"]), COURSE_TREAD_DEPTH + 0.05 + 1e-9
            )

    def test_course_ground_extends_past_the_goal_and_the_finish_region(self):
        # A straight course whose mesh ends where the finish region begins forces
        # every episode to fall off the terrain edge at the last metre, so the
        # navigation goal can never be reached.  The traversable ground must
        # continue past the goal and past the declared finish region.
        for difficulty in (0.0, 0.1, 0.8):
            with self.subTest(difficulty=difficulty):
                course = build_tile("course", difficulty, np.random.default_rng(42), obstacles=False)
                profile = course.metadata["course_profile"]
                required = max(COURSE_GOAL_OFFSET, profile["finish_end_offset"]) + 0.5
                direction = np.array([[0.0, 0.0, -1.0]])
                for local_x in (COURSE_GOAL_OFFSET, profile["finish_end_offset"], required):
                    origins = np.array([[START_IN_TILE[0] + local_x, START_IN_TILE[1], 5.0]])
                    hits, _, _ = course.ray.intersects_location(origins, direction, multiple_hits=False)
                    self.assertEqual(len(hits), 1, f"no ground at course x={local_x}")
                    self.assertAlmostEqual(float(hits[0, 2]), 0.0, places=6)

    def test_course_envelope_holds_across_randomised_tiles(self):
        # Every tile samples its own slope angle and stair rises, so the physical
        # caps must hold over many draws, and the sampled band must be explored.
        angles, rises = [], []
        for seed in range(48):
            tile = build_tile("course", 1.0, np.random.default_rng(seed), obstacles=False)
            profile = tile.metadata["course_profile"]
            angles.append(float(profile["slope_angle_deg"]))
            rises.append(float(profile["max_step_rise"]))
        self.assertGreaterEqual(min(angles), 5.0 - 1e-9)
        self.assertLessEqual(max(angles), 30.0 + 1e-9)
        self.assertGreater(max(angles), 25.0)
        self.assertGreaterEqual(min(rises), 0.04 - 1e-9)
        self.assertLessEqual(max(rises), 0.16 + 1e-9)
        self.assertGreater(max(rises), 0.12)

    def test_course_profile_is_continuous_and_walks_the_whole_span(self):
        # Dense mixed terrain must stay traversable: sample the walking surface
        # along the route and require no step larger than one legal riser, a
        # bounded summit, real features instead of a plane, minimal level ground,
        # and a level datum again at the finish line.
        for difficulty in (0.0, 0.5, 1.0):
            with self.subTest(difficulty=difficulty):
                tile = build_tile("course", difficulty, np.random.default_rng(23), obstacles=False)
                profile = tile.metadata["course_profile"]
                xs = np.arange(COURSE_LEAD_IN_END, profile["feature_end"], 0.1)
                heights = np.array([_course_surface_top(tile, float(x)) for x in xs])
                jumps = np.abs(np.diff(heights))
                self.assertLessEqual(float(jumps.max()), 0.16 + 1e-6)
                # Real features rather than a plane: the summit has to grow with
                # difficulty, from a single 4 cm step at difficulty zero upwards.
                self.assertGreater(float(heights.max()), 0.10 + 0.05 * difficulty)
                self.assertLessEqual(float(heights.max()), 1.0)
                # Almost no level ground: most samples sit off the datum.
                self.assertGreater(float((heights > 0.02).mean()), 0.25)
                finish_line = START_IN_TILE[0] + profile["finish_start_offset"]
                self.assertAlmostEqual(_course_surface_top(tile, finish_line), 0.0, delta=1e-6)
                self.assertAlmostEqual(
                    _course_surface_top(tile, profile["feature_end"] + 0.5), 0.0, delta=1e-6
                )
                self.assertAlmostEqual(_course_surface_top(tile, TILE_SIZE - 2.0), 0.0, delta=1e-6)

    def test_course_blocks_start_and_end_at_the_course_datum(self):
        # Blocks must compose. A block that ended above the datum would leave a
        # lip that neither the next block nor the level finish strip can absorb.
        for seed in (3, 11, 23):
            with self.subTest(seed=seed):
                tile = build_tile("course", 1.0, np.random.default_rng(seed), obstacles=False)
                blocks = tile.metadata["course_profile"]["blocks"]
                for block in blocks:
                    self.assertAlmostEqual(
                        _course_surface_top(tile, block["start"] - 0.02), 0.0, delta=0.03
                    )
                    self.assertAlmostEqual(
                        _course_surface_top(tile, block["end"] - 0.02), 0.0, delta=0.03
                    )

    def test_scene_has_one_distinct_origin_per_environment(self):
        mesh, origins = build_scene(7, "mixed", True, seed=13, difficulty=0.4)
        self.assertEqual(origins.shape, (7, 3))
        self.assertEqual(len(np.unique(origins[:, :2], axis=0)), 7)
        self.assertEqual(len(mesh.metadata["obstacle_faces"]), len(mesh.faces))
        self.assertTrue(mesh.metadata["obstacle_faces"].any())
        self.assertTrue((origins[:, 0] >= START_IN_TILE[0]).all())
        self.assertTrue((origins[:, 1] >= START_IN_TILE[1]).all())
        self.assertLessEqual(mesh.bounds[1, 0] - mesh.bounds[0, 0], 3 * TILE_SPACING + TILE_SIZE)

    def test_invalid_geometry_inputs_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unknown terrain"):
            build_tile("water", 0.5, np.random.default_rng(0), False)
        with self.assertRaisesRegex(ValueError, "difficulty"):
            build_tile("flat", 1.1, np.random.default_rng(0), False)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_scene(0, "flat", False, 0, 0.5)
        with self.assertRaisesRegex(ValueError, "bounds"):
            build_scene(1, "course", False, 0, 0.5, 0.8, 0.2)


if __name__ == "__main__":
    unittest.main()
