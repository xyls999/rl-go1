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
    GAP_FLOOR_DEPTH,
    GAP_LANDING_PAD_M,
    GAP_JUMP_WIDTH_M,
    GAP_RHYTHM_COUNT,
    GAP_RHYTHM_MIN,
    GAP_RHYTHM_PAD_M,
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
    GAP_WIDTH_MIN,
    GAP_WIDTH_MAX,
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
            # feature_end is the absolute cap; the curriculum span grows with
            # difficulty, so require a real obstacle run, not the full cap.
            self.assertGreaterEqual(blocks[-1]["end"], profile["feature_end"] * 0.5)
            # Route length is now part of the curriculum: the easy end
            # leaves a long flat approach rather than a full-length course.
            self.assertLess(profile["flat_approach_m"], 20.0)
        easy_profile = easy.metadata["course_profile"]
        hard_profile = hard.metadata["course_profile"]
        self.assertGreater(hard_profile["slope_height"], easy_profile["slope_height"])
        self.assertGreater(hard_profile["roughness_amplitude"], easy_profile["roughness_amplitude"])
        # The feature span grows with difficulty and only reaches the full
        # cap at difficulty 1, so assert the growth rather than the cap.
        # feature_end is the absolute cap, so compare where the blocks
        # actually stop: the curriculum span grows with difficulty.
        self.assertGreater(hard_profile["blocks"][-1]["end"],
                           easy_profile["blocks"][-1]["end"])
        self.assertGreater(hard_profile["blocks"][-1]["end"], 25.0)

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

    def test_course_gaps_mix_trenches_with_the_normal_features(self):
        # The mixed route must keep slopes, stairs, short walkable trenches,
        # long jump-required trenches and safe takeoff/landing pads. The short
        # gaps can use shorter walking pads, but never shorten a long-gap pad.
        difficulty = 0.6
        tile = build_tile("course_gaps", difficulty, np.random.default_rng(29), obstacles=False)
        profile = tile.metadata["course_profile"]
        blocks = profile["blocks"]
        gap_blocks = [block for block in blocks if block["kind"] == "gap"]
        self.assertGreaterEqual(len(gap_blocks), 2)
        self.assertTrue(set(profile["feature_kinds"]) - {"gap"} & {"slope", "stairs", "rough", "bumps", "ledge"})
        for block in gap_blocks:
            self.assertGreaterEqual(block["width"], GAP_WIDTH_MIN - 1e-9)
            self.assertLessEqual(block["width"], GAP_WIDTH_MAX + 1e-9)
            self.assertGreater(block["depth"], GAP_FLOOR_DEPTH - 1e-9)
            pad = block["end"] - block["void_end"]
            if block["jump_required"]:
                self.assertGreaterEqual(pad, GAP_LANDING_PAD_M - 1e-9)
            else:
                self.assertGreaterEqual(pad, 0.60 - 1e-9)
        # The plain course stays gap-free: the mix is opt-in per terrain kind.
        plain = build_tile("course", difficulty, np.random.default_rng(29), obstacles=False)
        self.assertEqual(plain.metadata["course_profile"]["gap_count"], 0)

    def test_gap_width_is_the_difficulty_curriculum_variable(self):
        # A single policy only learns the leap if the void grows with the
        # curriculum; a fixed specialist-width void would open every episode at
        # exam difficulty (and it is unrecoverable if the trench is shallow).
        widths = {}
        depths = {}
        for difficulty in (0.1, 0.9):
            profile = build_tile("course_gaps", difficulty, np.random.default_rng(11),
                                 obstacles=False).metadata["course_profile"]
            widths[difficulty] = max(profile["gap_widths"])
            depths[difficulty] = max(block["depth"] for block in profile["blocks"]
                                     if block["kind"] == "gap")
        self.assertLess(widths[0.1], widths[0.9])
        self.assertGreaterEqual(widths[0.1], GAP_WIDTH_MIN - 1e-9)
        self.assertLessEqual(widths[0.9], GAP_WIDTH_MAX + 1e-9)
        # Every trench must be deep enough that stepping in is a failure,
        # not a recoverable dip the policy can walk through.
        for difficulty, depth in depths.items():
            self.assertGreater(depth, 0.48)

    def test_course_gaps_contains_a_continuous_trench_rhythm(self):
        # One uninterrupted run of close-spaced voids is required, otherwise a
        # single leap clears the whole gap section and the policy never has to
        # plan a second take-off point.
        def rhythm_run(difficulty):
            profile = build_tile("course_gaps", difficulty, np.random.default_rng(3),
                                 obstacles=False).metadata["course_profile"]
            blocks = profile["blocks"]
            best = run = 0
            for block in blocks:
                if block.get("rhythm"):
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            return best, sum(1 for b in blocks if b.get("rhythm"))

        def wide_gaps(difficulty):
            profile = build_tile("course_gaps", difficulty, np.random.default_rng(3),
                                 obstacles=False).metadata["course_profile"]
            blocks = profile["blocks"]
            return (profile.get("wide_gap_count", 0),
                    [b for b in blocks if b.get("jump_required")])

        # The mixed route guarantees a consecutive jump-required run even in
        # the easy band. Widen the gaps with difficulty, not their presence.
        for difficulty in (0.0, 0.3, 0.6, 1.0):
            run, total = rhythm_run(difficulty)
            self.assertGreaterEqual(run, GAP_RHYTHM_MIN)
            self.assertGreaterEqual(total, GAP_RHYTHM_MIN)
            wide_count, wide_blocks = wide_gaps(difficulty)
            self.assertGreaterEqual(wide_count, GAP_RHYTHM_MIN)
            for block in wide_blocks:
                self.assertGreater(block["width"], GAP_JUMP_WIDTH_M)
        self.assertEqual(rhythm_run(0.0)[0], GAP_RHYTHM_MIN)

    def test_gap_course_is_a_dense_trench_route_below_the_datum(self):
        # Regression guard: the dedicated jump course must be built by the
        # course branch. A kind that falls through every geometry branch yields
        # a plain slab, trains a walking policy by accident and reports a
        # meaningless 100 % completion.
        difficulty = 0.3
        tile = build_tile("gap_course", difficulty, np.random.default_rng(7), obstacles=False)
        profile = tile.metadata["course_profile"]
        self.assertGreaterEqual(profile["gap_count"], 10)
        for width in profile["gap_widths"]:
            self.assertGreaterEqual(width, GAP_WIDTH_MIN - 1e-9)
            self.assertLessEqual(width, GAP_WIDTH_MAX + 1e-9)
        # Every pad between consecutive trenches must outrun the hop.
        for block in profile["blocks"]:
            if block["kind"] == "gap":
                self.assertGreaterEqual(block["end"] - block["void_end"], GAP_LANDING_PAD_M - 1e-9)
        # Trench floors sit below the datum, deeper than the step envelope.
        self.assertLess(float(tile.bounds[0, 2]), -0.25)
        # The finish strip stays level ground so completion is reachable.
        self.assertAlmostEqual(_course_surface_top(tile, COURSE_FEATURE_END + 0.5), 0.0, delta=1e-6)

    def test_gap_trench_is_a_real_void_spanning_the_tile(self):
        # A negative height patch inside the base slab is buried geometry: the
        # robot walks on the slab top and never falls in. The trench must be a
        # real void with a low floor, spanning the full tile width so it cannot
        # be walked around.
        for kind in ("gap_course", "course_gaps"):
            with self.subTest(kind=kind):
                tile = build_tile(kind, 0.6, np.random.default_rng(7), obstacles=False)
                block = next(
                    b for b in tile.metadata["course_profile"]["blocks"] if b["kind"] == "gap"
                )
                x_mid = 0.5 * (block["start"] + block["void_end"])
                for y in (1.0, 8.0, 15.0):
                    hits, _, _ = tile.ray.intersects_location(
                        [[x_mid, y, 3.0]], [[0.0, 0.0, -1.0]], multiple_hits=False
                    )
                    self.assertEqual(len(hits), 1)
                    self.assertLess(float(hits[0][2]), -0.25)
                for x in (block["start"] - 0.25, block["void_end"] + 0.25):
                    hits, _, _ = tile.ray.intersects_location(
                        [[x, 8.0, 3.0]], [[0.0, 0.0, -1.0]], multiple_hits=False
                    )
                    self.assertEqual(len(hits), 1)
                    self.assertAlmostEqual(float(hits[0][2]), 0.0, delta=1e-6)

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

class RcCourseRosterTests(unittest.TestCase):
    """The user's roster: one terrain per environment, no flat filler, and a
    single-line gap discriminator enforced in the geometry."""

    def test_roster_matches_the_requested_families(self):
        from go1_hier_sim.envs.task_curriculum import TERRAIN_FAMILIES
        self.assertEqual(tuple(TERRAIN_FAMILIES), (
            "stairs_up_long", "stairs_down_long", "stairs_mixed",
            "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
            "tire_field", "gaps_short", "gaps_long", "combined",
        ))

    def test_short_gaps_are_all_walkable_and_long_gaps_all_need_a_jump(self):
        from go1_hier_sim.envs.geometry import GAP_JUMP_THRESHOLD_M
        for difficulty in (0.0, 0.3, 0.6, 1.0):
            with self.subTest(difficulty=difficulty):
                short = build_tile("gaps_short", difficulty, np.random.default_rng(5),
                                   obstacles=False).metadata["course_profile"]
                long_ = build_tile("gaps_long", difficulty, np.random.default_rng(5),
                                   obstacles=False).metadata["course_profile"]
                self.assertGreaterEqual(short["gap_count"], 10)
                self.assertGreaterEqual(long_["gap_count"], 10)
                for width in short["gap_widths"]:
                    self.assertLessEqual(width, GAP_JUMP_THRESHOLD_M + 1e-9)
                for width in long_["gap_widths"]:
                    self.assertGreater(width, GAP_JUMP_THRESHOLD_M)
                self.assertEqual(short["wide_gap_count"], 0)
                self.assertEqual(long_["wide_gap_count"], long_["gap_count"])

    def test_each_obstacle_has_a_dedicated_full_length_course(self):
        # Mutations caught: a family removed from the roster, a flat filler at
        # low difficulty, or the narrow staircase replacing itself with tyres.
        pure = {
            "stairs_up_long": {"stair_ascent", "ramp"},
            "stairs_down_long": {"stair_descent", "ramp"},
            "stairs_narrow_8_mixed": {"final_stairs"},
            "stairs_mixed": {"stairs", "long_stairs"},
            "slope_up": {"slope"}, "slope_down": {"slope"},
            "slope_mixed": {"slope"}, "tire_field": {"tire_field"},
            "gaps_short": {"gap"}, "gaps_long": {"gap"},
        }
        for kind, permitted in pure.items():
            for difficulty in (0.0, 0.1, 0.6):
                with self.subTest(kind=kind, difficulty=difficulty):
                    tile = build_tile(kind, difficulty, np.random.default_rng(9), False)
                    blocks = tile.metadata["course_profile"]["blocks"]
                    self.assertTrue(blocks)
                    self.assertTrue({b["kind"] for b in blocks} <= permitted)
                    self.assertGreaterEqual(blocks[-1]["end"], 30.0)
                    self.assertLessEqual(blocks[-1]["end"], 33.0)
                    self.assertLess(len(tile.faces), 35000)
                    route = ([b for b in blocks if b["kind"] ==
                              ("stair_ascent" if kind == "stairs_up_long" else "stair_descent")]
                             if kind in ("stairs_up_long", "stairs_down_long") else blocks)
                    for before, after in zip(route, route[1:]):
                        self.assertAlmostEqual(before["end"], after["start"], places=6)
                    if kind == "stairs_narrow_8_mixed":
                        self.assertGreaterEqual(sum(b["steps"] == 8 for b in blocks), 3)
                        self.assertTrue(all(abs(t - .35) < 1e-9 for b in blocks
                                            for t in b["tread_depths"]))
                        heights = [h for b in blocks for h in b["riser_heights"]]
                        self.assertTrue(all(abs(h-heights[0]) < 1e-9 for h in heights))
                    if kind in ("slope_up", "slope_down"):
                        direction = 1 if kind == "slope_up" else -1
                        self.assertTrue(all(b["peak"] * direction > 0 for b in blocks))

    def test_downhill_is_a_visible_terrain_surface_not_buried_under_base(self):
        # The old z=0 slab hid every below-datum downhill patch from PhysX.
        # Test the actual first-hit surface, not just a negative metadata peak.
        for kind in ("slope_down", "slope_mixed", "combined"):
            with self.subTest(kind=kind):
                tile = build_tile(kind, 0.6, np.random.default_rng(9), False)
                dip = next(b for b in tile.metadata["course_profile"]["blocks"]
                           if b["kind"] == "slope" and b["peak"] < 0)
                x = (dip["start"] + dip["end"]) / 2
                hits, _, _ = tile.ray.intersects_location(
                    [[x, 8.0, 2.0]], [[0., 0., -1.]], multiple_hits=False,
                )
                self.assertEqual(len(hits), 1)
                self.assertLess(float(hits[0][2]), -0.02)

    def test_combined_always_contains_tires_both_gap_regimes_stairs_and_grades(self):
        # A missing tyre segment or jump-required trench in the easy band is a
        # curriculum defect even if those appear by chance at higher levels.
        from go1_hier_sim.envs.geometry import GAP_JUMP_THRESHOLD_M, TIRE_THICKNESS
        for difficulty in (0.0, 0.1, 0.6):
            for seed in (5, 11):
                with self.subTest(difficulty=difficulty, seed=seed):
                    tile = build_tile("combined", difficulty, np.random.default_rng(seed), False)
                    blocks = tile.metadata["course_profile"]["blocks"]
                    tires = [b for b in blocks if b["kind"] == "tire_field"]
                    gaps = [b for b in blocks if b["kind"] == "gap"]
                    self.assertGreaterEqual(len(tires), 3)
                    self.assertTrue(all(b["tyres"] >= 8 for b in tires))
                    self.assertTrue(all(abs(b["peak"] - TIRE_THICKNESS) < 1e-6 for b in tires))
                    self.assertGreaterEqual(sum(b["width"] <= GAP_JUMP_THRESHOLD_M for b in gaps), 3)
                    self.assertGreaterEqual(sum(b["width"] > GAP_JUMP_THRESHOLD_M for b in gaps), 3)
                    self.assertTrue(any(b["kind"] in ("stairs", "long_stairs") for b in blocks))
                    slopes = [b for b in blocks if b["kind"] == "slope"]
                    self.assertTrue(any(b["peak"] > 0 for b in slopes))
                    self.assertTrue(any(b["peak"] < 0 for b in slopes))
                    self.assertLessEqual(blocks[-1]["end"], 33.0)
                    self.assertGreaterEqual(blocks[-1]["end"], 30.0)
                    for before, after in zip(blocks, blocks[1:]):
                        self.assertAlmostEqual(before["end"], after["start"], places=6)

    def test_combined_final_eight_step_stairs_split_the_last_tire_field(self):
        # Regression: a monolithic tyre tail previously omitted the requested
        # eight-rise ascent/descent; all seeds must retain tyres on BOTH sides.
        for difficulty in (0.0, 0.1, 0.6, 1.0):
            for seed in (5, 11):
                with self.subTest(difficulty=difficulty, seed=seed):
                    tile = build_tile("combined", difficulty, np.random.default_rng(seed), False)
                    blocks = tile.metadata["course_profile"]["blocks"]
                    long_end = max(i for i, b in enumerate(blocks)
                                   if b["kind"] == "gap" and b["jump_required"])
                    tail = blocks[long_end + 1:]
                    self.assertEqual([b["kind"] for b in tail],
                                     ["tire_field", "final_stairs", "tire_field"])
                    front, stairs, back = tail
                    self.assertGreaterEqual(front["tyres"], 8)
                    self.assertGreaterEqual(back["tyres"], 8)
                    self.assertEqual(stairs["steps"], 8)  # 8 up AND 8 down
                    earlier = next(b for b in blocks if b["kind"] == "stairs")
                    self.assertEqual(len(stairs["riser_heights"]), 8)
                    self.assertEqual(len(stairs["descent_heights"]), 8)
                    ordinary = earlier["riser_heights"] + earlier["descent_heights"]
                    self.assertTrue(all(abs(h - ordinary[0]) < 1e-9 for h in ordinary))
                    self.assertGreaterEqual(max(stairs["riser_heights"]), 0.16)
                    self.assertGreaterEqual(max(stairs["tread_depths"]), 0.45)
                    self.assertGreaterEqual(min(stairs["tread_depths"]), 0.35 - 1e-9)
                    self.assertLessEqual(back["end"], 33.0)
                    self.assertGreaterEqual(back["end"], 30.0)

    def test_final_stairs_eight_actual_mesh_steps_have_the_matching_riser(self):
        tile = build_tile("combined", 0.1, np.random.default_rng(5), False)
        blocks = tile.metadata["course_profile"]["blocks"]
        first = next(b for b in blocks if b["kind"] == "stairs")
        final = next(b for b in blocks if b["kind"] == "final_stairs")
        x = final["start"]
        sample_x = []
        for tread in final["tread_depths"][:8]:
            sample_x.append(x + tread / 2)
            x += tread
        hits, _, _ = tile.ray.intersects_location(
            [[x, 8.0, 2.0] for x in sample_x],
            [[0., 0., -1.]] * 8, multiple_hits=False,
        )
        heights = sorted(float(hit[2]) for hit in hits)
        self.assertEqual(len(heights), 8)
        self.assertGreater(final["riser_heights"][2], first["riser_heights"][0])
        for actual, expected in zip(heights, np.cumsum(final["riser_heights"])):
            self.assertAlmostEqual(actual, expected, delta=1e-5)

    def test_combined_keeps_diagram_section_order_across_seeds(self):
        def order(seed):
            blocks = build_tile("combined", 0.1, np.random.default_rng(seed), False).metadata["course_profile"]["blocks"]
            return tuple(b["kind"] for b in blocks if b["kind"] != "gap")
        self.assertEqual(order(5), order(11))
        self.assertEqual(order(5)[:4], ("tire_field", "slope", "slope", "stairs"))

    def test_tire_field_is_a_continuous_band_over_the_whole_span(self):
        from go1_hier_sim.envs.geometry import FEATURE_STOP, TIRE_THICKNESS
        profile = build_tile("tire_field", 0.3, np.random.default_rng(5),
                             obstacles=False).metadata["course_profile"]
        bands = [b for b in profile["blocks"] if b["kind"] == "tire_field"]
        self.assertEqual(len(bands), 1)
        band = bands[0]
        self.assertGreaterEqual(band["tyres"], 20)
        # A full-coverage family cannot shrink with the difficulty band.
        self.assertGreater(band["end"] - band["start"], 25.0)
        self.assertGreaterEqual(band["peak"], 0.04)
        self.assertLessEqual(band["peak"], 0.08)

    def test_every_family_paves_the_run_up_to_the_finish_strip(self):
        # No unpaved hole between the last feature and the flat finish strip:
        # a dense gap family that stopped early used to leave one.
        from go1_hier_sim.envs.geometry import COURSE_FEATURE_END
        for kind in ("stairs_up_long", "slope_mixed", "tire_field",
                     "gaps_short", "gaps_long"):
            with self.subTest(kind=kind):
                tile = build_tile(kind, 0.3, np.random.default_rng(11), obstacles=False)
                self.assertGreater(float(tile.bounds[1, 0]), COURSE_FEATURE_END)
