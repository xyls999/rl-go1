"""Real tensor checks for the hier_v2 reward contracts, models and curriculum."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from go1_hier_sim.contracts import LOCO_OBS_DIM, NAV_OBS_DIM
from go1_hier_sim.envs.geometry import COURSE_GOAL_OFFSET
from go1_hier_sim.hier_v2.model_v2 import (
    EncoderActorCore,
    EncoderCriticCore,
    HierActorCriticV2,
    build_actor,
    build_critic,
    detect_actor_variant,
)
from go1_hier_sim.hier_v2.rewards_v2 import (
    LOCO_V2_FIELDS,
    NAV_V2_FIELDS,
    LocoV2Inputs,
    NavV2Inputs,
    NavV2Tracker,
    compute_loco_v2_terms,
    finish_crossed,
    load_loco_v2_rewards,
    load_nav_v2_rewards,
    local_ground,
    scan_grid,
    wall_score,
)
from go1_hier_sim.hier_v2.robustness_v2 import (
    RobustnessCurriculum,
    mix_for,
)
from go1_hier_sim.models import ActorCore


ROOT = Path(__file__).resolve().parents[1]


def loco_inputs(n=3, contacts=(True, False, True, False)) -> LocoV2Inputs:
    torch.manual_seed(7)
    command = torch.zeros(n, 3)
    command[:, 0] = 0.5
    contact = torch.tensor(list(contacts), dtype=torch.bool).expand(n, 4).contiguous()
    force_z = torch.where(contact, torch.full((n, 4), 60.0), torch.zeros(n, 4))
    return LocoV2Inputs(
        command=command,
        lin_vel_b=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3).contiguous(),
        ang_vel_b=torch.zeros(n, 3),
        gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3).contiguous(),
        joint_pos_rel=torch.zeros(n, 12),
        joint_vel=torch.zeros(n, 12),
        joint_acc=torch.zeros(n, 12),
        applied_torque=torch.zeros(n, 12),
        joint_lower=torch.full((n, 12), -1.0),
        joint_upper=torch.full((n, 12), 1.0),
        foot_xy_speed=torch.zeros(n, 4),
        foot_z=torch.where(contact, torch.full((n, 4), -0.30), torch.full((n, 4), -0.18)),
        foot_contacts=contact,
        foot_force_z=force_z,
        foot_force_xy=torch.zeros(n, 4),
        first_contact=contact,
        last_air_time=torch.where(contact, torch.zeros(n, 4), torch.full((n, 4), 0.35)),
        action=torch.zeros(n, 12),
        previous_action=torch.zeros(n, 12),
        previous_previous_action=torch.zeros(n, 12),
        base_height=torch.full((n,), 0.33),
        terrain_height_delta=torch.zeros(n),
        contact_count=torch.zeros(n),
        failure=torch.zeros(n, dtype=torch.bool),
    )


def nav_inputs(n=2, distance=5.0, wall=False) -> NavV2Inputs:
    rows, columns = 16, 32
    scan = torch.full((n, rows * columns), -0.30)
    if wall:
        # A riser spans the full lateral width of the first forward columns.
        scan = scan.reshape(n, rows, columns)
        scan[:, :, :2] = 0.10
        scan = scan.reshape(n, rows * columns)
    return NavV2Inputs(
        distance=torch.full((n,), distance),
        goal_dir_body=torch.tensor([[1.0, 0.0]]).expand(n, 2).contiguous(),
        lin_vel_b=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3).contiguous(),
        ang_vel_b=torch.zeros(n, 3),
        roll=torch.zeros(n),
        pitch=torch.zeros(n),
        command=torch.tensor([[0.6, 0.0, 0.0]]).expand(n, 3).contiguous(),
        scan=scan,
        contact_count=torch.zeros(n),
        done=torch.zeros(n, dtype=torch.bool),
        failure=torch.zeros(n, dtype=torch.bool),
        dt=0.1,
        scan_valid=torch.ones(n, rows * columns, dtype=torch.bool),
        scan_rows=rows,
        scan_columns=columns,
    )


class RewardContractTests(unittest.TestCase):
    def test_project_contracts_load(self):
        loco = load_loco_v2_rewards(ROOT / "configs" / "rewards_loco_v2.toml")
        nav = load_nav_v2_rewards(ROOT / "configs" / "rewards_nav_v2.toml")
        self.assertEqual(set(loco), set(LOCO_V2_FIELDS))
        self.assertEqual(set(nav), set(NAV_V2_FIELDS))
        self.assertGreater(loco["track_lin_vel_xy"]["weight"], 0)
        self.assertLess(loco["termination"]["weight"], 0)
        self.assertGreater(nav["reach_goal"]["weight"], 0)
        self.assertLess(nav["goal_braking"]["weight"], 0)

    def test_goal_completion_is_crossing_the_finish_line_not_stopping_or_aiming(self):
        # The task ends when the finish line is crossed anywhere across the
        # course. Asking for a standstill, or for alignment on the centre line,
        # were second requirements that a drifting policy can never satisfy.
        forward = torch.tensor([18.00, 19.50, 22.00, 19.50, 22.00])
        lateral = torch.tensor([0.00, 0.00, 0.00, 5.00, -4.50])
        crossed = finish_crossed(forward, lateral, 19.10, 4.0)
        self.assertEqual(crossed.tolist(), [False, True, True, False, False])
        with self.assertRaisesRegex(ValueError, "half_width"):
            finish_crossed(forward, lateral, 19.10, 0.0)

    def test_nav_contract_declares_the_real_course_finish_line(self):
        # The navigation task must own its finish geometry. Reading it from the
        # default locomotion contract is wrong: that table still describes the
        # short legacy course, so completion fired mid-course right after the
        # stairs instead of at the real goal.
        nav = load_nav_v2_rewards(ROOT / "configs" / "rewards_nav_v2.toml")
        finish = nav["reach_goal"]
        self.assertGreater(finish["half_width"], 0)
        # The finish must sit in the final stretch of the course, at or before
        # the navigation goal, and well past the stair exit at course x=9.10.
        self.assertGreater(finish["start_offset"], 15.0)
        self.assertLessEqual(finish["start_offset"], COURSE_GOAL_OFFSET)

    def test_wrong_sign_and_unknown_table_fail(self):
        source = (ROOT / "configs" / "rewards_nav_v2.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nav.toml"
            path.write_text(source + "\n[extra]\nweight = 1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_nav_v2_rewards(path)
            path.write_text(source.replace("weight = -2.0", "weight = 2.0"), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_nav_v2_rewards(path)

    def test_missing_table_fails(self):
        source = (ROOT / "configs" / "rewards_loco_v2.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loco.toml"
            start = source.index("[feet_slide]")
            end = source.index("[feet_stumble]")
            path.write_text(source[:start] + source[end:], encoding="utf-8")
            with self.assertRaises(ValueError):
                load_loco_v2_rewards(path)


class ScanGeometryTests(unittest.TestCase):
    def test_scan_grid_round_trip(self):
        scan = torch.arange(16 * 32, dtype=torch.float32).reshape(1, -1)
        grid, valid = scan_grid(scan, 16, 32)
        self.assertEqual(tuple(grid.shape), (1, 16, 32))
        self.assertTrue(bool(valid.all()))
        self.assertEqual(float(grid[0, 0, 0]), 0.0)
        self.assertEqual(float(grid[0, 0, 31]), 31.0)

    def test_scan_grid_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            scan_grid(torch.zeros(1, 10), 16, 32)

    def test_local_ground_ignores_invalid_rays(self):
        grid = torch.full((1, 4, 4), 5.0)
        valid = torch.zeros(1, 4, 4, dtype=torch.bool)
        valid[0, 0, 0] = True
        ground = local_ground(grid, valid)
        self.assertAlmostEqual(float(ground[0]), 5.0, places=5)

    def test_wall_score_rises_with_riser(self):
        flat = torch.full((1, 1, 4), -0.30)
        valid = torch.ones(1, 1, 4, dtype=torch.bool)
        ground = torch.full((1,), -0.30)
        low = wall_score(flat, valid, 0.03, 0.12, ground)
        riser = flat.clone()
        riser[0, 0, :] = 0.20
        high = wall_score(riser, valid, 0.03, 0.12, ground)
        self.assertLess(float(low[0]), 0.05)
        self.assertGreater(float(high[0]), 0.90)

    def test_flat_ground_stays_below_the_wall_threshold(self):
        cfg = load_nav_v2_rewards(ROOT / "configs" / "rewards_nav_v2.toml")["wall_proximity_brake"]
        flat = torch.full((1, 16, 32), -0.30)
        valid = torch.ones(1, 16, 32, dtype=torch.bool)
        ground = torch.full((1,), -0.30)
        score = wall_score(
            flat[:, :, : int(cfg["front_columns"])],
            valid[:, :, : int(cfg["front_columns"])],
            cfg["temperature"], cfg["height_threshold"], ground,
        )
        self.assertLess(float(score[0]), float(cfg["wall_score_threshold"]))

    def test_wall_score_invalid_rays_are_not_free_space(self):
        sector = torch.full((1, 1, 4), 0.20)
        valid = torch.zeros(1, 1, 4, dtype=torch.bool)
        ground = torch.full((1,), -0.30)
        score = wall_score(sector, valid, 0.06, 0.08, ground)
        self.assertEqual(float(score[0]), 0.0)


class LocoV2TermTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_loco_v2_rewards(ROOT / "configs" / "rewards_loco_v2.toml")

    def test_terms_are_finite_and_complete(self):
        terms = compute_loco_v2_terms(loco_inputs(), self.cfg)
        self.assertEqual(set(terms), set(LOCO_V2_FIELDS))
        for name, value in terms.items():
            self.assertTrue(torch.isfinite(value).all(), name)

    def test_perfect_tracking_scores_one(self):
        terms = compute_loco_v2_terms(loco_inputs(), self.cfg)
        self.assertAlmostEqual(float(terms["track_lin_vel_xy"][0]), 1.0, places=5)
        self.assertAlmostEqual(float(terms["track_ang_vel_z"][0]), 1.0, places=5)

    def test_static_command_keeps_foot_rewards_off(self):
        inputs = loco_inputs()
        inputs.command = torch.zeros(3, 3)
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertEqual(float(terms["feet_air_time"][0]), 0.0)
        self.assertEqual(float(terms["feet_clearance"][0]), 0.0)

    def test_stumble_penalty_only_for_large_lateral_force(self):
        inputs = loco_inputs()
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertEqual(float(terms["feet_stumble"][0]), 0.0)
        # legged_gym semantics: stumble when |f_xy| > ratio * |f_z|.
        inputs.foot_force_xy = torch.zeros(3, 4)
        inputs.foot_force_xy[:, 0] = 300.0
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertEqual(float(terms["feet_stumble"][0]), 1.0)
        inputs.foot_force_xy *= 0.5
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertEqual(float(terms["feet_stumble"][0]), 0.0)

    def test_terrain_raises_clearance_target(self):
        flat = loco_inputs(n=1, contacts=(True, False, False, False))
        flat.terrain_height_delta = torch.zeros(1)
        flat.foot_contacts = torch.tensor([[True, False, False, False]])
        flat.foot_z = torch.tensor([[-0.30, -0.21, -0.30, -0.30]])[0:1]
        low = compute_loco_v2_terms(flat, self.cfg)["feet_clearance"]

        raised = copy.deepcopy(flat)
        raised.foot_z = torch.tensor([[-0.30, -0.29, -0.30, -0.30]])[0:1]
        high = compute_loco_v2_terms(raised, self.cfg)["feet_clearance"]
        self.assertGreater(float(low[0]), float(high[0]))

    def test_action_magnitude_limit_is_zero_below_soft_limit(self):
        terms = compute_loco_v2_terms(loco_inputs(), self.cfg)
        self.assertEqual(float(terms["action_magnitude_limit"][0]), 0.0)
        inputs = loco_inputs()
        inputs.action = torch.full((3, 12), 3.4)
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertGreater(float(terms["action_magnitude_limit"][0]), 1.0)

    def test_termination_matches_failure_flag(self):
        inputs = loco_inputs()
        inputs.failure = torch.tensor([True, False, True])
        terms = compute_loco_v2_terms(inputs, self.cfg)
        self.assertEqual(terms["termination"].tolist(), [1.0, 0.0, 1.0])


class NavV2TrackerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_nav_v2_rewards(ROOT / "configs" / "rewards_nav_v2.toml")
        self.tracker = NavV2Tracker(2, "cpu", self.cfg)
        self.tracker.step(nav_inputs(2, distance=5.0))

    def test_terms_are_finite_and_complete(self):
        terms = self.tracker.step(nav_inputs(2, distance=5.0))
        self.assertEqual(set(terms), set(NAV_V2_FIELDS))
        for name, value in terms.items():
            self.assertTrue(torch.isfinite(value).all(), name)

    def test_first_sample_has_no_progress(self):
        tracker = NavV2Tracker(2, "cpu", self.cfg)
        terms = tracker.step(nav_inputs(2, distance=5.0))
        self.assertEqual(float(terms["path_progress"].sum()), 0.0)

    def test_approach_rewards_progress_and_retreat_penalises(self):
        forward = nav_inputs(2, distance=4.5)
        terms = self.tracker.step(forward)
        self.assertGreater(float(terms["path_progress"][0]), 0.0)

        retreat = nav_inputs(2, distance=6.0)
        retreat.lin_vel_b = torch.tensor([[-0.5, 0.0, 0.0]]).expand(2, 3).contiguous()
        retreat.command = torch.zeros(2, 3)
        terms = self.tracker.step(retreat)
        self.assertLess(float(terms["path_progress"][0]), 0.0)

    def test_reach_goal_fires_once(self):
        tracker = NavV2Tracker(1, "cpu", self.cfg)
        tracker.step(nav_inputs(1, distance=5.0))
        first = tracker.step(nav_inputs(1, distance=0.5))
        second = tracker.step(nav_inputs(1, distance=0.5))
        self.assertEqual(float(first["reach_goal"][0]), 1.0)
        self.assertEqual(float(second["reach_goal"][0]), 0.0)

    def test_settled_goal_requires_low_speed(self):
        moving = self.tracker.step(nav_inputs(2, distance=0.5))
        self.assertEqual(float(moving["settled_goal"][0]), 0.0)
        still = nav_inputs(2, distance=0.5)
        still.lin_vel_b = torch.zeros(2, 3)
        still.ang_vel_b = torch.zeros(2, 3)
        terms = self.tracker.step(still)
        self.assertEqual(float(terms["settled_goal"][0]), 1.0)

    def test_stuck_penalty_grows_then_resets_on_motion(self):
        tracker = NavV2Tracker(1, "cpu", self.cfg)
        stuck = nav_inputs(1, distance=5.0)
        stuck.lin_vel_b = torch.zeros(1, 3)
        tracker.step(nav_inputs(1, distance=5.0))
        early = tracker.step(stuck)["stuck_duration"]
        for _ in range(int(self.cfg["stuck_duration"]["grace_steps"]) + 5):
            late = tracker.step(stuck)["stuck_duration"]
        self.assertEqual(float(early[0]), 0.0)
        self.assertGreater(float(late[0]), 0.0)
        moving = tracker.step(nav_inputs(1, distance=5.0))
        self.assertEqual(float(moving["stuck_duration"][0]), 0.0)

    def test_braking_penalises_fast_approach_near_goal(self):
        tracker = NavV2Tracker(1, "cpu", self.cfg)
        tracker.step(nav_inputs(1, distance=3.0))
        fast = nav_inputs(1, distance=0.8)
        fast.lin_vel_b = torch.tensor([[0.6, 0.0, 0.0]])
        terms = tracker.step(fast)
        self.assertGreater(float(terms["goal_braking"][0]), 0.0)

    def test_wall_terms_react_to_front_riser(self):
        tracker = NavV2Tracker(1, "cpu", self.cfg)
        tracker.step(nav_inputs(1, distance=5.0, wall=False))
        open_terms = tracker.step(nav_inputs(1, distance=5.0, wall=False))
        wall_terms = tracker.step(nav_inputs(1, distance=5.0, wall=True))
        self.assertEqual(float(open_terms["wall_proximity_brake"][0]), 0.0)
        self.assertGreater(float(wall_terms["wall_proximity_brake"][0]), 0.0)
        self.assertGreater(float(wall_terms["obstacle_evasion"][0]), 0.0)

    def test_failure_and_contacts_are_reported(self):
        inputs = nav_inputs(2, distance=5.0)
        inputs.failure = torch.tensor([True, False])
        inputs.contact_count = torch.tensor([0.0, 3.0])
        terms = self.tracker.step(inputs)
        self.assertEqual(terms["termination"].tolist(), [1.0, 0.0])
        self.assertEqual(terms["undesired_contacts"].tolist(), [0.0, 1.0])

    def test_unsafe_tilt_and_recovery(self):
        tracker = NavV2Tracker(1, "cpu", self.cfg)
        tracker.step(nav_inputs(1, distance=5.0))
        tilted = nav_inputs(1, distance=5.0)
        tilted.roll = torch.tensor([0.8])
        terms = tracker.step(tilted)
        self.assertEqual(float(terms["unsafe_tilt"][0]), 1.0)
        # Tilting further is a regression, so the recovery signal is negative.
        self.assertLess(float(terms["tilt_recovery"][0]), 0.0)

        recovering = nav_inputs(1, distance=5.0)
        recovering.roll = torch.tensor([0.25])
        terms = tracker.step(recovering)
        self.assertEqual(float(terms["unsafe_tilt"][0]), 0.0)
        self.assertGreater(float(terms["tilt_recovery"][0]), 0.0)

    def test_reset_clears_persistent_state(self):
        self.tracker.reset(torch.tensor([0]))
        terms = self.tracker.step(nav_inputs(2, distance=4.0))
        self.assertEqual(float(terms["path_progress"][0]), 0.0)
        self.assertGreater(float(terms["path_progress"][1]), 0.0)

    def test_weighted_sum_matches_manual_combination(self):
        terms = self.tracker.step(nav_inputs(2, distance=5.0))
        expected = sum(float(self.cfg[name]["weight"]) * terms[name] for name in terms)
        torch.testing.assert_close(self.tracker.weighted_sum(terms), expected)


class EncoderModelTests(unittest.TestCase):
    def test_variant_detection(self):
        self.assertEqual(detect_actor_variant(EncoderActorCore("navigation").state_dict()), "v2")
        self.assertEqual(detect_actor_variant(ActorCore("navigation").state_dict()), "v1")

    def test_builders_match_variant(self):
        self.assertIsInstance(build_actor("navigation", EncoderActorCore("navigation").state_dict()),
                              EncoderActorCore)
        self.assertIsInstance(build_actor("navigation", ActorCore("navigation").state_dict()),
                              ActorCore)
        self.assertIsInstance(build_critic("locomotion", EncoderCriticCore("locomotion").state_dict()),
                              EncoderCriticCore)

    def test_navigation_actor_dimensions(self):
        actor = EncoderActorCore("navigation")
        self.assertEqual(actor.obs_dim, NAV_OBS_DIM)
        self.assertEqual(actor.num_actions, 3)
        self.assertEqual(actor.gru.input_size, 81)
        out, hidden = actor(torch.randn(4, NAV_OBS_DIM), torch.zeros(1, 4, 128))
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertEqual(tuple(hidden.shape), (1, 4, 128))

    def test_locomotion_actor_dimensions(self):
        actor = EncoderActorCore("locomotion")
        self.assertEqual(actor.obs_dim, LOCO_OBS_DIM)
        self.assertEqual(actor.num_actions, 12)
        out, _ = actor(torch.randn(2, LOCO_OBS_DIM), torch.zeros(1, 2, 128))
        self.assertEqual(tuple(out.shape), (2, 12))

    def test_latent_is_unit_norm(self):
        actor = EncoderActorCore("navigation")
        obs = torch.randn(5, NAV_OBS_DIM)
        values = obs[..., actor.prefix_dim:actor.prefix_dim + actor.num_geometry]
        confidence = obs[..., actor.prefix_dim + actor.num_geometry:actor.obs_dim]
        latent = actor.scan_encoder(torch.cat((values * confidence, confidence), dim=-1))
        torch.testing.assert_close(latent.norm(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_sequence_forward(self):
        actor = EncoderActorCore("navigation")
        out, hidden = actor(torch.randn(6, 4, NAV_OBS_DIM), torch.zeros(1, 4, 128))
        self.assertEqual(tuple(out.shape), (6, 4, 3))
        self.assertEqual(tuple(hidden.shape), (1, 4, 128))

    def test_wrong_dimensions_raise(self):
        actor = EncoderActorCore("navigation")
        with self.assertRaises(ValueError):
            actor(torch.randn(4, NAV_OBS_DIM + 1), torch.zeros(1, 4, 128))
        with self.assertRaises(ValueError):
            actor(torch.randn(4, NAV_OBS_DIM), torch.zeros(2, 4, 128))

    def test_adapter_act_and_evaluate(self):
        policy = HierActorCriticV2(
            {"policy": torch.zeros(4, NAV_OBS_DIM), "critic": torch.zeros(4, NAV_OBS_DIM + 7)},
            {"policy": ["policy"], "critic": ["critic"]}, 3, "navigation",
        )
        obs = {"policy": torch.randn(4, NAV_OBS_DIM), "critic": torch.randn(4, NAV_OBS_DIM + 7)}
        actions = policy.act(obs)
        self.assertEqual(tuple(actions.shape), (4, 3))
        policy.get_actions_log_prob(actions)
        self.assertEqual(tuple(policy.evaluate(obs).shape), (4, 1))
        # act_inference advances the recurrent state, so determinism is checked
        # by restoring the hidden state instead of calling it twice.
        hidden_before = policy.get_hidden_states()[0].clone()
        first = policy.act_inference(obs)
        policy.hidden_actor = hidden_before
        second = policy.act_inference(obs)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())
        policy.reset(torch.tensor([True, False, False, False]))
        self.assertEqual(float(policy.hidden_actor[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(policy.hidden_actor[:, 1].abs().sum()), 0.0)
        policy.update_normalization(obs)

    def test_adapter_rejects_bad_action_dimension(self):
        with self.assertRaises(ValueError):
            HierActorCriticV2(
                {"policy": torch.zeros(4, NAV_OBS_DIM), "critic": torch.zeros(4, NAV_OBS_DIM + 7)},
                {"policy": ["policy"], "critic": ["critic"]}, 12, "navigation",
            )


class RobustnessCurriculumTests(unittest.TestCase):
    def test_mix_thresholds(self):
        self.assertEqual(mix_for(0), {"clean": 1.0})
        self.assertEqual(mix_for(24), {"clean": 1.0})
        self.assertEqual(set(mix_for(25)), {"clean", "mild"})
        self.assertEqual(set(mix_for(120)), {"clean", "mild", "medium", "hard"})
        self.assertEqual(mix_for(5000), mix_for(300))

    def test_negative_iteration_fails(self):
        with self.assertRaises(ValueError):
            mix_for(-1)

    def test_mixes_sum_to_one(self):
        for iteration in (0, 25, 100, 250, 9999):
            self.assertAlmostEqual(sum(mix_for(iteration).values()), 1.0, places=6)

    def test_sample_shapes_and_ranges(self):
        curriculum = RobustnessCurriculum(64, "cpu", seed=3)
        batch = curriculum.sample(250)
        self.assertEqual(tuple(batch.uwb_noise_std.shape), (64,))
        self.assertEqual(tuple(batch.command_gain.shape), (64, 3))
        self.assertTrue((batch.uwb_noise_std >= 0).all())
        self.assertTrue((batch.uwb_dropout >= 0).all())
        self.assertTrue((batch.action_delay_steps >= 0).all())
        self.assertTrue((batch.action_delay_steps <= 1).all())
        self.assertEqual(len(batch.levels), 4)

    def test_clean_mix_produces_no_disturbance(self):
        curriculum = RobustnessCurriculum(32, "cpu", seed=5)
        batch = curriculum.sample(0)
        self.assertEqual(float(batch.uwb_noise_std.abs().sum()), 0.0)
        self.assertEqual(float(batch.uwb_dropout.abs().sum()), 0.0)
        torch.testing.assert_close(batch.command_gain, torch.ones(32, 3))

    def test_describe_reports_every_level(self):
        curriculum = RobustnessCurriculum(8, "cpu", seed=1)
        described = curriculum.describe(250)
        self.assertEqual(set(described), {f"robustness/{name}"
                                          for name in ("clean", "mild", "medium", "hard")})


if __name__ == "__main__":
    unittest.main()
