import tempfile
from pathlib import Path
import unittest

from go1_hier_sim.reward_config import load_stage1_rewards, require_matching_reward_config


ROOT = Path(__file__).resolve().parents[1]


class RewardConfigTests(unittest.TestCase):
    def test_project_reward_config_is_complete(self):
        rewards = load_stage1_rewards(ROOT / "configs" / "rewards_stage1.toml")
        self.assertEqual(len(rewards), 42)
        self.assertLess(rewards["joint_torque_l2"]["weight"], 0)
        self.assertEqual(rewards["feet_air_time"]["weight"], 0)
        self.assertEqual(rewards["step_frequency"]["weight"], 0)
        self.assertEqual(rewards["step_frequency"]["max_hz"], 4.0)
        self.assertEqual(rewards["four_feet_grounded"]["weight"], 0)
        self.assertEqual(rewards["foot_contact_cycle"]["max_air_time"], 0.65)
        self.assertEqual(rewards["air_time_variance"]["weight"], 0)
        self.assertEqual(rewards["air_time_variance"]["max_time"], 0.50)
        self.assertEqual(rewards["flat_trot_pattern"]["weight"], 0)
        self.assertLess(rewards["flat_trot_pattern"]["terrain_relief"], 0.5)
        self.assertEqual(rewards["foot_clearance"]["weight"], 0)
        self.assertEqual(rewards["phase_contact"]["flat_hz"], 2.40)
        self.assertEqual(rewards["phase_contact"]["terrain_hz"], 2.40)
        self.assertEqual(rewards["phase_contact"]["transition_width"], 0.12)
        self.assertEqual(rewards["phase_clearance"]["terrain_height"], 0.130)
        self.assertEqual(rewards["base_height"]["target_height"], 0.340)
        self.assertLess(rewards["foot_load_balance"]["weight"], 0)
        self.assertGreater(rewards["velocity_direction"]["weight"], 0)
        self.assertLess(rewards["cross_track_l2"]["weight"], 0)
        self.assertEqual(rewards["cross_track_l2"]["max_cost"], 4.0)
        self.assertEqual(rewards["goal_reached"]["weight"], 200.0)
        self.assertEqual(rewards["goal_reached"]["half_width"], 4.0)
        # The finish strip must sit at the end of the current course. It used to
        # hold the 16-metre-tile value 9.10, which is where the stairs end, so
        # completion fired in the middle of the route instead of at the goal.
        self.assertEqual(rewards["goal_reached"]["start_offset"], 19.10)
        self.assertEqual(rewards["goal_reached"]["end_offset"], 20.10)
        self.assertLessEqual(rewards["track_linear_velocity"]["weight"], 2.0)
        self.assertGreater(
            rewards["foot_clearance"]["terrain_target_height"],
            rewards["foot_clearance"]["target_height"],
        )
        self.assertGreater(
            rewards["foot_clearance"]["terrain_min_height"],
            rewards["foot_clearance"]["min_height"],
        )
        self.assertLess(rewards["joint_deviation_l2"]["weight"], 0)
        self.assertLess(rewards["base_roll_pitch_rate_l2"]["weight"], 0)
        self.assertLess(rewards["base_orientation_l2"]["weight"], 0)
        self.assertEqual(rewards["foot_lateral"]["max_outward_deviation"], 0.06)
        self.assertLess(rewards["hip_abduction_deviation_l2"]["weight"], 0)
        self.assertLess(rewards["hip_abduction_velocity_l2"]["weight"], 0)
        self.assertLess(rewards["small_steps"]["min_length"], rewards["small_steps"]["target_length"])
        self.assertLess(rewards["small_steps"]["target_length"], rewards["small_steps"]["max_length"])

    def test_unknown_or_wrong_sign_fails(self):
        source = (ROOT / "configs" / "rewards_stage1.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.toml"
            path.write_text(source + "\n[unknown]\nweight = 1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_stage1_rewards(path)
            path.write_text(source.replace("weight = -0.00020", "weight = 0.00020"), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_stage1_rewards(path)
            path.write_text(source.replace("terrain_relief = 0.30", "terrain_relief = 1.5", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_stage1_rewards(path)

    def test_resume_requires_the_exact_reward_contract(self):
        rewards = load_stage1_rewards(ROOT / "configs" / "rewards_stage1.toml")
        require_matching_reward_config({"reward_config": rewards}, rewards)
        with self.assertRaisesRegex(ValueError, "no reward_config"):
            require_matching_reward_config({}, rewards)
        changed = {name: dict(values) for name, values in rewards.items()}
        changed["feet_slide"]["weight"] = -0.2
        with self.assertRaisesRegex(ValueError, "differs"):
            require_matching_reward_config({"reward_config": changed}, rewards)


if __name__ == "__main__":
    unittest.main()
