import tempfile
import unittest
from pathlib import Path

import torch

from go1_hier_sim.checkpoints import state_hash
from go1_hier_sim.models import ActorCore, VisionModel
from go1_hier_sim.training import (DeploymentPerception, PerceptionDistillationRunner,
                                   SequenceReplay, perception_loss)


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)

    @staticmethod
    def sequence(kind="terrain", driver="teacher", n=2):
        policy_kind = "locomotion" if kind == "terrain" else "navigation"
        obs_dim = 1073 if kind == "terrain" else 193
        events = []
        camera_ticks = (0, 10)
        policy_ticks = (0, 4, 8, 12) if policy_kind == "locomotion" else (0, 20)
        timeline = [(tick, 0, "camera") for tick in camera_ticks]
        timeline += [(tick, 1, policy_kind) for tick in policy_ticks]
        for tick, _, event_kind in sorted(timeline):
            if event_kind == "camera":
                terrain = torch.cat((torch.rand(n, 512) * 2 - 1, torch.ones(n, 512)), -1)
                obstacle = torch.cat((torch.rand(n, 72), torch.ones(n, 72)), -1)
                events.append({"kind": "camera", "tick": tick,
                               "image": torch.rand(n, 2, 60, 80),
                               "state": torch.randn(n, 45),
                               "terrain": terrain, "obstacle": obstacle})
            else:
                events.append({"kind": event_kind, "tick": tick,
                               "teacher_obs": torch.randn(n, obs_dim)})
        return {
            "source": "simulator", "driver": driver,
            "initial": {"vision": torch.zeros(1, n, 256),
                        "teacher_policy": torch.zeros(1, n, 128),
                        "student_policy": torch.zeros(1, n, 128)},
            "events": events,
        }

    def replay(self, kind, driver):
        replay = SequenceReplay(capacity=2, max_bytes=16 * 1024 * 1024, seed=2)
        replay.append(self.sequence(kind, driver))
        return replay

    def test_masked_geometry_and_confidence_loss(self):
        target = torch.cat((torch.tensor([[0.5, 10.0]]).repeat(1, 36),
                            torch.tensor([[1.0, 0.0]]).repeat(1, 36)), -1)
        prediction = torch.cat((torch.tensor([[0.0, -10.0]]).repeat(1, 36),
                                torch.tensor([[0.8, 0.2]]).repeat(1, 36)), -1).requires_grad_()
        loss, info = perception_loss(prediction, target)
        loss.backward()
        self.assertEqual(info["valid_geometry_count"], 36)
        self.assertGreater(loss.item(), 0)

    def test_replay_is_bounded_real_and_round_trips(self):
        replay = self.replay("terrain", "teacher")
        replay.append(self.sequence("terrain", "teacher"))
        replay.append(self.sequence("terrain", "teacher"))
        self.assertEqual(len(replay), 2)
        self.assertLessEqual(replay.bytes_used, replay.max_bytes)
        bad = self.sequence()
        bad.pop("source")
        with self.assertRaisesRegex(ValueError, "real simulator"):
            replay.append(bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "replay.pt"
            replay.save(path)
            loaded = SequenceReplay.load(path, capacity=2, max_bytes=replay.max_bytes)
            self.assertEqual(len(loaded), 2)

    def test_stage3_updates_only_active_vision_and_checks_clocks(self):
        vision, policy = VisionModel(), ActorCore("locomotion")
        frozen_policy = state_hash(policy.state_dict())
        frozen_obstacle = state_hash(vision.obstacle.state_dict())
        active = state_hash(vision.terrain.state_dict())
        runner = PerceptionDistillationRunner(
            vision, policy, "stage3a", {"burn_in": 1, "train_frames": 1, "batch_size": 1})
        metrics = runner.update(self.replay("terrain", "teacher"))
        self.assertEqual(metrics["teacher_sequences"], 1)
        self.assertNotEqual(active, state_hash(vision.terrain.state_dict()))
        self.assertEqual(frozen_obstacle, state_hash(vision.obstacle.state_dict()))
        self.assertEqual(frozen_policy, state_hash(policy.state_dict()))
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))

        bad = self.sequence("terrain", "teacher")
        next(event for event in bad["events"] if event["kind"] == "locomotion" and event["tick"] == 4)["tick"] = 5
        broken = SequenceReplay()
        broken.append(bad)
        with self.assertRaisesRegex(ValueError, "required simulator clock"):
            runner.update(broken)

    def test_stage4_is_student_driven_and_exactly_half_teacher_replay(self):
        teacher = self.replay("obstacle", "teacher")
        student = self.replay("obstacle", "student")
        runner = PerceptionDistillationRunner(
            VisionModel(), ActorCore("navigation"), "stage4", branch="obstacle",
            config={"burn_in": 1, "train_frames": 1, "batch_size": 2},
            teacher_replay=teacher)
        metrics = runner.update(student)
        self.assertEqual((metrics["teacher_sequences"], metrics["student_sequences"]), (1, 1))
        with self.assertRaisesRegex(ValueError, "student-driven"):
            runner._sequence_loss(self.sequence("obstacle", "teacher"))

    def test_reset_is_per_environment_and_missing_camera_fails(self):
        provider = DeploymentPerception(VisionModel(), 2)
        with self.assertRaisesRegex(RuntimeError, "before a camera"):
            provider.terrain()
        provider.update_camera(torch.rand(2, 2, 60, 80), torch.randn(2, 45))
        untouched = provider.hidden_terrain[:, 1].clone()
        provider.reset(torch.tensor([0]))
        self.assertEqual(torch.count_nonzero(provider.hidden_terrain[:, 0]).item(), 0)
        torch.testing.assert_close(provider.hidden_terrain[:, 1], untouched)
        with self.assertRaisesRegex(RuntimeError, "before a camera"):
            provider.obstacle()

    def test_checkpoint_callback_receives_iteration_and_infos(self):
        calls = []
        runner = PerceptionDistillationRunner(
            VisionModel(), ActorCore("locomotion"), "stage3a",
            {"burn_in": 1, "train_frames": 1, "batch_size": 1},
            checkpoint_callback=lambda owner, path, metadata: calls.append((owner, path, metadata)))
        runner.iteration = 7
        result = runner.save("unused.pt", infos={"loss": 1.0})
        self.assertIsNone(result)
        self.assertIs(calls[0][0], runner)
        self.assertEqual(calls[0][2], {"iteration": 7, "branch": "terrain", "infos": {"loss": 1.0}})


if __name__ == "__main__":
    unittest.main()
