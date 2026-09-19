"""Real tensor/model checks; optional RSL integration is not a simulation test."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from go1_hier_sim.checkpoints import save_checkpoint, state_hash
from go1_hier_sim.export import export_checkpoint
from go1_hier_sim.models import (
    ActorCore,
    CriticCore,
    HierActorCritic,
    VisionBranch,
    VisionModel,
    migrate_locomotion_heading_state,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    @staticmethod
    def observations(stage, n=3):
        width = 1073 if stage == "locomotion" else 193
        return {"policy": torch.randn(n, width), "critic": torch.randn(n, width + 7)}

    def policy(self, stage="locomotion", n=3):
        return HierActorCritic(self.observations(stage, n),
                               {"policy": ["policy"], "critic": ["critic"]},
                               12 if stage == "locomotion" else 3, stage)

    def test_stage_dimensions_and_separate_critic(self):
        for stage, width, critic_width, encoded, actions in (("locomotion", 1073, 1080, 81, 12),
                                                               ("navigation", 193, 200, 81, 3)):
            with self.subTest(stage=stage):
                actor, critic = ActorCore(stage), CriticCore(stage)
                self.assertEqual(actor.obs_dim, width)
                self.assertEqual(critic.actor_dim + 7, critic_width)
                self.assertEqual(actor.gru.input_size, encoded)
                self.assertEqual(critic.gru.input_size, encoded + 7)
                out, hidden = actor(torch.randn(4, 3, width), torch.zeros(1, 3, 128))
                value, _ = critic(torch.randn(4, 3, width + 7), torch.zeros(1, 3, 128))
                self.assertEqual(out.shape, (4, 3, actions))
                self.assertEqual(value.shape, (4, 3, 1))
                self.assertEqual(hidden.shape, (1, 3, 128))
                self.assertTrue({id(p) for p in actor.parameters()}.isdisjoint(
                    {id(p) for p in critic.parameters()}))

    def test_actor_sequence_matches_step_execution(self):
        for stage, width in (("locomotion", 1073), ("navigation", 193)):
            model = ActorCore(stage).eval()
            obs, initial = torch.randn(5, 3, width), torch.randn(1, 3, 128)
            with torch.no_grad():
                expected, expected_hidden = model(obs, initial)
                hidden, steps = initial.clone(), []
                for frame in obs:
                    output, hidden = model(frame, hidden)
                    steps.append(output)
            torch.testing.assert_close(torch.stack(steps), expected)
            torch.testing.assert_close(hidden, expected_hidden)

    def test_unobserved_geometry_values_cannot_change_decisions(self):
        for stage in ("locomotion", "navigation"):
            for model in (ActorCore(stage), CriticCore(stage)):
                width = model.obs_dim if isinstance(model, ActorCore) else model.actor_dim + 7
                obs = torch.randn(3, width)
                start, count = model.prefix_dim, model.num_geometry
                obs[:, start + count:start + 2 * count] = 0
                changed = obs.clone()
                changed[:, start:start + count] += 10
                hidden = torch.randn(1, 3, 128)
                for left, right in zip(model(obs, hidden), model(changed, hidden)):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_geometry_encoder_receives_masked_values_and_confidence(self):
        for stage in ("locomotion", "navigation"):
            for model in (ActorCore(stage), CriticCore(stage)):
                width = model.obs_dim if isinstance(model, ActorCore) else model.actor_dim + 7
                obs = torch.randn(2, width)
                start, count = model.prefix_dim, model.num_geometry
                obs[:, start + count:start + 2 * count] = torch.rand(2, count)
                captured = []
                handle = model.encoder[0].register_forward_pre_hook(
                    lambda _module, inputs: captured.append(inputs[0].detach().clone())
                )
                try:
                    model(obs, torch.zeros(1, 2, 128))
                finally:
                    handle.remove()
                values = obs[:, start:start + count]
                confidence = obs[:, start + count:start + 2 * count]
                torch.testing.assert_close(
                    captured[0].squeeze(0), torch.cat((values * confidence, confidence), dim=-1),
                    rtol=0, atol=0,
                )

    def test_runtime_state_reset_and_saved_states_are_independent(self):
        policy = self.policy()
        obs = self.observations("locomotion")
        policy.act(obs)
        policy.evaluate(obs)
        saved_actor, saved_critic = policy.get_hidden_states()
        policy.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(policy.hidden_actor[:, [0, 2]], saved_actor[:, [0, 2]])
        torch.testing.assert_close(policy.hidden_critic[:, [0, 2]], saved_critic[:, [0, 2]])
        self.assertEqual(torch.count_nonzero(policy.hidden_actor[:, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(policy.hidden_critic[:, 1]).item(), 0)
        self.assertGreater(torch.count_nonzero(saved_actor[:, 1]).item(), 0)
        self.assertGreater(torch.count_nonzero(saved_critic[:, 1]).item(), 0)
        policy.reset()
        self.assertEqual(torch.count_nonzero(policy.hidden_actor).item(), 0)
        self.assertEqual(torch.count_nonzero(policy.hidden_critic).item(), 0)

    def test_action_noise_is_bounded(self):
        policy, obs = self.policy(), self.observations("locomotion")
        with torch.no_grad():
            policy.log_std.fill_(100.0)
        policy.act(obs)
        self.assertLessEqual(policy.action_std.max().item(), 1.5 + 1e-6)
        with torch.no_grad():
            policy.log_std.fill_(-100.0)
        policy.act(obs)
        self.assertGreaterEqual(policy.action_std.min().item(), 0.05 - 1e-6)

    def test_visual_sequence_memory_and_branch_independence(self):
        model = VisionModel()
        self.assertTrue({id(p) for p in model.terrain.parameters()}.isdisjoint(
            {id(p) for p in model.obstacle.parameters()}))
        image, state = torch.rand(3, 2, 2, 60, 80), torch.randn(3, 2, 45)
        for branch in (model.terrain, model.obstacle):
            initial = torch.randn(1, 2, 256)
            output, expected_hidden = branch(image, state, initial)
            hidden, steps = initial.clone(), []
            for frame, proprio in zip(image, state):
                step, hidden = branch(frame, proprio, hidden)
                steps.append(step)
            torch.testing.assert_close(torch.stack(steps), output)
            torch.testing.assert_close(hidden, expected_hidden)
            self.assertEqual(output.shape, (3, 2, branch.num_values * 2))
            values = output[..., :branch.num_values]
            if branch.kind == "terrain":
                self.assertTrue(((values > -1) & (values < 1)).all())
            else:
                self.assertTrue(((values > 0) & (values < 1)).all())
            self.assertTrue(((output[..., branch.num_values:] > 0) &
                             (output[..., branch.num_values:] < 1)).all())
            hidden[:, 0] = 0
            next_output, _ = branch(image[0], state[0], hidden)
            isolated, _ = branch(image[0, 0:1], state[0, 0:1], torch.zeros(1, 1, 256))
            torch.testing.assert_close(next_output[0:1], isolated)

    def test_frozen_decisions_backpropagate_into_each_visual_branch(self):
        for kind, stage, prefix in (("terrain", "locomotion", 49),
                                     ("obstacle", "navigation", 49)):
            vision, actor = VisionBranch(kind), ActorCore(stage).eval().requires_grad_(False)
            initial_actor_hash, initial_vision_hash = state_hash(actor.state_dict()), state_hash(vision.state_dict())
            optimizer = torch.optim.Adam(vision.parameters(), lr=1e-4)
            geometry, _ = vision(torch.rand(2, 2, 60, 80), torch.randn(2, 45), torch.zeros(1, 2, 256))
            action, _ = actor(torch.cat((torch.randn(2, prefix), geometry), -1), torch.zeros(1, 2, 128))
            loss = (action - torch.ones_like(action)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in vision.cnn.parameters()))
            self.assertTrue(all(p.grad is None for p in actor.parameters()))
            optimizer.step()
            self.assertEqual(initial_actor_hash, state_hash(actor.state_dict()))
            self.assertNotEqual(initial_vision_hash, state_hash(vision.state_dict()))

    def test_schema3_heading_migration_preserves_and_shifts_inputs(self):
        actor_rows = 3 * 128
        legacy = {
            "actor.gru.weight_ih_l0": torch.arange(actor_rows * 79).reshape(actor_rows, 79),
            "critic.gru.weight_ih_l0": torch.arange(actor_rows * 86).reshape(actor_rows, 86),
            "actor.head.0.bias": torch.randn(256),
        }
        migrated = migrate_locomotion_heading_state(legacy)
        self.assertEqual(migrated["actor.gru.weight_ih_l0"].shape, (actor_rows, 81))
        self.assertEqual(migrated["critic.gru.weight_ih_l0"].shape, (actor_rows, 88))
        for name in ("actor.gru.weight_ih_l0", "critic.gru.weight_ih_l0"):
            torch.testing.assert_close(migrated[name][:, :47], legacy[name][:, :47])
            self.assertEqual(torch.count_nonzero(migrated[name][:, 47:49]).item(), 0)
            torch.testing.assert_close(migrated[name][:, 49:], legacy[name][:, 47:])
        self.assertIs(migrated["actor.head.0.bias"], legacy["actor.head.0.bias"])

    def test_bad_interfaces_and_nonfinite_observations_fail(self):
        with self.assertRaisesRegex(ValueError, "actor observation"):
            ActorCore()(torch.randn(3, 235), torch.zeros(1, 3, 128))
        with self.assertRaisesRegex(ValueError, "critic hidden"):
            CriticCore("navigation")(torch.randn(3, 200), torch.zeros(1, 2, 128))
        with self.assertRaisesRegex(ValueError, "visual hidden"):
            VisionBranch("terrain")(torch.rand(3, 2, 60, 80), torch.randn(3, 45), torch.zeros(1, 3, 128))
        with self.assertRaisesRegex(ValueError, "image"):
            VisionBranch("terrain")(torch.rand(3, 2, 80, 60), torch.randn(3, 45), torch.zeros(1, 3, 256))
        policy, obs = self.policy(), self.observations("locomotion")
        obs["policy"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            policy.update_normalization(obs)
        with self.assertRaisesRegex(ValueError, "reset mask"):
            policy.reset(torch.ones(2))

    def test_checkpoint_exports_include_encoders_and_dynamic_sequences(self):
        cases = (("stage1", "locomotion", ActorCore("locomotion")),
                 ("stage2", "navigation", ActorCore("navigation")),
                 ("stage3b", "vision", VisionModel()))
        with tempfile.TemporaryDirectory() as folder:
            for stage, name, model in cases:
                checkpoint = Path(folder) / f"{stage}.pt"
                save_checkpoint(checkpoint, stage, {name: model})
                exported = torch.jit.load(str(export_checkpoint(checkpoint, Path(folder) / stage)))
                self.assertEqual(set(model.state_dict()), set(exported.state_dict()))
                for n in (1, 4):
                    for sequence in (False, True):
                        shape = (3, n) if sequence else (n,)
                        if name == "vision":
                            inputs = (torch.rand(*shape, 2, 60, 80), torch.randn(*shape, 45),
                                      torch.randn(1, n, 256), torch.randn(1, n, 256))
                        else:
                            inputs = (torch.randn(*shape, model.obs_dim), torch.randn(1, n, 128))
                        with torch.no_grad():
                            expected, actual = model(*inputs), exported(*inputs)
                        for left, right in zip(expected, actual):
                            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("rsl_rl") is not None and
                         importlib.util.find_spec("tensordict") is not None,
                         "RSL-RL/tensordict unavailable; real PPO update and unpadding remain unverified")
    def test_rsl_padded_sequences_ignore_padding_and_preserve_rollout_memory(self):
        from tensordict import TensorDict

        policy = self.policy("navigation", 2)
        masks = torch.tensor([[True, True, True], [True, True, True],
                              [True, False, False], [True, False, False]])
        observations = TensorDict({
            "policy": torch.randn(4, 3, 193),
            "critic": torch.randn(4, 3, 200),
        }, batch_size=[4, 3])
        hidden_actor = torch.randn(1, 3, 128)
        hidden_critic = torch.randn(1, 3, 128)
        saved_states = policy.get_hidden_states()

        policy.act(observations, masks=masks, hidden_state=hidden_actor)
        actor_mean = policy.action_mean.clone()
        critic_value = policy.evaluate(observations, masks=masks, hidden_state=hidden_critic)

        changed = observations.clone()
        changed["policy"][~masks] = 1.0e6
        changed["critic"][~masks] = -1.0e6
        policy.act(changed, masks=masks, hidden_state=hidden_actor)
        torch.testing.assert_close(policy.action_mean, actor_mean, rtol=0, atol=0)
        torch.testing.assert_close(
            policy.evaluate(changed, masks=masks, hidden_state=hidden_critic),
            critic_value, rtol=0, atol=0,
        )
        self.assertEqual(actor_mean.shape, (4, 2, 3))
        self.assertEqual(critic_value.shape, (4, 2, 1))
        for before_state, after_state in zip(saved_states, policy.get_hidden_states()):
            torch.testing.assert_close(before_state, after_state)

    @unittest.skipUnless(importlib.util.find_spec("rsl_rl") is not None and
                         importlib.util.find_spec("tensordict") is not None,
                         "RSL-RL/tensordict unavailable; real PPO update and unpadding remain unverified")
    def test_real_rsl_ppo_update_with_reset_trajectories(self):
        from tensordict import TensorDict
        from go1_hier_sim.training.on_policy import CheckedPPO

        for stage in ("locomotion", "navigation"):
            n, steps = 4, 8
            policy = self.policy(stage, n)
            obs = TensorDict(self.observations(stage, n), batch_size=[n])
            algo = CheckedPPO(policy, num_learning_epochs=2, num_mini_batches=2,
                              learning_rate=3e-4, device="cpu")
            algo.init_storage("rl", n, steps, obs, [policy.actor.num_actions])
            before = state_hash(policy.state_dict())
            with torch.inference_mode():
                for step in range(steps):
                    action = algo.act(obs)
                    rewards = -(action.square().mean(-1))
                    dones = torch.tensor([step == 2, step == 4, step == 6, False])
                    obs = TensorDict(self.observations(stage, n), batch_size=[n])
                    algo.process_env_step(obs, rewards, dones, {})
                algo.compute_returns(obs)
            saved_states = policy.get_hidden_states()
            losses = algo.update()
            self.assertTrue(losses)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
            self.assertNotEqual(before, state_hash(policy.state_dict()))
            for before_state, after_state in zip(saved_states, policy.get_hidden_states()):
                torch.testing.assert_close(before_state, after_state)


if __name__ == "__main__":
    unittest.main()
