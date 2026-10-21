"""Use the installed RSL-RL PPO implementation with the project policy class."""

from copy import deepcopy
import math

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config

from ..checkpoints import load_checkpoint
from ..contracts import require_finite
from ..models import HierActorCritic


class CheckedPPO(PPO):
    def process_env_step(self, obs, rewards, dones, extras):
        require_finite(rewards, "PPO rewards")
        for key, value in obs.items():
            require_finite(value, f"PPO observation {key}")
        return super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        losses = super().update()
        def validate(value, name):
            if value is None:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    validate(item, f"{name}.{key}")
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    validate(item, f"{name}[{index}]")
                return
            if isinstance(value, torch.Tensor):
                require_finite(value, name)
                return
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"{name} is nonfinite: {value}")

        validate(losses, "PPO losses")
        for name, parameter in self.policy.named_parameters():
            require_finite(parameter, f"PPO parameter {name}")
        return losses


class HierOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", checkpoint_callback=None):
        self.checkpoint_callback = checkpoint_callback
        self.logger_type = train_cfg.get("logger", "tensorboard")
        super().__init__(env, deepcopy(train_cfg), log_dir, device)

    def _construct_algorithm(self, obs):
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        policy_class_name = policy_cfg.pop("class_name")
        if policy_class_name == "HierActorCritic":
            policy_class = HierActorCritic
        elif policy_class_name == "HierActorCriticV2":
            from ..hier_v2.model_v2 import HierActorCriticV2

            policy_class = HierActorCriticV2
        else:
            raise ValueError("hierarchical tasks require HierActorCritic or HierActorCriticV2")
        if algorithm_cfg.pop("class_name") != "PPO":
            raise ValueError("the supported reinforcement learning algorithm is PPO")
        if self.cfg.get("empirical_normalization"):
            raise ValueError("use the fixed deployment normalization in contracts.py")
        if algorithm_cfg.get("symmetry_cfg") is not None:
            raise ValueError("symmetry augmentation is not defined for the recurrent hierarchical interface")
        policy = policy_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
        ).to(self.device)
        algorithm = CheckedPPO(
            policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm_cfg
        )
        algorithm.init_storage(
            "rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions]
        )
        return algorithm

    def save(self, path, infos=None):
        if self.checkpoint_callback is None:
            from ..checkpoints import save_checkpoint

            stage = self.alg.policy.actor.stage
            save_checkpoint(
                path, "stage1" if stage == "locomotion" else "stage2",
                {stage: self.alg.policy.actor, "policy": self.alg.policy},
                optimizer=self.alg.optimizer,
                metadata={"iteration": self.current_learning_iteration, "infos": infos},
            )
        else:
            return self.checkpoint_callback(
                self, path, {"iteration": self.current_learning_iteration, "infos": infos})

    def load(self, path, load_optimizer=True):
        stage = self.alg.policy.actor.stage
        data = load_checkpoint(path, "stage1" if stage == "locomotion" else "stage2")
        if "policy" not in data["models"]:
            raise ValueError("on-policy checkpoint has no complete recurrent policy state")
        self.alg.policy.load_state_dict(data["models"]["policy"], strict=True)
        optimizer_state = data.get("optimizer")
        if load_optimizer:
            if optimizer_state is None:
                raise ValueError("checkpoint has no optimizer state for resume")
            self.alg.optimizer.load_state_dict(optimizer_state)
        iteration = data.get("metadata", {}).get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("checkpoint has no valid learning iteration")
        self.current_learning_iteration = iteration
        return data["metadata"].get("infos")
