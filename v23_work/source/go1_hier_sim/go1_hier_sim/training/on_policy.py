"""Use the installed RSL-RL PPO implementation with the project policy class."""

from copy import deepcopy
import math
import os

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config

from ..checkpoints import load_checkpoint
from ..contracts import require_finite
from ..models import HierActorCritic


def snapshot_policy_mask(mask):
    """Capture the action source for exactly one transition, not a live view.

    The environment resets some slots inside ``step`` and then reuses its mask
    buffer; PPO must retain each transition's own action-source bit.
    """
    if mask.ndim != 1:
        raise ValueError("policy action-source mask must have shape [envs]")
    return mask.detach().clone()


def zero_masked_advantages(advantages, mask):
    """Drop the advantages of samples the gate did not take from this policy.

    ``advantages`` is [steps, envs, 1] and ``mask`` is [steps, envs]. A zero
    mask entry means the environment applied another policy's action for that
    sample, so its advantage is set to zero: the surrogate loss then has a zero
    gradient there and the leap actor only learns from its own leaps, while the
    critic still fits the value of every state for bootstrapping.
    """
    if mask.shape != advantages.shape[:2]:
        raise ValueError("advantage mask must match the advantage batch shape")
    return advantages.masked_fill(mask.unsqueeze(-1) <= 0.0, 0.0)


def recurrent_distill_epoch(actor, obs_seq, dones, teacher_seq, valid, *, lam):
    """One supervised epoch of DAgger teacher labels along the rollout recurrence.

    Hidden state advances with the actual visited trajectory and zeroes exactly
    where an episode ended, mirroring deployment rollout semantics.
    """
    if obs_seq.ndim != 3 or obs_seq.shape[:2] != teacher_seq.shape[:2]:
        raise ValueError("distill batch shapes must be [steps, envs, ...] aligned")
    steps, envs = obs_seq.shape[0], obs_seq.shape[1]
    if dones.shape != (steps, envs) or valid.shape != (steps, envs):
        raise ValueError("distill dones/valid must match [steps, envs]")
    # Rollout tensors are created under inference mode; re-materialize them so
    # the supervised pass can build an autograd graph.
    obs_seq = obs_seq.detach().clone()
    teacher_seq = teacher_seq.detach().clone()
    dones = dones.detach().clone()
    valid = valid.detach().clone()
    hidden = torch.zeros(1, envs, 128, device=obs_seq.device)
    total = obs_seq.new_zeros(())
    count = 0
    for t in range(steps):
        mean, hidden = actor(obs_seq[t], hidden)
        rows = valid[t]
        if bool(rows.any()):
            squared = (mean[rows] - teacher_seq[t][rows]).square().mean(dim=-1)
            total = total + lam * squared.sum()
            count += int(rows.sum())
        hidden = torch.where(dones[t][None, :, None], torch.zeros_like(hidden), hidden)
    if count == 0:
        return obs_seq.new_zeros((), requires_grad=True), 0
    return total / count, count


class CheckedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy_mask_steps = []
        self._teacher_steps = []
        self._teacher_batch = None
        self.distill_lambda = float(os.environ.get("GO1_V29_LAMBDA", "0.5"))

    def process_env_step(self, obs, rewards, dones, extras):
        require_finite(rewards, "PPO rewards")
        for key, value in obs.items():
            require_finite(value, f"PPO observation {key}")
        mask = extras.get("policy_mask")
        self._policy_mask_steps.append(None if mask is None else snapshot_policy_mask(mask))
        teacher = extras.get("teacher_mean")
        teacher_valid = extras.get("teacher_valid")
        if (teacher is None) != (teacher_valid is None):
            raise ValueError("teacher_mean and teacher_valid must arrive together")
        if teacher is not None:
            require_finite(teacher, "teacher action labels")
            if teacher.ndim != 2:
                raise ValueError("teacher labels must be per-environment [envs, actions]")
            self._teacher_steps.append((teacher.detach().clone(),
                                        teacher_valid.detach().clone().bool()))
        return super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        super().compute_returns(obs)
        masks = self._policy_mask_steps
        self._policy_mask_steps = []
        if masks and all(mask is not None for mask in masks):
            stacked = torch.stack(masks, dim=0).to(self.storage.advantages.device)
            if stacked.shape == self.storage.advantages.shape[:2] and bool((stacked <= 0.0).any()):
                if os.environ.get("GO1_DEBUG_MASK") == "1":
                    share = float((stacked > 0.0).to(torch.float32).mean().item())
                    print("MASK", "kept_samples=%.4f" % share, flush=True)
                self.storage.advantages.copy_(
                    zero_masked_advantages(self.storage.advantages, stacked))
        steps = self._teacher_steps
        self._teacher_steps = []
        if steps and all(mean is not None for mean, _ in steps):
            self._teacher_batch = (
                torch.stack([mean for mean, _ in steps], dim=0).to(self.storage.device),
                torch.stack([valid for _, valid in steps], dim=0).to(self.storage.device))

    def update(self):
        losses = super().update()
        if self._teacher_batch is not None:
            teacher, valid = self._teacher_batch
            self._teacher_batch = None
            if any(p.requires_grad for p in self.policy.actor.parameters()):
                actor_obs = torch.cat([self.storage.observations[group]
                                       for group in self.policy.obs_groups["policy"]], dim=-1)
                dones = self.storage.dones.squeeze(-1).bool().to(actor_obs.device)
                loss, count = recurrent_distill_epoch(
                    self.policy.actor, actor_obs, dones, teacher, valid, lam=self.distill_lambda)
                if count:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                   getattr(self, "max_grad_norm", 1.0))
                    self.optimizer.step()
                    losses["distill_loss"] = float(loss.detach().item())
                    losses["distill_samples"] = float(count)
                    print("V29_DISTILL samples=%d loss=%.6f" % (count, losses["distill_loss"]),
                          flush=True)
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

    def _configure_multi_gpu(self) -> None:
        """Explicit Gloo opt-in for tiny /dev/shm; default NCCL remains unchanged.

        The older v9-only flag still requires v9 contact mode. A separate
        transport-only flag lets other terrain teachers use four CUDA ranks
        without activating the uncalibrated v9 termination path.
        """
        generic = os.environ.get("GO1_DDP_BACKEND")
        legacy = os.environ.get("GO1_V9_DDP_BACKEND")
        if generic is not None and legacy is not None:
            raise ValueError("do not set both DDP backend flags")
        if generic is not None:
            if generic != "gloo":
                raise ValueError("GO1_DDP_BACKEND only supports explicit gloo")
        elif legacy is None:
            return super()._configure_multi_gpu()
        elif legacy != "gloo" or os.environ.get("GO1_V9_GAP_CONTACT") != "1":
            raise ValueError("GO1_V9_DDP_BACKEND only supports v9 gap-contact Gloo")
        self.gpu_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return
        self.gpu_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.environ.get("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match local rank '{self.gpu_local_rank}'")
        if not 0 <= self.gpu_local_rank < self.gpu_world_size or not 0 <= self.gpu_global_rank < self.gpu_world_size:
            raise ValueError("distributed rank must fall within world size")
        torch.distributed.init_process_group(backend="gloo", rank=self.gpu_global_rank,
                                             world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

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
        elif policy_class_name == "HierActorCriticV3":
            from ..hier_v2.map_attention_v3 import HierActorCriticV3

            policy_class = HierActorCriticV3
        else:
            raise ValueError("hierarchical tasks require HierActorCritic v1, v2 or v3")
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
