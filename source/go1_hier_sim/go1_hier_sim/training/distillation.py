"""Event-accurate temporal perception distillation."""

from collections import deque
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import math
import random
import tempfile

import torch
from torch.nn import functional as F

from ..checkpoints import save_checkpoint, state_hash
from ..contracts import interface_manifest, require_finite


_STAGE_SPEC = {
    "stage3a": ("terrain", "locomotion", "teacher", 1.0e-4),
    "stage3b": ("obstacle", "navigation", "teacher", 1.0e-4),
}


def perception_loss(prediction, target):
    """Unknown geometry contributes no regression loss, but supervises confidence."""
    if prediction.shape != target.shape or prediction.shape[-1] not in (144, 512):
        raise ValueError("perception prediction and target must share the geometry contract")
    require_finite(prediction, "perception prediction")
    require_finite(target, "perception target")
    count = prediction.shape[-1] // 2
    validity = target[..., count:]
    if ((validity < 0) | (validity > 1)).any():
        raise ValueError("geometry validity must lie in [0,1]")
    predicted_validity = prediction[..., count:]
    if ((predicted_validity < 0) | (predicted_validity > 1)).any():
        raise ValueError("predicted confidence must lie in [0,1]")
    weight = validity.sum()
    errors = F.smooth_l1_loss(prediction[..., :count], target[..., :count], reduction="none")
    geometry = (errors * validity).sum() / weight.clamp_min(1)
    confidence = F.binary_cross_entropy(predicted_validity, validity)
    return geometry + 0.2 * confidence, {
        "geometry_loss": float(geometry.detach()) if weight.item() else None,
        "confidence_loss": float(confidence.detach()),
        "valid_geometry_count": int((validity > 0).sum().item()),
    }


def _copy_tree(value, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: _copy_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item, device) for item in value)
    return deepcopy(value)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _validate_event_sequence(sequence):
    if not isinstance(sequence, dict) or not sequence.get("events") or "initial" not in sequence:
        raise ValueError("a replay sequence needs events and initial recurrent state")
    if sequence.get("source") != "simulator":
        raise ValueError("replay source must be the real simulator event stream")
    if sequence.get("driver") not in {"teacher", "student"}:
        raise ValueError("replay must identify its actual teacher or student driver")
    last_tick = -1
    camera_count = 0
    for event in sequence["events"]:
        if not isinstance(event, dict) or event.get("kind") not in {"camera", "locomotion", "navigation", "reset"}:
            raise ValueError("unknown replay event")
        tick = event.get("tick")
        if not isinstance(tick, int) or tick < last_tick:
            raise ValueError("replay physics ticks must be monotonic nonnegative integers")
        last_tick = tick
        if event["kind"] == "camera":
            camera_count += 1
            for key in ("image", "state", "terrain", "obstacle"):
                if key not in event or not isinstance(event[key], torch.Tensor):
                    raise ValueError(f"camera event is missing real {key}")
        elif event["kind"] in {"locomotion", "navigation"}:
            if "teacher_obs" not in event or not isinstance(event["teacher_obs"], torch.Tensor):
                raise ValueError("policy event is missing the teacher observation from the actual state")
        elif "done" not in event or not isinstance(event["done"], torch.Tensor):
            raise ValueError("reset event is missing its per-environment mask")
    if not camera_count:
        raise ValueError("replay sequence contains no real camera samples")


class SequenceReplay:
    """Bounded CPU replay of real simulator events and recurrent starting states."""

    def __init__(self, capacity=256, max_bytes=512 * 1024 * 1024, metadata=None, seed=0):
        if capacity <= 0 or max_bytes <= 0:
            raise ValueError("replay capacity and byte budget must be positive")
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self.metadata = dict(metadata or {})
        self.sequences = deque()
        self.bytes_used = 0
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.sequences)

    def append(self, sequence):
        _validate_event_sequence(sequence)
        data = _copy_tree(sequence)
        size = _tensor_bytes(data)
        if size > self.max_bytes:
            raise ValueError("a sequence exceeds the replay byte budget")
        while self.sequences and (len(self.sequences) >= self.capacity or self.bytes_used + size > self.max_bytes):
            self.bytes_used -= _tensor_bytes(self.sequences.popleft())
        self.sequences.append(data)
        self.bytes_used += size

    def sample(self, count, driver=None):
        if count <= 0:
            raise ValueError("replay sample count must be positive")
        population = list(self.sequences)
        if driver is not None:
            if driver not in {"teacher", "student"}:
                raise ValueError("unknown replay driver")
            population = [sequence for sequence in population if sequence["driver"] == driver]
        if not population:
            raise ValueError("cannot sample an empty replay partition")
        return [population[self.random.randrange(len(population))] for _ in range(count)]

    def save(self, path):
        if not self.sequences:
            raise ValueError("refusing to save empty replay")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save({"format": "go1_event_replay_v1", "interfaces": interface_manifest(),
                        "metadata": self.metadata, "sequences": list(self.sequences)}, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, expected_metadata=None, **kwargs):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"required teacher replay is missing: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != "go1_event_replay_v1" or data.get("interfaces") != interface_manifest():
            raise ValueError("teacher replay format or observation interfaces differ")
        metadata = data.get("metadata", {})
        for key, expected in (expected_metadata or {}).items():
            if metadata.get(key) != expected:
                raise ValueError(f"teacher replay provenance mismatch: {key}")
        replay = cls(metadata=metadata, **kwargs)
        for sequence in data.get("sequences", []):
            replay.append(sequence)
        if not len(replay):
            raise ValueError("teacher replay contains no sequences")
        return replay


def _module_hash(module):
    return state_hash(module.state_dict())


def _reset_hidden(hidden, done):
    if done.ndim != 1 or done.shape[0] != hidden.shape[1]:
        raise ValueError("reset mask does not match recurrent batch")
    return hidden * (~done.bool()).to(hidden.dtype).view(1, -1, 1)


class PerceptionDistillationRunner:
    """Train one visual branch while downstream and teacher weights stay frozen.

    Camera events carry real frames, 45-field state and both geometry labels.
    Policy events carry the teacher observation sampled from the same actual
    simulator trajectory.  The three initial hidden states are explicit so
    teacher, student and vision memory can never alias.
    """

    def __init__(self, vision, policy, stage, config=None, *, branch=None, device="cpu",
                 teacher_replay=None, checkpoint_callback=None):
        cfg = dict(config or {})
        if stage in _STAGE_SPEC:
            expected_branch, policy_kind, driver, default_lr = _STAGE_SPEC[stage]
            if branch is not None and branch != expected_branch:
                raise ValueError(f"{stage} trains the {expected_branch} branch")
            branch = expected_branch
        elif stage == "stage4":
            if branch not in {"terrain", "obstacle"}:
                raise ValueError("stage4 requires branch='terrain' or branch='obstacle'")
            policy_kind = "locomotion" if branch == "terrain" else "navigation"
            driver, default_lr = "student", 1.0e-5
        else:
            raise ValueError(f"unknown distillation stage: {stage}")
        if getattr(policy, "stage", None) != policy_kind:
            raise ValueError(f"{stage} requires a frozen {policy_kind} ActorCore")

        self.vision = vision.to(device)
        self.policy = policy.to(device).eval().requires_grad_(False)
        self.stage, self.branch, self.policy_kind, self.driver = stage, branch, policy_kind, driver
        self.device = torch.device(device)
        self.burn_in = int(cfg.get("burn_in", 16))
        self.train_frames = int(cfg.get("train_frames", 32))
        self.batch_size = int(cfg.get("batch_size", 8))
        self.geometry_weight = float(cfg.get("geometry_weight", 1.0))
        self.validity_weight = float(cfg.get("validity_weight", 0.2))
        self.action_weight = float(cfg.get("action_weight", 1.0))
        learning_rate = float(cfg.get(
            "learning_rate_stage4" if stage == "stage4" else "learning_rate_stage3", default_lr))
        values = (self.geometry_weight, self.validity_weight, self.action_weight, learning_rate)
        if self.burn_in < 0 or self.train_frames <= 0 or self.batch_size <= 0:
            raise ValueError("invalid distillation sequence or batch configuration")
        if not all(math.isfinite(value) and value >= 0 for value in values) or learning_rate <= 0:
            raise ValueError("distillation weights and learning rate must be finite and valid")
        if stage == "stage4" and self.batch_size % 2:
            raise ValueError("stage4 batch_size must be even for exact 1:1 replay mixing")
        if stage == "stage4" and teacher_replay is None:
            raise ValueError("stage4 requires bounded teacher replay for exact 1:1 mixing")
        self.teacher_replay = teacher_replay
        self.checkpoint_callback = checkpoint_callback
        self.iteration = 0

        self.active = getattr(self.vision, branch)
        self.frozen_branch = getattr(self.vision, "obstacle" if branch == "terrain" else "terrain")
        self.active.train().requires_grad_(True)
        self.frozen_branch.eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.active.parameters(), lr=learning_rate)
        self._policy_hash = _module_hash(self.policy)
        self._frozen_branch_hash = _module_hash(self.frozen_branch)

    def _assert_frozen(self):
        if _module_hash(self.policy) != self._policy_hash:
            raise RuntimeError("frozen downstream policy weights or buffers changed")
        if _module_hash(self.frozen_branch) != self._frozen_branch_hash:
            raise RuntimeError("inactive visual branch weights or buffers changed")
        if any(parameter.requires_grad for parameter in self.policy.parameters()):
            raise RuntimeError("downstream policy was unfrozen")
        if any(parameter.requires_grad for parameter in self.frozen_branch.parameters()):
            raise RuntimeError("inactive visual branch was unfrozen")

    def _initial(self, sequence, name, width):
        value = sequence["initial"].get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
            raise ValueError(f"initial {name} must have shape [1,N,{width}]")
        require_finite(value, f"initial {name}")
        return value.to(self.device).clone()

    def _sequence_loss(self, sequence, expected_driver=None):
        _validate_event_sequence(sequence)
        required_driver = self.driver if expected_driver is None else expected_driver
        if sequence["driver"] != required_driver:
            raise ValueError(f"{self.stage} requires {required_driver}-driven trajectories")
        camera_total = sum(event["kind"] == "camera" for event in sequence["events"])
        required = self.burn_in + self.train_frames
        if camera_total != required:
            raise ValueError(f"sequence must contain exactly {required} camera frames")

        visual_hidden = self._initial(sequence, "vision", 256)
        teacher_hidden = self._initial(sequence, "teacher_policy", 128)
        student_hidden = self._initial(sequence, "student_policy", 128)
        n = visual_hidden.shape[1]
        latest_prediction = None
        camera_index = 0
        geometry_terms, confidence_terms, action_terms = [], [], []
        valid_count = 0
        last_tick = {"camera": None, "locomotion": None, "navigation": None}

        for event in sequence["events"]:
            kind = event["kind"]
            if kind == "reset":
                done = event["done"].to(self.device)
                visual_hidden = _reset_hidden(visual_hidden, done)
                teacher_hidden = _reset_hidden(teacher_hidden, done)
                student_hidden = _reset_hidden(student_hidden, done)
                if latest_prediction is not None:
                    latest_prediction = latest_prediction * (~done.bool()).to(latest_prediction.dtype)[:, None]
                continue
            expected_period = {"camera": 10, "locomotion": 4, "navigation": 20}[kind]
            previous_tick = last_tick[kind]
            if previous_tick is not None and event["tick"] - previous_tick != expected_period:
                raise ValueError(f"{kind} events do not follow the required simulator clock")
            last_tick[kind] = event["tick"]

            if kind == "camera":
                image = event["image"].to(self.device)
                state = event["state"].to(self.device)
                target = event[self.branch].to(self.device)
                require_finite(image, "camera image")
                require_finite(state, "camera state")
                require_finite(target, f"teacher {self.branch}")
                if image.shape[0] != n or state.shape != (n, 45):
                    raise ValueError("camera event batch differs from recurrent state")
                latest_prediction, visual_hidden = self.active(image, state, visual_hidden)
                if camera_index == self.burn_in - 1:
                    visual_hidden = visual_hidden.detach()
                    latest_prediction = latest_prediction.detach()
                    teacher_hidden = teacher_hidden.detach()
                    student_hidden = student_hidden.detach()
                learning = camera_index >= self.burn_in
                if learning:
                    count = target.shape[-1] // 2
                    if target.shape != latest_prediction.shape:
                        raise ValueError("teacher geometry does not match active visual branch")
                    validity = target[..., count:]
                    if ((validity < 0) | (validity > 1)).any():
                        raise ValueError("geometry validity must lie in [0,1]")
                    error = F.smooth_l1_loss(latest_prediction[..., :count], target[..., :count], reduction="none")
                    weight = validity.sum()
                    geometry_terms.append((error * validity).sum() / weight.clamp_min(1))
                    confidence_terms.append(F.binary_cross_entropy(latest_prediction[..., count:], validity))
                    valid_count += int((validity > 0).sum().item())
                camera_index += 1
                continue

            if kind != self.policy_kind:
                continue
            teacher_obs = event["teacher_obs"].to(self.device)
            require_finite(teacher_obs, "teacher observation")
            if teacher_obs.shape != (n, self.policy.obs_dim):
                raise ValueError("teacher observation does not match the actual policy state")
            if latest_prediction is None:
                raise ValueError("policy event occurred before a real camera event")
            with torch.no_grad():
                teacher_action, teacher_hidden = self.policy(teacher_obs, teacher_hidden)
            student_obs = torch.cat((teacher_obs[..., :self.policy.prefix_dim], latest_prediction), dim=-1)
            student_action, student_hidden = self.policy(student_obs, student_hidden)
            if camera_index > self.burn_in:
                action_terms.append(F.mse_loss(student_action, teacher_action))

        if len(geometry_terms) != self.train_frames:
            raise RuntimeError("camera learning window was not fully consumed")
        if not action_terms:
            raise ValueError("sequence contains no active-policy event in the learning window")
        geometry = torch.stack(geometry_terms).mean()
        confidence = torch.stack(confidence_terms).mean()
        action = torch.stack(action_terms).mean()
        total = (self.geometry_weight * geometry + self.validity_weight * confidence +
                 self.action_weight * action)
        return total, geometry.detach(), confidence.detach(), action.detach(), valid_count

    def update(self, replay, student_replay=None):
        """Run one update; Stage4 always mixes teacher/student sequences exactly 1:1."""
        self.active.train()
        self.frozen_branch.eval()
        self._assert_frozen()
        if self.stage == "stage4":
            if student_replay is None:
                student_replay = replay
            half = self.batch_size // 2
            teacher_batch = self.teacher_replay.sample(half, driver="teacher")
            student_batch = student_replay.sample(half, driver="student")
            batch = [(sequence, "teacher") for sequence in teacher_batch]
            batch += [(sequence, "student") for sequence in student_batch]
            teacher_count, student_count = half, half
        else:
            batch = [(sequence, "teacher") for sequence in replay.sample(self.batch_size, driver="teacher")]
            teacher_count, student_count = self.batch_size, 0

        losses = [self._sequence_loss(sequence, driver) for sequence, driver in batch]
        total = torch.stack([item[0] for item in losses]).mean()
        if not torch.isfinite(total):
            raise FloatingPointError("distillation produced a nonfinite loss")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = [parameter.grad for parameter in self.active.parameters()]
        if not any(gradient is not None and torch.count_nonzero(gradient).item() for gradient in gradients):
            raise RuntimeError("active visual branch received no gradient")
        if any(parameter.grad is not None for parameter in self.policy.parameters()):
            raise RuntimeError("frozen policy unexpectedly received gradients")
        if any(parameter.grad is not None for parameter in self.frozen_branch.parameters()):
            raise RuntimeError("inactive visual branch unexpectedly received gradients")
        self.optimizer.step()
        for name, parameter in self.active.named_parameters():
            require_finite(parameter, f"visual parameter {name}")
        self._assert_frozen()
        self.iteration += 1
        return {
            "loss": float(total.detach()),
            "geometry_loss": float(torch.stack([item[1] for item in losses]).mean()),
            "confidence_loss": float(torch.stack([item[2] for item in losses]).mean()),
            "action_loss": float(torch.stack([item[3] for item in losses]).mean()),
            "valid_geometry_count": sum(item[4] for item in losses),
            "teacher_sequences": teacher_count,
            "student_sequences": student_count,
        }

    @torch.no_grad()
    def collect(self, env, replay, camera_frames=None):
        """Collect one real event sequence using the stage's required driver.

        ``env.hier`` owns the 200 Hz clock and the lossless 20 Hz camera FIFO.
        Calling this method never manufactures images or geometry when an event
        is absent.  Stage4 installs this vision model into the runtime before
        collecting, so all recorded transitions after camera bootstrap use the
        student geometry.
        """
        runtime = getattr(env, "hier", None)
        required_api = ("consume_camera_event", "camera_events_pending", "current_tick",
                        "refresh", "set_perception", "teacher_locomotion_obs",
                        "teacher_navigation_obs", "locomotion_obs", "navigation_obs")
        if runtime is None or any(not hasattr(runtime, name) for name in required_api):
            raise TypeError("environment does not expose the hierarchical event runtime")
        if getattr(env, "num_envs", None) is None or getattr(env, "device", None) is None:
            raise TypeError("environment does not expose num_envs and device")
        frame_count = self.burn_in + self.train_frames if camera_frames is None else int(camera_frames)
        if frame_count != self.burn_in + self.train_frames:
            raise ValueError("collected sequences must exactly match burn-in plus learning frames")
        n, env_device = int(env.num_envs), torch.device(env.device)
        provider = None
        bootstrap_events = []
        if self.stage == "stage4":
            provider = DeploymentPerception(self.vision, n, env_device)

        sequence = {
            "source": "simulator", "driver": self.driver,
            "initial": {"vision": torch.zeros(1, n, 256, device=env_device),
                        "teacher_policy": torch.zeros(1, n, 128, device=env_device),
                        "student_policy": torch.zeros(1, n, 128, device=env_device)},
            "events": [],
        }
        drive_hidden = torch.zeros(1, n, 128, device=env_device)
        pending_done = None
        cameras = 0
        action_dim = 12 if self.policy_kind == "locomotion" else 3
        clear_perception = getattr(runtime, "clear_perception", None)
        if callable(clear_perception):
            clear_perception()
        env.reset()

        # Start each sequence at a camera boundary. Stage3 reaches it under the
        # teacher; Stage4 uses an unrecorded zero-action sensor bootstrap before
        # installing the student provider. No teacher action enters Stage4.
        while runtime.camera_events_pending == 0:
            if self.stage == "stage4":
                bootstrap_action = torch.zeros(n, action_dim, device=env_device)
            else:
                runtime.refresh()
                teacher_obs = getattr(runtime, f"teacher_{self.policy_kind}_obs")
                bootstrap_action, drive_hidden = self.policy(teacher_obs, drive_hidden)
            result = env.step(bootstrap_action)
            if not isinstance(result, (tuple, list)) or len(result) < 4:
                raise TypeError("environment step must return observations, reward, terminated and truncated")
        if self.stage != "stage4":
            sequence["initial"]["teacher_policy"] = drive_hidden.clone()
        if self.stage == "stage4":
            while runtime.camera_events_pending:
                event = runtime.consume_camera_event()
                if event is None:
                    raise RuntimeError("camera FIFO count disagrees with its consumer")
                provider.update_camera(event["image"], event["state"])
                bootstrap_events.append(event)
            # Obstacle tuning keeps the already-trained terrain branch frozen,
            # but it still drives the frozen 50 Hz locomotion policy.
            runtime.set_perception(provider, drive_terrain=True,
                                   drive_obstacle=self.branch == "obstacle")

        while True:
            reset_recorded = torch.zeros(n, dtype=torch.bool, device=env_device)
            while bootstrap_events or runtime.camera_events_pending:
                event = bootstrap_events.pop(0) if bootstrap_events else runtime.consume_camera_event()
                if event is None:
                    raise RuntimeError("camera FIFO count disagrees with its consumer")
                reset_mask = event.get("reset_mask")
                sequence["events"].append({"kind": "camera", "tick": int(event["tick"]),
                                           **{key: event[key] for key in ("image", "state", "terrain", "obstacle")}})
                if isinstance(reset_mask, torch.Tensor) and reset_mask.any():
                    sequence["events"].append({"kind": "reset", "tick": int(event["tick"]),
                                               "done": reset_mask})
                    reset_recorded |= reset_mask.to(env_device).bool()
                    drive_hidden = _reset_hidden(drive_hidden, reset_mask.to(env_device))
                cameras += 1
                if cameras > frame_count:
                    raise RuntimeError("an environment step crossed the requested camera sequence boundary")
            if pending_done is not None:
                pending_done = pending_done & ~reset_recorded
            if pending_done is not None and pending_done.any():
                sequence["events"].append({"kind": "reset", "tick": int(runtime.current_tick),
                                           "done": pending_done})
                drive_hidden = _reset_hidden(drive_hidden, pending_done)
                if provider is not None:
                    provider.reset(torch.nonzero(pending_done, as_tuple=False).flatten())
                pending_done = None

            runtime.refresh()
            teacher_obs = getattr(runtime, f"teacher_{self.policy_kind}_obs").clone()
            live_obs = getattr(runtime, f"{self.policy_kind}_obs")
            driver_obs = teacher_obs if self.driver == "teacher" else live_obs
            chosen_action, drive_hidden = self.policy(driver_obs, drive_hidden)
            require_finite(chosen_action, f"{self.driver} driver action")
            sequence["events"].append({"kind": self.policy_kind, "tick": int(runtime.current_tick),
                                       "teacher_obs": teacher_obs,
                                       "executed_action": chosen_action.clamp(-3.0, 3.0).clone()})
            if cameras == frame_count:
                break

            result = env.step(chosen_action)
            if not isinstance(result, (tuple, list)) or len(result) < 4:
                raise TypeError("environment step must return observations, reward, terminated and truncated")
            terminated, truncated = result[2], result[3]
            if not isinstance(terminated, torch.Tensor) or not isinstance(truncated, torch.Tensor):
                raise TypeError("environment termination outputs must be tensors")
            pending_done = terminated.bool() | truncated.bool()

        # A policy decision and camera capture may share a physics tick.  The
        # environment exposes the decision before step() and the camera after
        # step(); replay orders equal-tick sensing first so the student sees the
        # real frame that existed when that decision was made.
        priority = {"camera": 0, "locomotion": 1, "navigation": 1, "reset": 2}
        sequence["events"].sort(key=lambda event: (event["tick"], priority[event["kind"]]))
        replay.append(sequence)
        return sequence

    def save(self, path, infos=None):
        metadata = {"iteration": self.iteration, "branch": self.branch, "infos": infos}
        if self.checkpoint_callback is not None:
            return self.checkpoint_callback(self, path, metadata)
        return save_checkpoint(path, self.stage, {"vision": self.vision},
                               optimizer=self.optimizer, metadata=metadata)


class DeploymentPerception:
    """Inference camera provider; absent data is an error, never a fake frame."""

    def __init__(self, vision, num_envs, device="cpu"):
        self.vision = vision.to(device).eval()
        self.device = torch.device(device)
        self.hidden_terrain = torch.zeros(1, num_envs, 256, device=device)
        self.hidden_obstacle = torch.zeros_like(self.hidden_terrain)
        self._terrain = torch.zeros(num_envs, 512, device=device)
        self._obstacle = torch.zeros(num_envs, 144, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.hidden_terrain[:, ids] = 0
        self.hidden_obstacle[:, ids] = 0
        self._terrain[ids] = 0
        self._obstacle[ids] = 0
        self.ready[ids] = False

    @torch.no_grad()
    def update_camera(self, image, state):
        require_finite(image, "camera input")
        require_finite(state, "camera proprioception")
        terrain, obstacle, ht, ho = self.vision(
            image.to(self.device), state.to(self.device), self.hidden_terrain, self.hidden_obstacle)
        require_finite(terrain, "visual terrain")
        require_finite(obstacle, "visual obstacle")
        self._terrain, self._obstacle = terrain, obstacle
        self.hidden_terrain, self.hidden_obstacle = ht.detach(), ho.detach()
        self.ready.fill_(True)

    def terrain(self):
        if not self.ready.all():
            raise RuntimeError("terrain requested before a camera update after reset")
        return self._terrain

    def obstacle(self):
        if not self.ready.all():
            raise RuntimeError("obstacles requested before a camera update after reset")
        return self._obstacle


def run_distillation(args, workspace):
    """CLI entry point for Stage3A/3B teacher and Stage4 student collection."""
    from ..checkpoints import load_actor, load_checkpoint, load_vision
    from ..envs import make_env
    from ..models import VisionModel

    stage = args.stage
    if stage not in {"stage3a", "stage3b", "stage4"}:
        raise ValueError("distill requires --stage stage3a, stage3b or stage4")
    if args.iterations is None or args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    branch = args.branch
    if stage == "stage4" and branch not in {"terrain", "obstacle"}:
        raise ValueError("Stage4 requires --branch terrain or obstacle")
    branch = branch or ("terrain" if stage == "stage3a" else "obstacle")
    policy_kind = "locomotion" if branch == "terrain" else "navigation"
    policy_path = args.locomotion_checkpoint if policy_kind == "locomotion" else args.navigation_checkpoint
    if not policy_path:
        raise ValueError(f"--{policy_kind}-checkpoint is required")
    locomotion_path = args.locomotion_checkpoint
    if policy_kind == "navigation" and not locomotion_path:
        raise ValueError("navigation distillation requires --locomotion-checkpoint")

    workspace = Path(workspace)
    config_path = workspace / "configs" / "distillation.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"distillation config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_envs = args.num_envs or int(config["num_envs"])
    policy = load_actor(policy_path, policy_kind, args.device, require_validated=True)
    locomotion = (load_actor(locomotion_path, "locomotion", args.device, require_validated=True)
                  if policy_kind == "navigation" else None)

    source_checkpoint = args.resume or args.vision_checkpoint
    if stage in {"stage3b", "stage4"} and not source_checkpoint:
        raise ValueError(f"{stage} requires --vision-checkpoint or --resume")
    vision = (load_vision(source_checkpoint, args.device, require_validated=False)
              if source_checkpoint else VisionModel().to(args.device))
    teacher_replay = None
    if stage == "stage4":
        if not args.teacher_replay:
            raise ValueError("Stage4 requires --teacher-replay")
        teacher_replay = SequenceReplay.load(
            args.teacher_replay, capacity=int(config.get("replay_capacity", 256)),
            max_bytes=int(config.get("replay_max_bytes", 512 * 1024 * 1024)))

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if getattr(args, "run_name", ""):
        run_name += f"_{args.run_name}"
    output_dir = (Path(args.output_dir) if args.output_dir else
                  workspace / "checkpoints" / stage / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream = {"locomotion": locomotion_path, "navigation": args.navigation_checkpoint,
                "vision": source_checkpoint, "teacher_replay": args.teacher_replay}

    def checkpoint_callback(owner, path, metadata):
        metadata.update({"purpose": "train", "capability_validated": False, "upstream": upstream})
        return save_checkpoint(path, stage, {"vision": owner.vision}, owner.optimizer, metadata)

    runner = PerceptionDistillationRunner(
        vision, policy, stage, config, branch=branch, device=args.device,
        teacher_replay=teacher_replay, checkpoint_callback=checkpoint_callback)
    if args.resume:
        resumed = load_checkpoint(args.resume, expected_stage=stage)
        if resumed.get("optimizer") is None:
            raise ValueError("resume checkpoint has no optimizer state")
        runner.optimizer.load_state_dict(resumed["optimizer"])
        iteration = resumed.get("metadata", {}).get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("resume checkpoint has no valid iteration")
        runner.iteration = iteration

    replay = SequenceReplay(capacity=int(config.get("replay_capacity", 256)),
                            max_bytes=int(config.get("replay_max_bytes", 512 * 1024 * 1024)),
                            metadata={"stage": stage, "branch": branch, "seed": args.seed}, seed=args.seed)
    env = make_env(stage=policy_kind, num_envs=num_envs, device=args.device, enable_cameras=True,
                   locomotion_policy=locomotion, seed=args.seed, terrain=args.terrain, purpose="train")
    save_interval = int(config.get("save_interval", 50))
    try:
        env.reset()
        for _ in range(args.iterations):
            runner.collect(env, replay)
            metrics = runner.update(replay)
            if runner.iteration % save_interval == 0:
                runner.save(output_dir / f"model_{runner.iteration}.pt", metrics)
        runner.save(output_dir / f"model_{runner.iteration}.pt", metrics)
        if stage in {"stage3a", "stage3b"}:
            replay.save(output_dir / "teacher_replay.pt")
    finally:
        env.close()
    print(f"DISTILLATION FINISHED; checkpoints: {output_dir}")
    return output_dir
