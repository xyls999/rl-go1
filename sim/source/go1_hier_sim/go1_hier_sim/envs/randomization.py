"""Domain randomization the stock Isaac Lab velocity task does not cover.

The sim-to-real recipe in Rudin et al. (App. A) and the RSL-RL stack randomizes
physics properties and actuator behaviour so a policy keeps margin on hardware.
Isaac Lab already ships friction, mass, centre of mass, restitution and push
terms; the motor model (PD gains and the simulated effort ceiling) is
robot-specific and lives here. Every randomization restores the nominal value
first, so repeated resets cannot compound the scale.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


# Per-actuator nominal values captured on the first call. Keyed by id() because
# actuators are never re-created after the scene is loaded.
_NOMINAL: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}


def randomize_motor_model(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stiffness_scale: tuple[float, float] = (0.90, 1.10),
    damping_scale: tuple[float, float] = (0.90, 1.10),
    effort_scale: tuple[float, float] = (0.90, 1.10),
):
    """Scale PD gains and the effort limit of every actuator per reset.

    A real Go1 unit's joint friction and motor strength differ from the nominal
    model by roughly this much, so training with a fixed model over-fits one
    exact actuator and loses the margin the hardware needs. Gains are written
    back as ``nominal * scale`` rather than ``current * scale`` because the term
    runs on every reset of every environment.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not asset.is_initialized:
        return

    for actuator in asset.actuators.values():
        key = id(actuator)
        if key not in _NOMINAL:
            nominal_stiffness = actuator.stiffness.detach().clone()
            nominal_damping = actuator.damping.detach().clone()
            limit = getattr(actuator, "effort_limit_sim", None)
            nominal_limit = None if limit is None else limit.detach().clone()
            _NOMINAL[key] = (nominal_stiffness, nominal_damping, nominal_limit)
        nominal_stiffness, nominal_damping, nominal_limit = _NOMINAL[key]

        def _scale(reference: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
            sampled = torch.empty(
                (len(env_ids),) + reference.shape[1:], device=reference.device
            ).uniform_(float(bounds[0]), float(bounds[1]))
            return sampled

        actuator.stiffness[env_ids] = nominal_stiffness[env_ids] * _scale(
            nominal_stiffness, stiffness_scale
        )
        actuator.damping[env_ids] = nominal_damping[env_ids] * _scale(
            nominal_damping, damping_scale
        )
        if nominal_limit is not None and hasattr(actuator, "effort_limit_sim"):
            actuator.effort_limit_sim[env_ids] = nominal_limit[env_ids] * _scale(
                nominal_limit, effort_scale
            )
