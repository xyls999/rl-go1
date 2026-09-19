"""Training entry points; importing this package does not start Isaac Sim."""

from .distillation import (DeploymentPerception, PerceptionDistillationRunner,
                           SequenceReplay, perception_loss, run_distillation)

__all__ = ["DeploymentPerception", "PerceptionDistillationRunner", "SequenceReplay",
           "perception_loss", "CheckedPPO", "HierOnPolicyRunner"]
__all__.append("run_distillation")


def __getattr__(name):
    if name in {"CheckedPPO", "HierOnPolicyRunner"}:
        from .on_policy import CheckedPPO, HierOnPolicyRunner

        return {"CheckedPPO": CheckedPPO, "HierOnPolicyRunner": HierOnPolicyRunner}[name]
    raise AttributeError(name)
