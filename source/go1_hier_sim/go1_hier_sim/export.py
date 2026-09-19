"""Export complete perception/control graphs, with explicit recurrent state."""

from pathlib import Path

import torch

from .checkpoints import file_hash, load_actor, load_checkpoint, load_vision, write_json
from .contracts import LOCO_OBS_DIM, NAV_OBS_DIM, interface_manifest


def export_checkpoint(checkpoint, output_dir):
    payload = load_checkpoint(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = payload["stage"]
    if stage in {"stage1", "stage2"}:
        kind = "locomotion" if stage == "stage1" else "navigation"
        model = load_actor(checkpoint, kind)
        nobs = LOCO_OBS_DIM if kind == "locomotion" else NAV_OBS_DIM
        inputs = (torch.randn(3, nobs), torch.randn(1, 3, 128))
    else:
        kind = "vision"
        model = load_vision(checkpoint).eval()
        inputs = (torch.rand(3, 2, 60, 80), torch.randn(3, 45),
                  torch.randn(1, 3, 256), torch.randn(1, 3, 256))
    scripted = torch.jit.script(model.cpu().eval())
    path = output_dir / f"{kind}.pt"
    scripted.save(str(path))
    loaded = torch.jit.load(str(path)).eval()
    with torch.no_grad():
        expected, actual = model(*inputs), loaded(*inputs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    write_json(output_dir / f"{kind}.json", {
        "interfaces": interface_manifest(), "source_sha256": file_hash(checkpoint),
        "export_sha256": file_hash(path), "stage": stage,
        "metadata": payload["metadata"], "explicit_hidden_state": True,
        "roundtrip_batch_size": 3, "roundtrip_passed": True,
        "input_domain": "normalized_policy_observation_from_contracts_v2",
    })
    return path
