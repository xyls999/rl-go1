from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from contracts_v1330 import EXPECTED, RUN_ID  # noqa: E402
from production_v1330 import (  # noqa: E402
    ALLOWED_MODES,
    mode_authorization,
    validate_mode_authorization,
)


def test_v1330_runner_allows_one_train_and_no_diagnostic_or_gate():
    assert RUN_ID == "go1-v1330-gait-gap-aware-s01-p1-d1"
    assert EXPECTED["num_envs"] == 512
    assert EXPECTED["iterations"] == 650
    assert EXPECTED["initialization"] == "random"
    assert EXPECTED["training_authorized"] is True
    assert EXPECTED["diagnostic_authorized"] is False
    assert EXPECTED["automatic_gate_authorized"] is False
    assert EXPECTED["automatic_retry_authorized"] is False
    assert EXPECTED["automatic_follow_on_authorized"] is False
    assert ALLOWED_MODES == ("preflight-seed", "audit-train-seed", "preflight-train", "train")
    assert mode_authorization("train") == "one-train-only"
    assert mode_authorization("audit-train-seed") == "seed-audit-train-only"
    manifest = json.loads((ROOT / "configs" / "manifest.json").read_text())
    spec = json.loads((ROOT / "configs" / "train_execution.json").read_text())
    assert manifest["run_authorized"] is (manifest["status"] == "authorized-one-shot")
    assert manifest["training_authorized"] is (manifest["status"] == "authorized-one-shot")
    assert manifest["diagnostic_authorized"] is False
    assert manifest["automatic_gate_authorized"] is False
    assert manifest["automatic_retry_authorized"] is False
    assert manifest["automatic_follow_on_authorized"] is False
    assert spec["run_id"] == RUN_ID
    assert spec["num_envs"] == 512 and spec["iterations"] == 650
    assert spec["initialization"] == "random"
    assert spec["diagnostic_authorized"] is False
    assert spec["diagnostic_seed"] is None
    assert spec["train_seed"] is None or type(spec["train_seed"]) is int
    expected_manifest_sha = hashlib.sha256((ROOT / "configs" / "manifest.json").read_bytes()).hexdigest()
    if manifest["status"] == "authorized-one-shot":
        assert spec["manifest_sha256"] == expected_manifest_sha
    else:
        assert spec["manifest_sha256"] == manifest["training_run_manifest_sha256"]
    for relative, expected in manifest["runner_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert set(manifest["runner_sha256"]) == {
        "scripts/contracts_v1330.py", "scripts/seed_audit_v1330.py",
        "scripts/preflight_v1330.py", "scripts/production_v1330.py",
        "scripts/verify_and_run_v1330.sh",
    }


def test_v1330_manifest_pins_only_the_approved_source_overlay_deltas():
    manifest = json.loads((ROOT / "configs" / "manifest.json").read_text())
    base = json.loads((ROOT.parent / "v1327" / "configs" / "manifest.json").read_text())
    hashes = manifest["source_files_sha256"]
    base_hashes = base["source_files_sha256"]
    helper = "go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py"
    changed = {name for name in base_hashes if hashes[name] != base_hashes[name]}
    assert changed == {
        "go1_hier_sim/go1_hier_sim/envs/config.py",
        "go1_hier_sim/go1_hier_sim/envs/runtime.py",
        "go1_hier_sim/go1_hier_sim/reward_config.py",
    }
    assert set(hashes) == set(base_hashes) | {helper}
    assert manifest["baseline"]["source/" + helper] is None
    assert len(manifest["sources"]) == len(hashes)


def test_v1330_modes_fail_closed_for_diagnostic_and_missing_training_authority():
    try:
        mode_authorization("evaluate")
    except ValueError:
        pass
    else:
        raise AssertionError("diagnostic evaluation must not be an available V1330 mode")

    try:
        validate_mode_authorization("train", {"V1330_RUN_AUTHORIZED": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("training must require the exact one-train authorization")

    try:
        validate_mode_authorization("preflight-train", {
            "PREFLIGHT_ONLY": "1",
            "V1330_RUN_AUTHORIZED": "one-train-only",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("read-only preflight must not carry run authorization")
