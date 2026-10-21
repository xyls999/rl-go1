# Go1 v10 Observable Leap Code Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task. Steps use checkbox (`- [ ]`) syntax for tracking. This workspace is not Git: preserve v9 sources by copying into `v10_work`, create remote versioned backups and SHA manifests rather than commits.

**Goal:** Deploy isolated trainable Go1 long-gap leap specialist code with a truth-free sensor gate, without claiming trained jump capability or running an autonomous long train.

**Architecture:** A fine-resolution physical ray scanner supplies a fail-closed, observable long-gap signature to the simulated handoff controller; it is not appended to the 1073-dimensional actor. A new randomly initialized recurrent jump actor trains only on steps where its actions are actually executed, with a frozen v8 walker on approach/exit. Geometric/contact truth stays in curriculum, critic, rewards and diagnostics; no checkpoint is promoted before the original 512-env 11-family gate.

**Tech Stack:** Python3.11, PyTorch2.7, Isaac Lab2.3.2, Isaac Sim5.1, pytest; remote `/config/Desktop/sim`, local `E:/tmp-issac`.

**Spec:** `docs/superpowers/specs/2026-10-16-go1-v10-p10-observable-leap-design.md`.

## Global Constraints

- Preserve 11 families, 0.35–0.8m/s commands, 1073 actor observations, 12 joint targets, GRU128; never insert geometry/contact truth into actor or gate.
- Baseline walker is frozen v8 `model_899.pt` SHA `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`; new jump actor random-init, not rebranded gait. Its use does not certify v8 under v9 termination.
- Keep all existing checkpoints, reports, videos and `lp_phased_stop.flag`; no watchdog, no automatic p2 or full training, one Isaac process at a time, no Git commit claim.
- At 0.1m coarse scan 0.24m vs0.29m can alias. If the fine observable scanner fails controlled short/long discrimination or timely trigger, stop before enabling jump/training and report the blocker; never fall back to `long_gap_segments` oracle decision.
- PhysX 16/512 checks and PPO 2–3 update smoke are integration evidence, **not** jump ability; only same-session 512-env deterministic 11/11 each≥90% can promote.

---

### Task 1: Isolate v10 source and prove fine-scan discriminator

**Files:** create `v10_work/source/go1_hier_sim/go1_hier_sim/envs/observable_gap_v10.py`, `v10_work/tests/test_observable_gap_v10.py`; stage copies of v9 `config.py`, `runtime.py`, `cli.py`, and relevant test files. Existing production code remains byte-identical until deployment.

**Interfaces:** `observed_gap_width(column_heights: Tensor[N,C], column_valid: BoolTensor[N,C], forward_offsets: Tensor[N,C], *, drop_m: float=0.30) -> tuple[Tensor[N], Tensor[N], BoolTensor[N]]` returns `(near_lip_m, width_m, reliable)`; no terrain labels/bounds as input. `long_gap_trigger(..., min_long_m: float=0.34, max_short_m: float=0.29, near_min_m: float=0.10, near_max_m: float=0.55) -> tuple[BoolTensor[N], Tensor[N]]` returns `(trigger, lip)`, false/inf on missing or ambiguous scans. Each returned width must be derived from near/far lip transitions with 0.025m sample spacing and multiple lateral rays; planar slopes/steps without a true 0.30m drop must stay false.

- [ ] Write RED tests for ray-visible 0.29m short vs0.34m long (nominal 0.24m/0.29m plus 0.05m low lip) for every 0.005m phase from 0 to0.025m, valid far lip, ahead at0.20–0.50m; assert all short false, all long true, missing/dropout far lip false and no NaN propagation. Run `PYTHONPATH=source/go1_hier_sim /root/env_isaaclab/bin/python -m pytest tests/test_observable_gap_v10.py -q` on a **staged** remote tree or `PYTHONPATH=v10_work/source` locally with available torch; see expected import/behavior failure.
- [ ] Implement only sample-profile classification in the pure module; assert [N,C] shape, strictly increasing finite forward columns for valid rays, valid local ground datum, near-edge deep transition and far-edge recovery. Reject widths whose quantization uncertainty overlaps the ray-visible [0.29,0.34] decision band; use consistent fail-closed `False/inf`, not a forced guess. Run the same tests GREEN and compare to the existing 0.1m scan to preserve the alias counterexample.
- [ ] Do not claim detection in Isaac until Task 2's real scanner test. Save original v9 file SHA in `v10_work/reports/staging_sha.json`.

### Task 2: Sensor-only Isaac gate and finite state handoff

**Files:** `v10_work/source/go1_hier_sim/go1_hier_sim/envs/config.py`, `v10_work/source/envs/runtime.py`, `v10_work/tests/test_observable_gap_v10.py`, `v10_work/tests/test_v10_handoff.py`.

**Interfaces:** opt-in `GO1_V10_OBSERVABLE_LEAP=1` creates a separate `gap_edge_scanner` (longitudinal 0.025m over 1.20m, lateral 0.10m, offset x=0.70m, 50Hz ground mesh). `HierarchicalRuntime.gap_leap_trigger()` uses the new physical ray hits, not `long_gap_segments`, `all_gap_segments` or task-family names. Its opt-in path is inert without a real `jump_policy` or frozen `base_policy`. Existing non-opt-in code path remains unchanged and may be removed only in a separate migration. Handoff state transitions are `walk→prepare→leap→land-confirm→walk` with per-env reset, hard max duration 60 control steps and no successful handback before 10 steps; actual action source sets `policy_mask` correctly.

- [ ] RED tests: short never triggers, long from sensor triggers with reliable near/far lips, no far lip/dropout/flat stair/sloped floor never triggers; changing every privileged gap-bounds tensor while keeping ray hits constant must not change the opt-in gate. Verify no new actor dimension and no checkpoint means no jump handoff.
- [ ] Implement opt-in scene scanner and runtime path, only referencing ground ray hit sensor and robot-relative ray coordinates. Recurrent hidden states advance/reset for frozen walker and specialist per environment, no gradient on frozen weights; persistent contact does not prematurely hand back. Test RED→GREEN and run full unit suite before Isaac.
- [ ] Run one serial 16-env real Isaac check with short and long widths at multiple difficulty/seed values, output raw `near_lip,width,reliable,trigger`, then 512-env initialization check for scene/raycast memory; if short/long overlap or many long triggers missed, STOP and report, do not secretly enable oracle fallback.

### Task 3: Fresh specialist checkpoint contract and bounded PPO smoke

**Files:** `v10_work/source/go1_hier_sim/go1_hier_sim/cli.py`, `v10_work/source/envs/runtime.py` (only as required by tests), `v10_work/tests/test_v10_leap_training.py`, `v10_work/scripts/v10_leap_smoke.sh`, isolated v10 TOML copied from v9 if reward terms unchanged.

**Interfaces:** explicit `--leap-base-checkpoint` plus `GO1_V10_OBSERVABLE_LEAP=1`; reject `--initialize-from` and `--resume` from a walk checkpoint for a new specialist. Save metadata keys `skill=observable_long_gap_leap_v10`, `initialized_from=None`, `frozen_walker_sha256`, `gate=observable_gap_v10`, `capability_validated=false` and unchanged policy obs/action dimensions. Prior `long_gap_clearance_step` is paid only after true flight, crossing the far lip and stable far-side support; training-only geometry truth never appears in observations/gate.

- [ ] RED tests for fresh actor origin, immutable base weights/norms, active-policy gradient mask, per-env GRU reset, strict metadata and refusing gait checkpoint as a jump checkpoint. Run focused pytest to observe the intended failures.
- [ ] Implement minimal CLI metadata/preflight and the jump-only masked PPO path; pass focused tests and full suite. Check `scripts/torchrun_sim.sh` still chooses distributed backend only by explicit v10 environment variable; do not change shared RSL-RL or container mounts.
- [ ] With stop flag, one sim lock and source checkpoint hash verified, run **only** serial 16-env 2–3 PPO updates plus complete actor TorchScript export/roundtrip (the repository's `export.py` uses `torch.jit.script`, not ONNX); any init/NaN/optimizer error is visible and halts. Never call the smoke checkpoint trained/validated jump ability.

### Task 4: Versioned deployment and audit

**Files:** versioned remote `reports/v10_backup_<UTC>/`, `reports/v10_observable_leap_manifest.json`, local `v10_work/reports/`, update `sim/agent.md`.

- [ ] Hash all v10 staged files, original v9 deployed files, v8/v9 checkpoints and `lp_phased_stop.flag`. Refuse unexpected drift; copy changed remote files into a new backup before replacing only matching project files.
- [ ] Deploy under lock after passing full tests. Re-run full remote `./sim.sh test`, targeted 16/512 PhysX checks and opt-in scanner smoke; save outputs and verify active process count zero after each. Export complete actor as TorchScript and compare roundtrip output/hidden state; hash the deployment manifest itself.
- [ ] Update `sim/agent.md` with actual observed status and unresolved scanner/contact risks, no fake jump checkpoint, no claim of Git commit or certification. Keep no simulation running and no watchdog/p2/full training; report artifacts and blockers. If Task 2 discrimination or Task 3 PPO fails, retain old deployed source and provide exact failure, do not claim Task 4 complete.

## Execution record (UTC 2026-10-15, local 2026-10-16)

- [x] Task 1 implementation/RED→GREEN: detector tests including old 0.1m-grid refusal, all sampling phases, invalid rays, 0.05m low-lip correction and lateral centre-stripe negative; original remote v9 Python sources have 34 SHA entries in `v10_work/reports/staging_sha.json`.
- [x] Task 2 core code/physical lead-in: v10 scene scanner, old 16×32 side-foot-lane confirmation, GRU per-env reset and observable re-arm; forced first-lip 16-env runs with short 0/8 and long 8/8 at two difficulty plans/seed variants; 512-env **initialization and 3 steps**, not 11-family capability validation. Output `reports/v10_scanner_physx_16_boundary_reviewed.log`, `reports/v10_scanner_physx_512_final.log`. No direct bounds-mutation integration test; source/side-by-side review shows v10 branch returns before legacy privileged-bound filter, but full PhysX false-positive sweep for every family/lateral offset remains pending.
- [x] Task 3 bounded chain: v8 walker source hash verified, fresh specialist training preflight and checkpoint SHA provenance, true per-step/terminal PPO mask snapshot, zero masked-step entropy and fixed-KL schedule. `reports/v10_mask_reset_probe.log` demonstrates terminal executed mask0 vs next reset mask1. Latest 16×3 PPO smoke `reports/v10_leap_16x3_smoke_final.log` retained mask fractions 0.0000/0.4883/0.2383; latest `model_2.pt` SHA `48f620fa99ed692419b48ab7f470e5a31ed685e841ef2c2a419e58e82b42fc09`, `capability_validated=false`. TorchScript `reports/v10_leap_export_smoke_final/locomotion.json` roundtrips action and hidden state. Scope deviation: used direct one-shot CLI commands rather than `v10_leap_smoke.sh`, and reused the v8 reward TOML for the bounded smoke; no claim that reward weights are appropriate for long training.
- [x] Task 4 versioned backup, protected SHA verification, 257 remote unit tests `reports/v10_full_tests_final.log`, manifest `reports/v10_observable_leap_manifest.json`, current process/flag verification and `sim/agent.md` updated. Work is **code-only opt-in deployment, not ability promotion**. Manifest itself is hashed separately with `sha256sum`.
- [ ] Remaining safety/ability gates: controlled PhysX sidewall negative + true load-bearing nonfoot positive contact contrast; lateral/yaw and all-family false-positive scan calibration; actual flight/landing and handoff quantitative checks across command speeds; long specialist training only after these gates; one deterministic 512-env session with all 11 families each≥90%. No automatic p2 or certification route is enabled for v10.

