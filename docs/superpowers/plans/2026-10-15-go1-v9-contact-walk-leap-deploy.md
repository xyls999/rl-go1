# Go1 v9 Gap Contact and Leap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task. This workspace is not Git; instead of commits, preserve versioned local sources, remote SHA manifests and untouched v8 backups.

**Goal:** Fix gap-only false termination, keep short gaps brisk, train an actual long-gap handoff controller, add tiny balance-energy term, diagnose tyre transition, and start an isolated v9 p1 train if safety probes pass.

**Architecture:** Separate a pure contact classifier from Isaac runtime; simulator geometry used only to localize and validate terminal contacts, never actor observations. Preserve v8 controls, model schema and 11-family course. Train v9 walker first; a separate freeze-and-train leap controller follows only after walker smoke. Keep one simulator process and deterministic 512-env gate independent of debug playback.

**Tech Stack:** Python 3.11, PyTorch 2.7, Isaac Lab 2.3.2, Isaac Sim 5.1, two A10 GPUs, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-go1-v9-gap-contact-leap-tire-transition-design.md`.

## Global Constraints

- v8 model/geometry/checkpoints, reports, stop flag and launcher bytes remain unchanged; backup and hash any deployed paths.
- Keep all 11 families and 0.35–0.8m/s commands, no watchdog, no p2, never two Isaac simulations at once.
- Never use simulation-only contact/terrain labels as actor inputs or call net force a precise contact point.
- 11/11 each ≥90% **in the same 512-environment deterministic evaluation** is the only promotion gate.
- Abort first-stage train if sensor cannot distinguish sustained ground-bearing nonfoot contact from stair-lip brushing in controlled PhysX probes; leave `hold`.

### Task 1: Isolated source and test contract

**Files:** create `v9_work/source/envs/gap_contact_v9.py`, `v9_work/tests/test_gap_contact_v9.py`; stage copies of v8 runtime/config/geometry and base reward_config for remote tests.

**Interfaces:** `gap_episode_termination(gap_active, contact_grounded, old_fallen, old_gap_fell)` returns `(fallen, fell_into_gap)` tensors; `grounded_nonfoot_contact(forces, body_heights, surface_heights, foot_mask, force_threshold, height_margin, vertical_fraction)` returns one bool per env; `gap_active_from_bounds(root_x, foot_x, bounds, margin)` handles rows for all actual gaps until hind feet leave the landing; it never masks the tyre field. Fail on empty body list or nonfinite sensor inputs.

- [ ] Add pure tensor tests: foot-only descent never terminal, body grounded terminal after consecutive 0.1s, side-face glancing not grounded, outside-gap fall unchanged, trailing foot keeps gap-active, reset clears streak, multi-env independent, nonfinite inputs rejected.
- [ ] Run `PYTHONPATH=source /root/env_isaaclab/bin/python -m pytest tests/test_gap_contact_v9.py -q` remotely and observe expected RED; write minimum pure helper; rerun GREEN.
- [ ] Integrate new all-gap tile bounds in separate scene metadata, without adding actor dimensions; test generated geometry and thresholds (actual short/long, no gap on tyres). Update `runtime._update_failures` under a new `GO1_V9_GAP_CONTACT=1` opt-in only; keep all old v8 paths unchanged when opt-in off. Record termination reason plus per-foot/nonfoot contacts and x before reset.
- [ ] Run 16-env real Isaac controlled probes: foot-only in trench, actual trunk/shin floor bearing, stair face glancing. Inspect force direction and local height and only then finalize thresholds; if ambiguous or requires privileged geometry beyond training/eval, fail the milestone without starting p1.

### Task 2: Reward contract and short-gap/tyre diagnostics

**Files:** `v9_work/source/envs/runtime.py`, `v9_work/source/envs/config.py`, `v9_work/source/reward_config.py`, `v9_work/configs/rewards_stage1_gaps_v9.toml`; tests `v9_work/tests/test_v9_rewards.py`.

- [ ] RED tests verify `balance_energy` raw score is finite [0,1], zero on airborne/unstable recovery, max weighted penalty 0.002/step and existing `joint_power_l1` unchanged; error on invalid config. Implement pure function and wire through strict rewards schema, manager and checkpoint metadata; GREEN.
- [ ] RED tests verify short/long gate separation: short gap no jump gate, short rewards only with ≥2-foot far support and bounded speed; long clearance still needs recorded flight + far support; implement minimal changes and GREEN.
- [ ] Write read-only per-episode telemetry for start-of-tyres and post-gap landing, body link contact, foot support, slip, roll/pitch rate, actual v/cmd, terminal x and reason. Compare mixed transition and same geometry tyre-only; do not modify tyre geometry, density, or termination rule there.

### Task 3: Deploy, calibrate and start first-stage walker

**Files:** `v9_work/scripts/v9_start_walk_p1.sh`, `v9_work/scripts/v9_deploy_manifest.py`, `v9_work/configs/curriculum_phases_v9_walk_p1.json` plus remote versioned reports and backup.

- [ ] Full remote `./sim.sh test` with captured exit and log; 16/512 real Isaac PhysX check, 2-rank 16-env 3-update PPO smoke and full actor export; ensure no active sim before/after each, preserve stop flag.
- [ ] Copy changed files to remote only after per-file SHA and backup; refuse missing/changed v8 originals, duplicate run ID or altered stop flag. Verify v8 model SHA and `scripts/torchrun_sim.sh` SHA.
- [ ] Start one independent v9-walk p1 (two ranks, 512 environments/rank, warm-start v8 `model_899.pt`, critic warmup 150, 900 iterations; speed 0.35–0.8), not watchdog or p2. Check first PPO updates and record exact run ID, PID, log, config hashes; fail visibly and do not claim trained capability if any probe/preflight fails.

### Task 4: Long-gap jump phase (not concurrently with Task 3 simulator)

**Files:** versioned v9 leap script/config; test gate hand-off hidden-state reset and active-policy mask with frozen walker; isolated jump checkpoint/report.

- [ ] RED test for trigger only on scan-confirmed long gaps, one handoff, airborne/landing hysteresis, no short-gap trigger; GREEN with real 16-env PPO smoke.
- [ ] Once walker checkpoint is stable, freeze walker and train masked leap-only policy with actual take-off/flight/far-side landing rewards; perform separate 512-env gap and 11-family full gate. If no valid leap policy emerges, retain `hold` and no auto-p2.

### Task 5: Retain evidence and correct project status

**Files:** `sim/agent.md`, versioned SHA manifest, `reports/v9_*` and any gate results.

- [ ] Document official-vs-playback distinction and contact classifier limitations; independently verify hashes, stop flag, no overlapping simulator; never promote unless same deterministic 512-batch 11/11 pass.

## Execution checkpoint, 2026-10-15 12:08 UTC

- v9 staging and versioned backup `reports/v9_backup_20261015T094448Z/` deployed. First 16-env contact probe exposed a calf ground-contact false trigger on flat approach x≈1.07m; the activation-window correction is covered by a failing-then-passing test. Second probe found legitimate-looking `FR_calf` ground bearing after the first gap x≈1.77m (but no exact contact point); dedicated sidewall friction negative-control remains unverified.
- Remote `reports/v9_full_tests.log` = 229 passed; 16/512 initial PhysX checks passed; two-rank 16-env PPO smoke completed iteration 2/3; exported actor graph roundtrip true. A first smoke failed because warm-start expected a missing v8 attention adapter; fixed with red-green v8→v9 attention-retention test; original failure log retained.
- Independent walker p1 started UTC 12:04, observed critic prewarm iteration 27/150 at 12:08; driver PID43063; `v9_work/reports/v9_p1_start_status.json`. No leap checkpoint, tyre causal A/B, capability gate, or p2. **Do not call these tasks complete until the remaining contact negatives and 512 evaluation are met.**
