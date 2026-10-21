# Go1 v7 Reward Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans; follow TDD, and do not start a second simulator.

**Goal:** Remove falsely paid LP-ACRL flight bonuses, then test bounded combined-course support, safe segment completion and tire-stall shaping against the same s22 initialization and deterministic 11-family gate.

**Architecture:** New immutable v7a and v7b reward TOMLs, pure PyTorch reward helpers plus per-environment event/stall state, optional RewardManager terms. Keep v5/v6 and all observation/action/network interfaces unchanged. New distinct driver configs and state/results/run IDs, sequential simulation only; never resume an older optimizer.

**Tech Stack:** Python 3.11, PyTorch, pytest via `./sim.sh test -q`, Isaac Lab on remote `/config/Desktop/sim`.

**Spec:** Approved in-chat v7 design: close false `leap_clear`, add bounded support near first three short gaps, delayed safe crossing and tire exit, targeted time-delayed stall penalty, then isolated A/B gates.

## Global Constraints

- Preserve combined layout, 11 families, speed 0.35–0.8 m/s, 512 environments, original s22 model and v6 files and old LP states; stop flag stays present.
- Reward contract changed: use `--initialize-from` s22 model_550.pt, never `--resume` older optimizer. No automatic p2 or old watchdog.
- Only one Isaac process at a time; complete one 900-iteration stage and its 11-family deterministic gate before launching the next.
- A gate passes ONLY when all 11 families individually reach >=90% in the same report; otherwise `hold`.
- Repository is not Git: isolate using new local `v7_work/`, backup each remote file before deployment, record original/deployed SHA-256 and verify immutable originals afterward.

### Task 1: Flight-bonus isolation

**Files:** Create `v7_work/test_v7_reward_config.py`, `v7_work/rewards_stage1_gaps_v7a_no_false_leap.toml`; `v7_work/` fixtures are copied from `v1_work` without touching v6 originals.

- [ ] First test that v7a parses, retains 50+ optional terms, `leap_clear.weight==0`, and all other v6 weights identical; verify RED, then create v7a TOML, verify GREEN.
- [ ] Add test reproducing raw false payout using a gap-family example with all-feet flight on finish flat but no real void crossing; show current `leap_clear` raw40, and assert v7a paid value0. Verify RED/GREEN without changing legacy v6 behavior.
- [ ] Assert source/plan old checkpoint and v6 checksums preserved.

### Task 2: Bounded v7b shaping, test first

**Files:** Create `v7_work/test_mixed_transition_v7.py`, `v7_work/mixed_transition_reward_v7.py`, `v7_work/rewards_stage1_gaps_v7b_supported_crossing.toml`. Modify copies of `reward_config.py`, `config.py`, `runtime.py`, `geometry.py` under `v7_work/`, then deploy to their live paths after passing tests.

- [ ] RED: tensor tests for low but recoverable height .21, fall-height .179, upright, two support contacts, moving/stationary/airborne, combined/noncombined and out-of-window, [0,1] bound and finite values.
- [ ] GREEN: `supported_short` only in x .85–3.65 combined; smoothly grade `min(root_z,body_height)` from .18 to .32 and `-gravity_z` from .55 to .9, require >=2 contacts and positive forward velocity, cap speed multiplier at 1 instead of paying faster running. Start weight .8.
- [ ] RED: per-env per-segment state tests: first three gaps paid at most once each only after actual far-side foot support, 5 stable control steps, >=.08m forward after crossing, root>=.27, upright>=.85, no fall/collision; tire segment only after actual tire-region visit, stable exit; reset and family isolation; no payment after stepping around a tire.
- [ ] GREEN: expose four per-environment reward-only segment bounds from generated mesh `course_profile.blocks` (first three short gaps and the FIRST tire field regardless of randomized middle-block order), with spawn-relative x and no terrain mesh change. Use these exact bounds for support, stable far-lip/tire exit, and stall windows. Maintain per-segment claimed/streak/crossing x state, bounded `safe_segment` [0,1], weight6; require tire-band lateral footprint and contact to reject bypass rewards.
- [ ] RED/GREEN: tire-only `tire_stall` after >=1s without progress and command>=.35, zero during brief foot adjustment, bounded [0,1], hysteresis reset after real movement; initial negative weight -2, no change on other terrains.
- [ ] Validate optional strict TOML schema, sign/bound/offset/order checks and legacy v5/v6 loader unchanged; register terms with RewardManager only for v7b. New state resets on every environment reset and uses per-env torch tensors.

### Task 3: Deploy and verify without simulator overlap

**Files:** `v7_work/` local implementation, `reports/go1_v7_deploy_manifest.json` and distinct `configs/curriculum_phases_v7a.json`, `configs/curriculum_phases_v7b.json` on remote.

- [ ] Confirm no live sim, stop flag present, original checkpoints/state SHA-256; backup each overwritten remote Python source and record SHA-256 before atomic copy. New TOMLs and driver configs must not exist before copy.
- [ ] Run `./sim.sh test -q`; use sequential one-env smoke for v7a and v7b, assert no reward-shape/finite errors, old v6 tests pass, v7a `leap_clear` paid0 even on finish flat, v7b signals only designated combined windows.
- [ ] Compare rollout reward stats per family, not only aggregate; verify no term accrues by standing/hopping or off-target family.

### Task 4: Sequential isolated experiments and authoritative gate

**Files:** Distinct v7a/v7b driver configs, states, result text, logs and checkpoints; execution log `docs/superpowers/plans/2026-10-14-go1-v7-reward-execution.md`.

- [ ] Ensure each starts from the same SHA-256 s22 model_550.pt, the same 512-instance deterministic sampling plan/seed and stage p1 900 iterations; only rewards differ, optimizer always fresh.
- [ ] Run v7a 900 iterations, 11-family deterministic gate; save checkpoint SHA, report and family counts. If training/gate fails infrastructure or report missing, stop; never substitute training rollouts.
- [ ] Only after v7a sim exits, run v7b identically; gate and retain both statuses, including holds. Never proceed to p2 automatically, even if a gate passes.
- [ ] Check no sim remains active, stop flag persists, v5/v6 originals and old LP states unchanged. Report per-family differences and residual risks; never claim success on aggregate completion alone.
