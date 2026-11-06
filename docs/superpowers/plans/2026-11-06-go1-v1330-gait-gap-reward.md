# V1330 Gait and Gap-Aware Reward Candidate Implementation Plan

> **For agentic workers:** Execute inline in this session; no subagents, training, gate, or remote simulation.

**Goal:** Create an untrained, isolated V1330 reward candidate that discourages excessive non-gap hopping and leg swing, makes stopping/retreating at a gap less attractive, and preserves width-aware safe gap-clearance credit.

**Architecture:** Fork the relevant V1329 source/config into a new V1330 directory; never edit V1329 or V1325–V1328. Add pure tensor helpers for local gap windows and gait penalties, integrate them in the copied runtime, and keep actor dimensions and observation/task plan unchanged.

**Tech Stack:** Python, PyTorch, TOML, JSON, pytest.

**Spec:** The user-approved design in the conversation immediately preceding this plan.

## Global Constraints

- V1325, V1326, V1327, V1328, and completed V1329 files/artifacts remain unchanged.
- Actor interface remains 1073 observations / 12 actions; no observation or task geometry edits.
- V1330 is an untrained candidate. Training, holdout, gate, checkpoint evaluation, and remote simulation are out of scope.
- Use existing short-gap supported-step and long-gap flight/stable-landing scorer; do not equate horizontal gap width with linearly increasing vertical foot height.
- Keep true fall/gap termination negative; reduce only the pre-terminal gap-intrusion shaping to a scale comparable with the one-time crossing credit.

---

### Task 1: Test gap-window and gait helper behavior

**Files:**
- Create: `v128_work/v132_randomized/v1330/tests/test_v1330_gait_gap_rewards.py`
- Create later: `v128_work/v132_randomized/v1330/source/go1_hier_sim/go1_hier_sim/envs/gait_gap_rewards_v1330.py`

**Interfaces:**
- `gap_windows(bounds, local_x, approach_distance, landing_distance, short_width_max)` returns `(near_any_gap, near_short_gap)` bool vectors for `[N,K,3]` bounds `(near_lip, far_lip, end)`.
- `fore_aft_swing_cost(foot_forward, contacts, near_gap, max_excursion)` returns one non-negative cost per environment; only swing feet beyond the soft envelope and outside a gap window are charged.
- `non_gap_airborne_cost(current_air_time, near_gap, grace_seconds, ramp_seconds)` returns one cost per environment; it is zero in a gap window and until all four feet have been airborne past the grace interval.
- `short_gap_clearance_target(base_target, near_short_gap, forward_progress, target_height)` applies a fixed lift target only while progressing near a short gap.
- `short_gap_lift_reward(touchdown_clearance, touchdown_events, target, near_short_gap, forward_progress, std)` returns a bounded leading-foot touchdown reward, disabled for long gaps and non-progress.

- [x] **Step 1: Write tests first** for short/long gap windows and no-gap rows, swing-envelope threshold/contact/gap masking, and airtime grace/gap masking.
- [x] **Step 2: Run those tests and confirm they fail** because the V1330 helper module does not exist.
- [x] **Step 3: Implement the five pure tensor helpers** with shape/range validation and bounded outputs.
- [x] **Step 4: Re-run the focused tests and confirm they pass.**

### Task 2: Integrate gait and short-gap lift shaping into isolated runtime

**Files:**
- Create: V1330 copy of V1329 `source/go1_hier_sim/go1_hier_sim/envs/runtime.py` and `reward_config.py`
- Modify: V1330 copies only
- Test: `v128_work/v132_randomized/v1330/tests/test_v1330_gait_gap_rewards.py`

**Interfaces:**
- Runtime computes body-forward foot offsets, near-gap masks from simulator-only gap bounds, and adds `fore_aft_swing_l2`, `non_gap_airborne`, and `short_gap_lift` terms.
- `short_gap_lift` applies a separate progress-gated leading-foot touchdown target/reward only in a near-short-gap window; long gaps continue to use the existing all-feet-flight and stable far-side touchdown scorer.

- [x] **Step 1: Add failing integration assertions** for term/config presence plus helper output shapes and masks for gap/non-gap cases.
- [x] **Step 2: Run the V1330 tests and confirm they fail** on missing terms/config fields.
- [x] **Step 3: Add the new reward fields to the V1330 strict config schema and compute the terms in V1330 runtime.**
- [x] **Step 4: Re-run the focused tests and confirm they pass.**

### Task 3: Create the V1330-only candidate package and conservative reward profile

**Files:**
- Create: `v128_work/v132_randomized/v1330/configs/` (copy required V1329 configs, then version V1330 identifiers/paths)
- Create: `v128_work/v132_randomized/v1330/README.md`
- Create: `v128_work/v132_randomized/v1330/configs/rewards_stage1_gaps_v1330_gait_gap.toml`

**Initial tuning choices:**
- `fallen.weight = -16` (one terminal event; moderate increase from -12).
- Reduce the pre-terminal gap intrusion term from `weight=-80, entry_multiplier=40` to `weight=-8, entry_multiplier=1.5`; preserve its depth signal and separate it from terminal gap failure.
- Add `stall_grow` and `backward_motion` with conservative negative weights and enable the existing V121 runtime hooks only in V1330.
- Add a soft fore-aft swing envelope, a mild non-gap full-flight penalty with grace time, and a short-gap touchdown-lift target; gap-window masks disable the walking penalties.
- Keep V1329 actor observation profile, task plan, and speed-feedback terms unchanged.

- [x] **Step 1: Add failing config tests** for the V1330 values, unchanged actor/task inputs, and all execution authorization flags remaining false.
- [x] **Step 2: Run the config tests and confirm they fail** before the V1330 package exists.
- [x] **Step 3: Copy required V1329 inputs into V1330, version the reward profile and create a README marking it untrained and unauthorized.**
- [x] **Step 4: Run the config tests and confirm they pass.**

### Task 4: Validate the isolated candidate

- [x] Run focused V1330 tests (8 passed) and existing V1329 tests (19 passed); no V1329 paths were edited.
- [x] Run V1330 Python compilation and whitespace validation.
- [x] Verify V1329 candidate source hashes (13/13) and reward/observation/task pins (3/3) match its manifests; verify V1330 manifest authorizations are false. V1325–V1328 were not edited.
- [x] Report local validation only; no training, gate, evaluation, or simulation launched.
