# Go1 V1333 A05 Objective-Aligned Training Implementation Plan

> **For agentic workers:** Execute inline in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in strict-completion reward and clearance-confirmed per-gap flight credit, deploy the isolated A05 candidate, and run exactly one 512-environment × 900-iteration training job.

**Architecture:** Preserve A03/A04 reward profiles and the legacy `success` path. A05 adds an optional `task_completed` reward that terminates the episode at the exact project completion contract, plus a separate escrow-style airborne-progress term whose pending metres are paid only when the existing stable per-gap clearance flag is set; an incomplete near-side touchdown cancels pending metres. A one-shot A05 runner pins the A03 parent checkpoint, reward delta, task/PPO inputs, unique outputs and non-authorized follow-ons.

**Tech Stack:** Python 3.11, PyTorch tensor helpers, pytest, TOML/JSON contracts, Isaac Lab V1333 training runner, remote `flock` lock.

**Spec:** Approved in chat; evidence/design basis: `docs/research/2026-11-12-go1-v1333-long-gap-completion-research.md`.

## Global Constraints

- Strict completion is exactly `task_finish_reached and not fallen and not fell_into_gap`; out-of-bounds remains a separately reported metric.
- Preserve A03/A04 profiles and legacy `success`; strict completion is an opt-in A05 term and terminates the episode at the completion event.
- Do not alter actor observations, speed bands, terrain/task distribution, A03 parent weights, or legacy reward weights.
- A05 is one fresh-seed, one-shot weights-only warm-start from A03 `model_299.pt`; 512 environments × 900 iterations; optimizer reset.
- Acquire and retain `/config/Desktop/sim/reports/v7_sim.lock` for every remote inspection, deployment, preflight and training operation.
- Training rollout receipt must remain `trained-unverified`; do not run diagnostic, independent evaluation, holdout, formal gate, retry, or automatic follow-on.
- Work only in `.worktrees/go1-v1333-a05` and the authorized remote V1333 project; never overwrite remote `main` or existing checkpoints/reports.

---

## File Map

- Create `v128_work/v132_randomized/v1333/source/go1_hier_sim/go1_hier_sim/envs/confirmed_airborne_progress_v1333.py`: pure tensor helper for pending/paid per-gap distance and clearance commit.
- Modify `.../envs/long_gap_completion_v1333.py`: pure opt-in termination mask for strict completion.
- Modify `.../envs/runtime.py` and `.../envs/config.py`: wire the two optional terms, completion termination and per-environment escrow state/reset.
- Modify `.../reward_config.py`: accept and validate the two new positive optional terms.
- Create `configs/rewards_stage1_gaps_v1333_longgap_a05.toml`: exact A03 reward delta with confirmed distance and strict completion terms; A03/A04 files stay unchanged.
- Create `scripts/train_long_gap_warmstart_v1333_a05.py` and `configs/long_gap_warmstart_plan_a05.json`: pinned one-shot run contract and receipts.
- Create/modify tests covering helper edge cases, strict termination, A05 reward delta, runner scope and output provenance.
- Create A05 code/design evidence under `reports/`; the remote runner will create unique authorization, seed audit, training log, receipt, stage report and checkpoint.

## Task 1: Tests First — Clearance Escrow and Strict Completion

**Files:**
- Create `tests/test_v1333_confirmed_airborne_progress.py`.
- Modify `tests/test_v1333_long_gap_completion.py`.

- [ ] Define the pure API `confirmed_airborne_forward_distance_reward(...) -> (raw_reward[N], pending_distance[N,K], paid_distance[N,K])`. Write helper tests asserting: eligible air distance only accumulates pending credit and returns zero reward before clearance; a set `cleared_gaps` commits pending metres once as `distance/control_dt`; repeated cleared steps do not pay again; a grounded near-side touchdown cancels pending metres; hard failure/intrusion cancels unconfirmed credit; per-gap pending+paid never exceeds `min(max_credit_m, gap_width)`; cleared gaps cannot earn again.
- [ ] Define `strict_completion_termination_mask(finish_reached, fallen, fell_into_gap, enabled)`. Write a test asserting it terminates only when enabled, finish is latched and neither fall flag is set; collision/out-of-bounds are not inputs; when disabled, the caller's legacy `success`/failure termination behavior is unchanged.
- [ ] Run the new tests before implementation on the target V1333 Python, while holding the simulator lock and setting `PYTHONPATH` to the candidate package:

```bash
export PYTHONPATH=/config/Desktop/sim/v128_work/v132_randomized/v1333/source/go1_hier_sim:/config/Desktop/sim/source/go1_hier_sim
/root/env_isaaclab/bin/python -B -m pytest /config/Desktop/sim/v128_work/v132_randomized/v1333/tests/test_v1333_confirmed_airborne_progress.py /config/Desktop/sim/v128_work/v132_randomized/v1333/tests/test_v1333_long_gap_completion.py -q
```

Expected RED: imports/API for the confirmed-credit helper and opt-in termination contract are missing. If failure is unrelated to those behaviors, fix the test setup and repeat until the expected failure is observed.

## Task 2: Implement Opt-In Runtime and Reward Contracts

**Files:**
- Create `source/go1_hier_sim/go1_hier_sim/envs/confirmed_airborne_progress_v1333.py`.
- Modify `source/go1_hier_sim/go1_hier_sim/envs/long_gap_completion_v1333.py`.
- Modify `source/go1_hier_sim/go1_hier_sim/envs/runtime.py`.
- Modify `source/go1_hier_sim/go1_hier_sim/envs/config.py`.
- Modify `source/go1_hier_sim/go1_hier_sim/reward_config.py`.

- [ ] Implement the pure helper with shape/finite/range validation. Maintain `[N,K]` pending and paid tensors. Accumulate only inside the existing long-gap flight eligibility window, with all four feet airborne, positive forward displacement, no intrusion/failure, a valid long-gap width, and the per-gap cap. Commit pending metres once when the corresponding existing stable clearance flag becomes true; cancel pending credit on failure/intrusion or grounded contact while the root remains short of the far lip. Preserve already-paid clearance credit.
- [ ] Add `strict_completion_termination_mask(finish_reached, fallen, fell_into_gap, enabled)` to the pure completion module and test it using literal tensor fixtures. Its contract is the existing completion mask; it must not include collision or out-of-bounds.
- [ ] Add optional `airborne_forward_progress_confirmed` and `task_completed` positive terms to the strict reward loader. Keep legacy `airborne_forward_progress` semantics/schema intact for A04.
- [ ] Wire A05 reward terms into runtime, using existing `self._v132_paid` as the stable per-gap clearance truth; add correctly shaped zero placeholders and reset new state per environment. Do not pay the old A04 term in the A05 profile.
- [ ] Enable strict task termination only when the optional `task_completed` term is present in the selected Stage-1 reward contract. In that profile, replace only the legacy success endpoint in `done` with the exact strict completion mask; retain fall, gap-fall, and out-of-bounds termination. Otherwise preserve the legacy `success` endpoint. Return the strict mask as the one-step reward term; preserve legacy `_success` and `goal_reached` reward/metrics unchanged.
- [ ] Run the Task 1 tests and existing A04 helper/completion tests. Expected GREEN with A03/A04 behavior preserved.

## Task 3: A05 Profile and One-Shot Runner

**Files:**
- Create `configs/rewards_stage1_gaps_v1333_longgap_a05.toml`.
- Create `configs/long_gap_warmstart_plan_a05.json`.
- Create `scripts/train_long_gap_warmstart_v1333_a05.py`.
- Create `tests/test_v1333_confirmed_training_runner_a05.py`.

- [ ] Create A05 reward profile by copying A03 unchanged and adding only `airborne_forward_progress_confirmed = {weight=20.0, max_credit_m=0.45}` and `task_completed = {weight=200.0}`; preserve legacy `goal_reached`, safety penalties, task geometry, and speed distribution.
- [ ] Create a runner contract for run id `go1-v1333-longgap-objective-aligned-warmstart900-a05`, parent A03 run/checkpoint `go1-v1333-longgap-from-best-entrance-warmstart300-a03/model_299.pt`, weights-only initialization with optimizer reset, 512 envs, 900 iterations, final iteration 899, fresh unused audited seed, existing `gaps_long` task plan and PPO profile. Pin the A03→A05 reward delta and all input/output fingerprints. Every diagnostic/evaluation/holdout/gate/retry/follow-on authorization remains false.
- [ ] Require absent unique outputs before launch; verify the lock is actually flocked (not merely open on FD 9) and the idle simulator; consume exactly one `V1333_LONG_GAP_TRAIN_AUTHORIZED=one-shot` token; write an exclusive authorization record before spawning the training child; never auto-retry. The shell wrapper must accept only its exact A05 runner parent and independently verify the lock is held before invoking `sim.py`.
- [ ] Validate report/run/iteration/task-plan/strict-completion fields; require `task_completed <= task_finish_reached`, `task_completion_rate == task_completed / episodes`, and strict completion rate no greater than no-fall/no-gap rate. Validate checkpoint iteration/provenance/hash, source fingerprints, log positive completion marker, and idle simulator after training. Store training completion as `trained-unverified` and explicitly report that no independent evaluation ran.
- [ ] Add tests for the exact plan constants, A03 parent provenance, exact two-term reward delta, 900-iteration command/environment, one-shot-only authorization, unique output paths, report scope, and refusal of diagnostic/evaluation/holdout/gate/retry flags.
- [ ] Run runner tests and A03/A04 runner tests. Expected GREEN without changing their historical run contracts.

## Task 4: Local Verification and GitHub Feature Branch

- [ ] Run the complete V1333 CPU test suite using `/root/env_isaaclab/bin/python -B -m pytest /config/Desktop/sim/v128_work/v132_randomized/v1333/tests -q` with the candidate source and base source in `PYTHONPATH`, while holding the simulator lock. Record full output; do not claim simulator integration from CPU-only tests.
- [ ] Run Python compilation for all changed/new modules and JSON/TOML parsing for A05 plan/profile.
- [ ] Review `git diff --check`, `git diff --stat`, and the full diff; verify no unrelated files are included.
- [ ] Commit only the A05 plan, candidate code/config/runner/tests and A05 documentation to branch `go1-v1333-a05-20261112`.
- [ ] Push the feature branch to `origin`; do not force-update `main` or overwrite another branch. If that exact feature branch already exists remotely, use `--force-with-lease` only against the observed remote head; never use bare `-f` against `main`.

## Task 5: Locked Remote Deployment, Preflight, and One Training Run

- [ ] Acquire `/config/Desktop/sim/reports/v7_sim.lock` non-blocking and retain the same lock descriptor for all remote project reads/writes, deployment, preflight, and training. If acquisition fails, stop without touching the simulator workspace.
- [ ] Under lock, verify target is exactly `/config/Desktop/sim/v128_work/v132_randomized/v1333`, no training/simulator process is active, A03 parent receipt/checkpoint/hash match the pinned contract, A05 output paths are absent, and current active source fingerprint matches the A03 receipt. Never overwrite A03/A04 inputs or outputs.
- [ ] Deploy only the committed A05 files, preserving a dated backup of every replaced remote source file and recording before/after SHA-256 manifests.
- [ ] Run the A05 runner preflight under lock. It must print a positive `preflight-passed-no-training-started` result and confirm the exact parent checkpoint, 512 envs, 900 iterations, fresh seed not yet selected, and output absence.
- [ ] Start exactly one A05 one-shot training command under the same held lock. Do not start diagnostic, independent evaluation, holdout, formal gate, retry, second run, watchdog, or other simulator process.
- [ ] Wait for training to exit; verify exit status, `TRAINING FINISHED` log marker, unique stage report/receipt/checkpoint, model iteration 899, hashes and provenance, result `trained-unverified`, and no simulator processes remain. If any guard, training, or verification fails, preserve all evidence and stop; do not retry.

## Task 6: Final Evidence

- [ ] Re-read the A05 receipt, stage report, seed audit, deployment manifest, and training log from the locked remote path; verify checkpoint SHA-256 independently.
- [ ] Report training rollout strict completion and safety counts only as training evidence, explicitly separate from independent ability evaluation. State exact GitHub branch/commit, remote run ID, checkpoint/report paths, test output, and any blocker.
- [ ] Stop after this deliverable; leave diagnostic, independent evaluation, holdout, and gate untouched.
