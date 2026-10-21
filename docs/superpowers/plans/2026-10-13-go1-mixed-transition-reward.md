# Go1 Mixed Transition Reward v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, event-driven stable landing reward for the first short-gap/tire section of `combined` without changing the route or network interfaces, and automatically run one isolated warm-start training stage plus its deterministic 11-family gate.

**Architecture:** An optional v6 reward table and pure tensor helper feed a new per-environment reward only for forward-moving combined landings in course_x 1.20–5.60 m. Legacy TOMLs keep their original 50-item signatures. A separate phased config/state warm-starts from the untouched s22 partial checkpoint; the driver fails closed on missing deterministic evaluation and stops after one stage.

**Tech Stack:** Python 3.11, PyTorch, unittest/pytest, Isaac Lab, bash, SHA-256 backups.

**Spec:** `docs/superpowers/specs/2026-10-13-go1-mixed-transition-reward-design.md`

## Global Constraints

- Remote `/config/Desktop/sim`, local `E:/tmp-issac`: neither is Git. Do not claim commits/worktrees; capture original SHA-256 and byte-for-byte backup for every changed remote source file.
- Never run multiple Isaac simulators concurrently. Keep existing `lp_phased_stop.flag` and restored `scripts/torchrun_sim.sh` intact; old watchdog remains stopped.
- Preserve original 11-family geometry, 0.35–0.8 m/s commands, actor/critic observation and action dimensions, and all old checkpoint/state paths. New reward signature requires `--initialize-from`, never `--resume` old checkpoint.
- A stage passes only when every one of 11 families reaches ≥90% in one *deterministic* gate; no fallback to rollout when v1's gate fails.
- The user chose B: after tests and serial smoke pass, launch one 900-iteration/512-environment stage and its deterministic gate automatically, then stop for evidence review (not unbounded stages).

---

### Task 1: Optional v6 reward contract, preserving v5

**Files:** Create remote `configs/rewards_stage1_gaps_v6_mixed_balance.toml` from v5; modify `source/go1_hier_sim/go1_hier_sim/reward_config.py`; test `tests/test_reward_config.py`.

**Interfaces:** `load_stage1_rewards(path)` returns the original 50 keys for v5 and 51 for v6. Optional table fields are `weight`, `start_offset`, `end_offset`, `minimum_height`, `target_height`, `minimum_upright`, `target_upright`.

- [x] **Step 1:** Add a unit test: v5 returns exactly 50 unchanged keys; v6 returns `mixed_transition_balance.weight == 1.5`; a temp v6 with `target_height <= minimum_height`, `target_upright <= minimum_upright`, or an unknown field raises ValueError.
- [x] **Step 2:** Run `PYTHONPATH=source/go1_hier_sim /root/env_isaaclab/bin/python -B -m pytest -q tests/test_reward_config.py` and observe the new test fail because v6 does not exist.
- [x] **Step 3:** Copy v5 to v6 without changing existing values, append:
  ```toml
  [mixed_transition_balance]
  weight = 1.5
  start_offset = 1.20
  end_offset = 5.60
  minimum_height = 0.23
  target_height = 0.31
  minimum_upright = 0.55
  target_upright = 0.90
  ```
  Extend loader to permit only this optional table, validate exact fields and ordered thresholds while leaving old configurations returned unchanged. All other unknown tables remain errors.
- [x] **Step 4:** Rerun focused tests, assert v5's JSON reward signature before/after is identical. Back up original source and hash before remote replacement.

### Task 2: Bounded tensor reward and stage-1 integration

**Files:** Create `source/go1_hier_sim/go1_hier_sim/envs/mixed_transition_reward.py`, test `tests/test_mixed_transition_reward.py`; modify `source/go1_hier_sim/go1_hier_sim/envs/runtime.py` and `envs/config.py` only at reward-table/boundary sites.

**Interfaces:** `mixed_transition_balance_score(root_z, body_height, gravity_z, forward_progress, touchdown_events, combined_mask, course_x, cfg) -> torch.Tensor` returns shape `(num_envs,)`, value in `[0,1]`. The stage-1 reward term name is `mixed_transition_balance`; no actor/critic observation field changes.

- [x] **Step 1:** Write focused CPU tensor tests: matched combined at x2 with root_z=.34, body_height=.34, gravity_z=-1, forward_progress=1, one touchdown =>1; each of non-combined, x outside [1.20,5.60], no touchdown, no forward progress, root_z=.18, body_height=.18, or gravity_z=-.48 =>0; mixed batch `(N,)` finite and bounded. Run and observe import failure before creating implementation.
- [x] **Step 2:** Implement score as `combined_mask & (start<=course_x<=end) & touchdown_events.any(-1)`, times `clip(forward_progress,0,1)`, times `clip((min(root_z,body_height)-minimum_height)/(target_height-minimum_height),0,1)`, times `clip(((-gravity_z)-minimum_upright)/(target_upright-minimum_upright),0,1)`. No reward for standing or flying.
- [x] **Step 3:** Rerun score tests. Register optional term in `RewardsCfg` (weight 0 when missing) and set weight from optional table only for Stage-1. Build `self._combined_family_mask` alongside existing `_gap_family_mask`, false without explicit terrain family plan; in `_build_locomotion_reward_terms` compute score after existing `body_height`, `forward_progress`, `touchdown_events`, and local x are available, or zeros for legacy config; add `(N,)` score to terms. Do not alter existing term values, `_fallen`, jump policy, action or observation code.
- [x] **Step 4:** Run focused reward/config/model tests and `./sim.sh test -q`. Verify legacy 50-key TOMLs and interface manifest unchanged, and the only nonzero added reward occurs for `combined` within first-section moving touchdown.

### Task 3: Isolated driver with deterministic-gate fail-close

**Files:** Modify `scripts/run_phased_curriculum.py`; create `configs/curriculum_phases_mixed_balance_v1.json` from full 11-family config; test `tests/test_full_roster_gate.py` or new `tests/test_phased_gate_source.py`.

**Interfaces:** `require_deterministic_gate_report(path: Path | None, config: dict) -> None` exits nonzero if `config['require_deterministic_gate']` and path is absent or unreadable. For old configs, existing fallback behavior is preserved. New config uses `state=lp_mixed_balance_v1_state.json`, `sampler_state=lp_mixed_balance_v1_acrl_state.json`, `results_file=lp_mixed_balance_v1_results.txt`, `init_checkpoint=checkpoints/stage1/2026-10-12_go1-full-roster11-s22-p1-d1/model_550.pt`, `max_stages=1`, `require_deterministic_gate=true`; all other speed/phase/env/gate fields match full roster.

- [x] **Step 1:** Test missing/empty deterministic report path with v1 flag raises SystemExit and cannot yield training-rollout gate; existing config without flag retains fallback; present report passes. Also test `results_file` selection preserves the legacy filename unless explicitly set. Run RED.
- [x] **Step 2:** Implement the helper and invoke it immediately after `run_gate_eval` returns, before selecting rollout/report or updating curriculum state. Resolve `results = WORKSPACE / config.get('results_file', 'lp_phased_results.txt')`. Rerun GREEN + full tests.
- [x] **Step 3:** Create v1 config with only the six permitted changes above; assert all 11 family quotas, gate min 90%, 512 envs, 900 iterations and 0.35–0.8 speed range remain intact. Verify new state/sampler paths did not exist before launch.

### Task 4: Deploy, serial smoke, launch one stage and collect gate

**Files:** Remote source backup copies and SHA ledger under `reports/mixed_transition_v1_deploy_manifest.json`; log `reports/mixed_transition_v1_smoke.log`, process ID/status under `reports/mixed_transition_v1_driver.pid` and existing run reports.

**Interfaces:** `GO1_REWARD_CONFIG_PATH=/config/Desktop/sim/configs/rewards_stage1_gaps_v6_mixed_balance.toml /root/env_isaaclab/bin/python scripts/run_phased_curriculum.py --base-id 2026-10-13_go1-mixed-transition-v1 --config configs/curriculum_phases_mixed_balance_v1.json --max-stages 1`.

- [x] **Step 1:** Record hashes/byte-for-byte backups, atomically deploy changed code/config/tests; run `./sim.sh test -q`, `bash -n scripts/torchrun_sim.sh`, verify checkpoint hash `d047ecea7accf0443165e7651b65572bd39f467e5c8e94744f917f73e66ffbac` and no active train/eval process.
- [x] **Step 2:** Under v6 config, launch exactly one one-env headless combined evaluation using the old checkpoint + `reports/diagnostic_s22_partial_model550_combined_plan.json` and `GO1_REWARD_SHAPE_AUDIT=1` (shape check); wait for exit, read return code and report/log, then verify no Isaac process. A failure blocks Stage-1 launch.
- [x] **Step 3:** Run a second one-env combined evaluator without shape audit and with `--debug-motion`, confirm `REWARD` logs expose `mixed_transition_balance` with finite scalar value and no shape/contract error; wait for complete cleanup. If the smoke still shows all zeros, inspect touchdown/forward/position inputs before training (do not tune blind).
- [x] **Step 4:** Start the v1 driver detached with v6 environment and new base ID, after checking STOP flag and no sim PIDs; record PID, command, source hashes, plan/state paths. Never remove old `lp_phased_stop.flag` or relaunch old watchdog.
- [x] **Step 5:** Check stage training started with `--initialize-from model_550.pt`, new reward metadata/table, 512 envs and one active torchrun; monitor for model_899.pt and a new deterministic `stage1-v2` gate from the exact v1 run ID. If gate missing or evaluator fails, stop/hold without claiming pass; report exact available status. If gate exists, cite per-family denominator/results and decision; do not extend beyond one stage without evidence.
