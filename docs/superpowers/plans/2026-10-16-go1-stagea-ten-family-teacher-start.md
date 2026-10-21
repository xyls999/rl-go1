# Go1 Stage-A Ten-Family Teacher Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This workspace is NOT a Git repository: replace commit steps with versioned backups and SHA-256 inventories; never claim commits.

**Goal:** Start one isolated, contract-compliant first teacher experiment on exactly the 10 non-`combined` terrain families; preserve the 11-family acceptance gate and all old artifacts.

**Architecture:** Reuse the validated v12 observable speed/zero-standing code and v11 11-family sampler, but select a local ten-family roster in the stage driver without modifying the global terrain registry. Begin with a shared walk/ten-family warm-start experiment from frozen v8 (untrusted under the new contract); diagnose each family before selecting any separate gap/tyre teachers. A ten-family *teacher-system* screen and a subsequent ten-family *single-student* screen are distinct from this first run; neither is a final promotion.

**Tech Stack:** Isaac Lab, Isaac Sim/PhysX, PyTorch, RSL-RL, Python 3.11, two GPU ranks, 512 environments/rank, pytest. Remote `/config/Desktop/sim` via SSH 8022; local isolated `E:/tmp-issac/v13_work/`.

**Spec:** `docs/superpowers/specs/2026-10-16-go1-ten-family-first-multiteacher-dagger-design.md`

## Global Constraints

- Training families exactly `TERRAIN_FAMILIES` minus `combined`; keep global eleven and original 11-family deterministic gate unchanged.
- Stage-1 sampled command and observable scan-adjusted target 0.35–0.8 m/s, no family-truth speed multiplier and no 2% standstill draws. Reset-time execution ramp may be below 0.35; max executed forward command ≤0.8.
- Any teacher and student actor: 1073 inputs, 12 outputs, GRU128, no privileged family/gap/contact truth. Training reward/critic/offline label selector alone may be privileged.
- Stage-A ten-family gate: deterministic same-run 512 environments, ≥40 completed episodes/family, every family ≥90%; teacher coverage is NOT one student's pass. Only after a separate single student's ten-family pass should the frozen student be evaluated on full 11 families.
- Only one Isaac job at a time. Keep `lp_phased_stop.flag` SHA `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`; no watchdog or automatic p2. Preserve v8/v9/v10/v11 weights, reports and videos; pre-deploy SHA backup of each overwritten production source file.
- Contact point ambiguity persists: a first teacher run is exploratory learning, not proof of actual jump, landing, safe non-foot contact or hardware capability.

---

### Task 1: Isolated Ten-Family Roster and Driver (pure Python)

**Files:**
- Create: `v13_work/source/go1_hier_sim/go1_hier_sim/envs/ten_roster_v13.py`
- Copy then modify: `v13_work/scripts/run_phased_curriculum.py` from `v11_status/scripts/run_phased_curriculum.py`
- Create: `v13_work/tests/test_v13_ten_roster.py`

**Interfaces:**
- Consumes: existing `TERRAIN_FAMILIES`, `LPSampler`, `rebalance_family_plan`, `make_balanced_gate_plan`.
- Produces: `select_training_roster(config: dict, full_roster: tuple[str, ...]) -> tuple[str, ...]`; driver uses its result in sampler, rebalancer, gate plan and gate-decision reporting. `full_roster` remains unmodified.

- [ ] **Step 1: Write failing pure tests.** Check `select_training_roster({'training_roster':'ten_except_combined'}, TERRAIN_FAMILIES)` returns 10 distinct names with no `combined`, rejects unknown mode and a full roster without exactly one `combined`. Use a synthetic 512-instance plan to assert stage and gate assign exactly 10 families, each gate family has 51 or 52 slots, all speed/difficulty levels unchanged, and `TERRAIN_FAMILIES` still contains `combined` afterward.
- [ ] **Step 2: Run RED** using remote project's Python when necessary (`PYTHONPATH=v13_work/source/go1_hier_sim:sim/scripts /root/env_isaaclab/bin/python -m pytest v13_work/tests/test_v13_ten_roster.py -q` from Linux workspace, or local Python if dependencies exist). Expected: missing `ten_roster_v13` or failing family-presence assertion, not an import/asset problem.
- [ ] **Step 3: Implement the small selector and driver substitution.** In the selector require mode exactly `ten_except_combined`, `len(full_roster)==11`, exactly one `combined`, and `len(set(full_roster))==11`; then return `tuple(f for f in full_roster if f!='combined')`. In the driver assign `families = select_training_roster(config, TERRAIN_FAMILIES)` before initializing the sampler; pass `families` instead of `TERRAIN_FAMILIES` to the sampler, stage rebalance and balanced gate. Log `balanced_10_families` using actual length; verify all 512 plan assignments are members of `families` before writing either plan, fail closed if not.
- [ ] **Step 4: Run GREEN** for the roster tests; then run existing `v12_work/tests/test_full_roster_gate.py` against the existing global 11-family acceptance path. Do not edit global `TERRAIN_FAMILIES` or the old 11-family gate.

### Task 2: Deterministic Speed Contract for the Driver

**Files:**
- Modify: `v13_work/scripts/run_phased_curriculum.py`
- Create: `v13_work/tests/test_v13_driver_speed.py`
- Copy unchanged from already diagnosed v12: `v13_work/source/go1_hier_sim/go1_hier_sim/envs/{config.py,runtime.py,speed_band_v12.py,episode_trace_v12.py}`, `v13_work/source/go1_hier_sim/go1_hier_sim/cli.py`.

**Interfaces:**
- Consumes: `config['speed_min']`, `config['speed_max']`, plan speed fields and `resolve_speed_band()`.
- Produces: `make_gate_command(workspace: Path, run_id: str, envs: int, episode_seconds: float, checkpoint: Path, config: dict) -> list[str]`; `run_gate_eval` executes this command. Training already passes `--speed-min/--speed-max`.

- [ ] **Step 1: Write failing test** that calls `make_gate_command` and asserts exactly one each of `--speed-min 0.35`, `--speed-max 0.8`, deterministic seed 2026, `--episodes 512`, `--num-envs 512`, and no stochastic-action or certification flag; no plan-to-runtime fallback to 0.45.
- [ ] **Step 2: Run RED** and check the failure is missing gate command interface or missing explicit band.
- [ ] **Step 3: Extract existing gate argument list into `make_gate_command`** and insert `--speed-min`/`--speed-max` values from config; have `run_gate_eval` call this builder. Use existing v12 `resolve_speed_band` to reject inconsistent plan/explicit bounds. Keep `GO1_TASK_PLAN` assignment and deterministic report parsing.
- [ ] **Step 4: Run GREEN** for new test and existing `v12_work/tests/test_v12_speed_band.py`, `v12_work/tests/test_v12_episode_trace.py`. Confirm Stage-1 config has `rel_standing_envs=0.0` and runtime command uses only sampled speed/observed scan, with code search plus the existing tests.

### Task 3: Freeze Stage-A Configuration and Preflight

**Files:**
- Create: `v13_work/configs/curriculum_phases_ten_teacher_v13.json` from v11 p1 configuration with new unique state, sampler_state, results and `training_roster: ten_except_combined`, no `family_priority.combined`, one p1 stage max; same 0.35–0.8, five speed levels, 512, phase p1 disturbance and existing rewards/PPO tables.
- Create: `v13_work/scripts/start_v13_ten_teacher.sh` for one stage under the existing `reports/v7_sim.lock` using `NPROC_PER_NODE=2`, `GO1_POLICY_VARIANT=map_attn_v3`, `GO1_V8_TERRAIN_CONTROL=1`, v8 `model_899.pt` warm-start and original `lp_phased_stop.flag` untouched.
- Create: `v13_work/tests/test_v13_launch_contract.py`.

**Interfaces:**
- Consumes: tasks 1–2 output, remote v8 checkpoint SHA `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`.
- Produces: unique run ID, one p1 experiment, deterministic ten-family gate report, no subsequent stage/p2.

- [ ] **Step 1: Write failing launch-contract test** asserting phase count1, max_stages1, 512 environments, five speed levels, no `combined` priority, distinct state/results paths, gate min_family_episodes40, training roster ten, and shell launcher has no watchdog, auto-p2 or stop-flag deletion.
- [ ] **Step 2: Run RED** for missing configuration/launcher.
- [ ] **Step 3: Write the minimal config and launcher.** Python preflight in launcher verifies SHA of original checkpoint, stop flag, desired source manifest; unique run paths absent; training-plan generator/one-rank dry-run emits only ten names. Check no `scripts/sim.py` or other phased driver exists and acquire lock without waiting. Do not set either v9 contact or v10 leap mode as a capability shortcut. Existing `GO1_REWARD_CONFIG_PATH`/`GO1_PPO_CONFIG_PATH` point to v11 rewards and currently frozen PPO tables.
- [ ] **Step 4: Run GREEN** and generate plans locally for both training and balanced ten-family gate. Save JSON and SHA files before any GPU test. Ensure the original 11-family gate plan remains present and unchanged.

### Task 4: Deploy Reversibly and Run Small Physics Smoke

**Files:**
- Remote backup under `/config/Desktop/sim/reports/v13_ten_teacher_backup_<UTC>/`, containing all overwritten `cli.py`, `envs/config.py`, `envs/runtime.py`, `scripts/run_phased_curriculum.py`, plus SHA manifest. Copy new speed/trace/roster helpers without replacing old checkpoints.
- Remote new config, launcher and tests in their parallel workspace paths; keep source under `/config/Desktop/sim/source/go1_hier_sim/go1_hier_sim/` only after manifest comparison.

- [ ] **Step 1: Query live remote simulator processes/GPU/shm**, stop flag SHA and production source SHA. Fail closed if any simulator/driver exists or baseline SHA differs from last verified values. Back up each production file before copy and write original/new SHA manifest.
- [ ] **Step 2: Deploy isolated v13 files** then run full `./sim.sh test`; capture complete log and nonzero exit. New focused tests, old roster11 tests and export shape contracts must pass before GPU use. Run a separate 2–16 environment, at-most-2-iteration PhysX/PPO chain smoke using a ten-family plan; check observed target command ≤0.8, no sampled standstill, no `combined`, checkpoints remain `capability_validated=false`. Restore production backup if deployment/checks fail; retain evidence.
- [ ] **Step 3: Store stage plan/config/reward/PPO/source/test/smoke SHA** and logs. Distinguish code test and smoke from 90% teacher gate.

### Task 5: Launch and Boundedly Supervise First Teacher Experiment

**Files:**
- Remote `reports/v13_ten_teacher_<run-id>_driver.log`, `reports/lp_plan_<run-id>.json`, gate plan/report, checkpoints, `reports/v13_ten_teacher_<run-id>.exit`.
- Local `v13_work/reports/` status and SHA snapshots; append tested facts to `sim/agent.md`.

- [ ] **Step 1: Recheck process absence and SHA manifest immediately before start**, then launch exactly one two-rank p1 job using `flock -n`, unique run ID, log and exit receipt. No watchdog, no automatic p2, no second Isaac. Explicitly report that launch is exploratory and incomplete.
- [ ] **Step 2: At startup and at first checkpoint** inspect both ranks, CPU/GPU memory, `NaN`, nonfinite, simulation-view invalidation, process exit and failure logs. Capture actual run ID and source/checkpoint hash. If a severe error appears, stop the job and preserve checkpoint/log; do not silently change physics/contact or force a pass.
- [ ] **Step 3: Upon exit**, capture full deterministic same-run 512 environment ten-family per-family fractions, speed extrema and SHA. Any <90% is `hold`; identify gaps/tyre/walk specialists to develop *serially* based on observed failures. Do not launch student distillation or mixed training until teacher-system gate later passes, and do not call this run the 11-family promotion.

## Self-review and stop rule

- The plan only starts the **first** shared teacher experiment; selecting and training further teachers, building a training-only router, ten-family DAgger and mixed-course work each require later separately testable plans informed by this run. No claim that a single 900-iteration experiment achieves ten-family ≥90%.
- If command/sensor contract tests, PhysX smoke, contact-report interface, stop flag or isolation checks fail, leave all new runs stopped and report the blocker; do not lower geometry, relax collision, raise speed or substitute a fake teacher.
