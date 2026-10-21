# Go1 Full-Roster Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the approved mixed track alongside a full-length single-obstacle course for every constituent obstacle; advance only when all 11 families exceed 90%.

**Architecture:** Extend the existing tile generator and LP-ACRL categorical roster without changing the actor observation/reward tensor shapes. Assign 46–47 of 512 training environments per family, gate all planned families, and start a fresh curriculum state from the s27 checkpoint under a separate base ID. Only one Isaac simulation may run on the two-A10 host.

**Tech Stack:** Python 3, NumPy, trimesh, PyTorch, Isaac Lab, bash.

**Spec:** `docs/2026-10-11-rc-course-rewrite-design.md` §11–12 (supersedes its historical §4/§7).

## Global Constraints

- Worktree `/config/Desktop/sim`, host `abc@10.200.212.241:8022`; no `.git` repository exists there, so no commit can be made.
- Two A10 GPUs, `/dev/shm` 64 MB; never start concurrent Isaac simulations.
- Keep checkpoint observation/reward shape unchanged, speed 0.35–0.8 m/s and the existing DR schedule.
- All 11 families must be present in every 512-environment plan; each must have at least 40 evaluation episodes and ≥0.90 completion before advancing.
- The accepted `combined` map, including its narrow 8-up/8-down section, is unchanged by this roster expansion.

---

### Task 1: Make every single-obstacle tile span the route

**Files:** `source/go1_hier_sim/go1_hier_sim/envs/geometry.py`, `source/go1_hier_sim/go1_hier_sim/envs/task_curriculum.py`, `tests/test_geometry.py`.

**Interfaces:** `build_tile(kind, difficulty, rng, obstacles)` returns `course_profile.blocks`; `TERRAIN_FAMILIES` contains the complete roster.

- [x] Write `test_each_obstacle_has_a_dedicated_full_length_course`: assert only the family’s obstacle block kinds, final block x in [30,33], and a dedicated `stairs_narrow_8_mixed` with multiple 8-up/8-down sections.
- [x] Run `./sim.sh test -q -k 'roster_matches or dedicated_full_length'` and observe the missing-family/short-course failures.
- [x] Add `stairs_narrow_8_mixed`, `slope_up`, `slope_down` to the LP roster and geometry allowlists; continue stair/grade/tyre blocks to x≈33 at all difficulties. The narrow stair course repeats `stairs_block(..., steps=8, tread_range=(0.35, 0.35), riser_height=shared_riser)`.
- [x] Write and fail `test_downhill_is_a_visible_terrain_surface_not_buried_under_base`; cut the z=0 slab under the negative-height walking corridor while keeping side ground, then rerun it green.
- [x] Run all tests and a CPU 11-tile `build_scene(..., tasks=[(family, 0.1) ...])` smoke check.

### Task 2: Guarantee every planned family passes the same gate

**Files:** `scripts/run_phased_curriculum.py`, `scripts/lp_acrl_stage.py`, `tests/test_full_roster_gate.py`.

**Interfaces:** `rebalance_family_plan(plan, TERRAIN_FAMILIES, difficulty_max) -> None`; `gate_decision(plan, report, gate_spec) -> (decision, stats)`.

- [x] Write and fail the test where a report lacks one planned family; ensure it is recorded at zero episodes and the gate holds.
- [x] Make `gate_decision` initialise its family table from `plan['instances']` rather than only the observed report.
- [x] Write and fail the test where a finite sampler draft omits one family; extract `rebalance_family_plan` and assign every name in `TERRAIN_FAMILIES` an equal share regardless of sampled draft.
- [x] Rerun targeted and full unit tests: `167 passed, 70 subtests passed` on 2026-10-12.

### Task 3: Isolate the new curriculum and launch safely

**Files:** `configs/curriculum_phases_full.json`, `scripts/watch_phased_curriculum_v3.sh`, `docs/2026-10-11-rc-course-rewrite-design.md`.

- [x] Save a new config with `state=lp_full_roster_state.json`, `sampler_state=lp_full_roster_acrl_state.json`, `min_family_episodes=40` and warm-start checkpoint `checkpoints/stage1/2026-10-11_go1-rc-roster-lvl3-s27-p1-d1/model_899.pt`.
- [x] Stop the old watchdog and old driver, allow its already-running s27 simulator to produce `model_899.pt`, verify no old simulator remains, then start `watch_phased_curriculum_v3.sh 2026-10-12_go1-full-roster11 configs/curriculum_phases_full.json`.
- [x] Verify the new stage plan enumerates all 11 families (46–47 environments each), exactly one two-worker simulator is running, and s01 has begun warm-up iterations.
- [ ] At s01 gate, save the deterministic evaluation and list completion/episode counts for all 11 families; do not call the new course complete until each is ≥0.90 in the same gate.
