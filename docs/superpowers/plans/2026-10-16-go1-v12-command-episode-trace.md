# Go1 v12 指令契约与逐回合诊断实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement each task with RED→GREEN checks. The workspace is NOT Git: use the isolated `v12_work/`, original SHA manifest and untouched v11 artifacts instead of claiming commits. User has directed continuation without repeated approval prompts.

**Goal:** In an isolated diagnostic variant, enforce real Stage-1 actor commands within the agreed 0.35–0.8 m/s plan band without using family truth, and link first-failure/reward evidence to completed evaluation episodes.

**Architecture:** Resolve effective bounds from the existing `GO1_TASK_PLAN` at config creation, then bound the observed-scan-adjusted command in one pure function. Expose opt-in, strictly matched `done` records via existing `pop_stage_records()`; count reward events only once per control tick, before reset, without adding actor inputs or changing normal report schema.

**Tech Stack:** Python3.11/PyTorch2.7/pytest; Isaac Lab2.3.2; remote `/config/Desktop/sim` and local `E:/tmp-issac/v12_work`.

**Spec:** `docs/superpowers/specs/2026-10-16-go1-v11-unified-roster11-design.md`; diagnosis `docs/v11-improvement-research-2026-10-16.md`. This is a diagnostic amendment, not certification of the prior v11 checkpoint.

## Global Constraints

- Exactly 11 terrain families; 0.35–0.8 m/s planned command; 1073 actor input, 12 actions, GRU128 unchanged.
- Never use terrain-family/gap segment/contact truth in actor command or deployment selection. Privileged information remains allowed only in training reward and diagnostic output.
- Preserve all old models/reports/video and `lp_phased_stop.flag`. No automatic p2, watchdog, parallel Isaac, or claim of leap/safety/11-family passage.
- If isolated baseline/tests/physical run fail, stop that phase and save raw logs. All traces are diagnostic, not a replacement for a same-session512 gate.

---

### Task 1: Effective speed-band resolution and bounded command

**Files:** Create `v12_work/source/go1_hier_sim/go1_hier_sim/envs/speed_band_v12.py`, create `v12_work/tests/test_v12_speed_band.py`; modify isolated `envs/config.py` and `envs/runtime.py`. Do not modify production source in place.

**Interfaces:** `resolve_speed_band(speed_min: float|None, speed_max: float|None, default_speed: float, plan_path: str|None, max_speed: float) -> tuple[float,float]`; `bounded_observable_command(sampled: torch.Tensor, scan_scale: torch.Tensor, low: float, high: float) -> torch.Tensor`.

- [x] Write tests with hand-derived values: plan `{speed_min: .35, speed_max: .8}` plus CLI None resolves (.35,.8), explicit matching values are accepted, explicit mismatch and nonfinite/out-of-range values reject. A sampled `[.8,.35]` with scan scales `[1.,.45]` returns `[.8,.35]`; two same sampled speeds with different *family* labels receive identical output because function has no family input. Tests fail while module absent.
- [x] Run remote focused pytest with `PYTHONPATH=$PWD/v12_work/source/go1_hier_sim` and `v12_work/tests/test_v12_speed_band.py` to record RED.
- [x] Implement pure functions, wire `make_config()` to resolve the plan bounds before cfg speed validation, and `runtime._desired_stage1_command()` to bound sampled forward speed after its existing **observable** scan scale. Remove truth-family-based forward boost only; keep reward-specific family masks and v9 contact-mode opt-in semantics unchanged. Lateral/yaw unchanged.
- [x] Run focused pytest GREEN and full 270 baseline tests with isolated package first in PYTHONPATH. Actor 1073/12/GRU128 remains unmodified; the existing v11 checkpoint loaded in both two-env and 512-env PhysX. Final isolated suite: 283 passed, 76 subtests passed.

### Task 2: Opt-in terminal episode trace, not post-success fall logs

**Files:** Modify isolated `cli.py` and `envs/runtime.py`; create `v12_work/tests/test_v12_episode_trace.py`.

**Interfaces:** `--episode-trace PATH` opt-in JSONL; each row keyed by actual completed `env_id` and per-env episode index, with episode `success/fallen/collision/out_of_bounds/fell_into_gap`, terminal `course_x/y`, speed, and per-episode reward-event counters. Never read post-reset pose for terminal location.

- [x] Write failing pure tests for matching `episode_metrics.env_ids` against `pop_stage_records()` after `done`; discard *only* initial zero-step reset records; duplicate/missing/mismatched or nonfinite data raises rather than fabricating x=0. A success and a later fall in the same environment must produce two distinct episode rows, not one failure.
- [x] Add counters for `supported_short`, `safe_segment`, `tire_progress` in runtime at cached reward build once per control tick; snapshot count and first trigger x into the existing stage record before reset, reset counters per environment. Unit-test reset and no double-count when reward cache is queried twice in one tick. These values are diagnostic only.
- [x] At evaluation with the flag, drain the initial reset records, then pair each completed episode with matching stage record at `done` before output; atomically save the trace and output SHA. Default evaluation outputs/metrics and all actor observations remain unchanged. Run focused and full isolated tests.

### Task 3: Serial measured contrast and safety decision

**Files:** `v12_work/reports/` for manifest, test logs and trace; append evidence to `docs/v11-improvement-research-2026-10-16.md` and `sim/agent.md` only after verification.

- [x] Stage isolated package remotely with a SHA manifest; no production overwrite. Confirm stop flag and no other Isaac process. Run 2-env prefix and 512-env diagnostic on the original v11 checkpoint. Verify sampled/scan-target command is within the plan band and **executed command max≤0.8**; note that router smoothing begins from0 and permits an initial executed minimum below0.35. `--episode-trace` rows must match actual completed records, not after-success raw `FALL_EVENT` lines. Evidence: `v12_work/reports/v12_diag_input_sha256.txt`, `v12_work/reports/diag_v12_commandtrace_512.jsonl`; original v11 512 trace separately reproduces official report SHA.
- [x] Compare against original same-plan v8/v11 frozen evidence; report decline by family and speed level in `docs/v11-improvement-research-2026-10-16.md`: corrected frozen v11 gap performance collapsed, so **no production deployment or new training** in this diagnostic tranche. Separately RED→GREEN tested the inherited 2% standing-sampler issue; closing it alone lifts old short-gap from47/53 to47/50 while combined, long and tyre remain below90. Do not train or advance p2 until the next controlled compliant-training design and unresolved contact controls are addressed.
