# Go1 Entrance Reward and Strict Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide real short-gap-specific safe-support feedback and an exact one-first-episode-per-environment 512-slot deterministic gate, without altering geometry or hard safety thresholds.

**Architecture:** Add a pure first-episode collector to an isolated CLI overlay and a pure entrance-support scorer to an isolated reward helper. The runtime continues to use V132.1's only authoritative per-gap `paid` event for one-off bonus and success. A new guarded V132.2 runner overlays only pinned sources under the exclusive lock, records a complete trace, restores active sources, and never proceeds to junction if entrance fails.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/PhysX, unittest/pytest, Bash, SHA-256, JSON/JSONL.

**Spec:** `docs/superpowers/specs/2026-11-01-go1-entrance-real-gap-reward-and-gate-design.md`

## Global Constraints

- Final `combined` mesh, entrance 2–5 real gaps of 0.08–0.24 m, 0.65–1.10 m pads, ≥0.55 m void depth and 1.30–3.00 m roll-in unchanged.
- Actor 1073 inputs, 12 actions, GRU128; no terrain family, segment boundary or privileged geometry in actor; speed command 0.35–0.8 m/s.
- No weaker collision, fall, gap-void, stable-touchdown, all-gap-credit or finish criterion; no student distillation, watchdog or automatic p2.
- New run IDs, new source/config/report/checkpoint paths; never overwrite V132/V132.1 assets or seed-2028 diagnostics. Only one Isaac instance; lock, GPU, SHA and trap restoration before/after overlay.
- Train/eval use identical reward source, observation mask, disturbance and terrain family. Final holdout is a freshly verified unused seed, 512 unique env IDs, one deterministic episode each, ≥90%; stop queue at first failure.

## File map

- `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/cli.py`: isolated copy of the **remote active** CLI, modified only to select strict-first evaluation. Do not copy the older local `sim/source/go1_hier_sim/go1_hier_sim/cli.py`.
- `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/strict_gate_v1322.py`: pure first-episode collector and validation.
- `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py`: pure reward-only short-gap approach score.
- `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/runtime.py`: isolated V132.1 runtime with entrance-only shaping added; v132 gap-credit, tire and hard-failure logic unchanged.
- `v128_work/v132_randomized/v1322/tests/test_strict_gate_v1322.py`, `test_entrance_support_v1322.py`, `test_runtime_entrance_contract_v1322.py`: focused red/green and source integration tests.
- `v128_work/v132_randomized/v1322/scripts/verify_and_train.sh`, `configs/teacher_entrance.json`, `configs/manifest.json`: one writer, pinned overlay, strict gate, full traces and fail-closed restoration. Configs derive from V132.1 and get unique run/state paths.

---

### Task 1: Exact first episode per environment

**Files:** Create `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/strict_gate_v1322.py`, `v128_work/v132_randomized/v1322/tests/test_strict_gate_v1322.py`; modify isolated remote-active `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/cli.py` and `v128_work/v132_randomized/v1322/scripts/run_teacher_v1322.py`.

**Interfaces:** `FirstEpisodeCollector(num_envs: int, trace_required: bool)`; `.accept(episode_rows: list[dict], trace_rows: list[dict] | None) -> None`; `.complete -> bool`; `.finish() -> tuple[list[dict], list[dict]]` raises on missing/duplicate slot or trace mismatch. Rows have integer `env_id`; exactly one first episode per ID. CLI opt-in `--strict-first-episode` requires `--episodes == --num-envs == 512` and `--episode-trace`; regular CLI behavior untouched.

- [ ] **Step 1: Write red tests.** Use a three-slot fixture below and separate tests for missing slot, duplicate trace ID and absent trace; require one episode per slot. The new `FirstEpisodeCollector` is the production symbol that makes this fail before implementation.

```python
collector = FirstEpisodeCollector(num_envs=3, trace_required=True)
collector.accept([{'env_id': 0, 'success': False}, {'env_id': 1, 'success': True}],
                 [{'env_id': 0, 'required_gap_count': 2, 'credited_gap_count': 0},
                  {'env_id': 1, 'required_gap_count': 2, 'credited_gap_count': 2}])
collector.accept([{'env_id': 0, 'success': True}, {'env_id': 2, 'success': True}],
                 [{'env_id': 0, 'required_gap_count': 2, 'credited_gap_count': 2},
                  {'env_id': 2, 'required_gap_count': 2, 'credited_gap_count': 2}])
records, trace = collector.finish()
assert [row['env_id'] for row in records] == [0, 1, 2]
assert len(trace) == 3 and sum(row['success'] for row in records) == 2
```
- [ ] **Step 2: Run the tests before implementation** via `/root/env_isaaclab/bin/python -m unittest discover -s v128_work/v132_randomized/v1322/tests -p test_strict_gate_v1322.py -v`; require assertion failure because collector/CLI opt-in does not exist, not an import typo.
- [ ] **Step 3: Implement collector and CLI opt-in.** On each done batch merge stage trace, then accept only first records. Require a complete trace and all IDs before reporting. Add `--strict-first-episode` to the isolated parser. The driver passes `--strict-first-episode --episode-trace reports/v1322_go1-v1322-real-entrance-s01-p1-d1_trace.jsonl` and rejects incomplete reports.

```python
strict = FirstEpisodeCollector(count, trace_required=True) if args.strict_first_episode else None
# in the existing done-batch branch, after merge_episode_trace:
if strict is not None:
    strict.accept(episode_rows, traced)
else:
    for item in episode_rows:
        env_id = int(item['env_id'])
        if len(records) >= episodes and env_id in seen_envs:
            continue
        seen_envs.add(env_id)
        records.append(item)
        if trace_path is not None:
            trace_rows.append(trace_by_env[env_id])
# after loop, before aggregating:
if strict is not None:
    records, trace_rows = strict.finish()
    assert len(records) == count == episodes
# report from exactly these records, then check each by_env episode count is 1
```
- [ ] **Step 4: Run red tests to green, then focused CLI tests and existing gate tests.** Confirm one slot per env and no stale `episode_metrics`/trace mismatch; no change to default evaluation mode.
- [ ] **Step 5: Commit only Task 1 files** under a scoped V132.2 message; never stage unrelated dirty files.

### Task 2: Real entrance support shaping using authoritative credit

**Files:** Create `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py`, `v128_work/v132_randomized/v1322/tests/test_entrance_support_v1322.py`, `v128_work/v132_randomized/v1322/tests/test_runtime_entrance_contract_v1322.py`; modify `v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/runtime.py`.

**Interfaces:** `entrance_support_step(bounds, local_x, foot_x, foot_z, contacts, root_z, upright, forward_velocity, command_speed, failed, paid, entrance_mask) -> torch.Tensor[N]`. Bounds `[N,24,3]`; all other scalar tensors `[N]`, feet `[N,4]`. Return [0,1] for a not-yet-paid real short gap within near/far approach window only if ≥2 actually supported feet on near/far pads (not down in void), upright, root height ≥0.23 m, forward motion >0.1 m/s, no hard failure. Never expose bounds to actor. `safe_segment` for entrance equals the same `v132_event` already credited by `gap_clearance_step`, not a second scoring machine.

- [ ] **Step 1: Write red tensor tests** using `bounds[0,0]=[1.50,1.62,2.42]`, root x `1.35`, supported feet x `1.30` and z `0`, two contacts, root z `.32`, upright `.96`, forward speed `.4`, command `.5`: score >0. Repeat with forward speed `0`, contacts all false, foot z `-.10`, hard failure, already-paid gap, entrance mask false or NaN bounds: score 0. Add source integration test that `safe_segment` equals the same one-off `gap_clearance_step` transition, and other families/geometry remain unchanged.

```python
bounds = torch.full((1, 24, 3), float('nan')); bounds[0, 0] = torch.tensor([1.50, 1.62, 2.42])
score = entrance_support_step(bounds, torch.tensor([1.35]), torch.tensor([[1.30]*4]),
    torch.zeros((1,4)), torch.ones((1,4), dtype=torch.bool),
    torch.tensor([.32]), torch.tensor([.96]), torch.tensor([.4]), torch.tensor([.5]),
    torch.tensor([False]), torch.zeros((1,24), dtype=torch.bool), torch.tensor([True]))
assert 0 < score.item() <= 1
```
- [ ] **Step 2: Run tests red** with the same `unittest discover` command targeting `test_entrance_support_v1322.py` and `test_runtime_entrance_contract_v1322.py`; verify failures correspond to missing reward-only function/integration.
- [ ] **Step 3: Implement the pure helper and entrance-only wiring.** Use finite `all_gap_segments`, support contact-foot Z and X outside the void, `~paid`, no failure, and factors clamped into [0,1]. The runtime calls `entrance_support_step` with the interface above for entrance only and `safe_segment += v132_event * entrance_mask` after `gap_clearance_step`; no other scoring or failure branch changes. Weights remain TOML `0.8` and `6.0`; no V19 jump shaping for short gaps.

```python
valid = torch.isfinite(bounds).all(-1) & ~paid & entrance_mask[:, None]
start, far, end = bounds.unbind(-1)
window = (local_x[:, None] >= start - .45) & (local_x[:, None] <= end + .20)
pad_contact = contacts[:, None, :] & (foot_z[:, None, :] >= -.02) & (
    (foot_x[:, None, :] <= start[:, :, None] - .02) |
    (foot_x[:, None, :] >= far[:, :, None] + .02))
eligible = valid & window & (pad_contact.sum(-1) >= 2) & ~failed[:, None]
# q_* are [N] height/upright/forward-command ratios clamped to [0,1]
return (eligible * (q_height * q_upright * q_motion)[:, None]).amax(-1)
```
- [ ] **Step 4: Run focused tests green and existing V132.1 gap/tire score tests.** Check 1073/12/GRU128 contract and no actor geometry via existing project tests.
- [ ] **Step 5: Commit only Task 2 files** under a scoped V132.2 message.

### Task 3: Guarded integration and honest diagnosis

**Files:** Create `v128_work/v132_randomized/v1322/scripts/verify_and_train.sh`, `v128_work/v132_randomized/v1322/configs/teacher_entrance.json`, `v128_work/v132_randomized/v1322/configs/manifest.json`, and `v128_work/v132_randomized/v1322/reports/` outputs; preserve V132.1 assets.

**Interfaces:** A CPU-only `PREFLIGHT_ONLY=1` mode validates hashes, empty output paths, reward/obs/disturbance parity, GPU≤256 MiB each and lock; overlay of candidate geometry (identical V132.1 SHA), V132.2 runtime/helper/CLI and V131 reward TOML; every exit traps exact restoration. Training stage uses seed 42 and a new run ID, only after pilot. Strict gate uses a freshly checked unused holdout seed and outputs 512-entry JSONL trace, checkpoint hash, 512-slot JSON and final decision. No junction launch on failure.

- [ ] **Step 1: Write red contract tests** for source SHA pins, 512 distinct first-episode slots, non-overwrite of old reports/checkpoints, gap-count telemetry keys in every trace row, and train/eval environment parity; run them and require expected assertion failure before writing scripts.
- [ ] **Step 2: Implement isolated config/manifest/runner and verify CPU-only preflight.** Derive config from V132.1 `teacher_entrance.json`, change every state/log/run/report path to a V132.2 name, retain reward TOML and matching masks/disturbances. Pin SHA256 of active and candidate CLI/runtime/geometry/config/helper and record backups. If any GPU exceeds 256 MiB or the exclusive lock is unavailable, abort before source overlay. Restore exact active hashes on every exit. Ensure `GO1_V29_TEACHER_DISTILL`, `GO1_WATCHDOG` and `GO1_AUTO_P2` are unset.

```bash
root=v128_work/v132_randomized/v1322
exec 9>reports/v7_sim.lock
flock -n 9 || exit 86
# validate GPU memory + active/candidate SHA + fresh output paths before any cp
declare -A live=(
  [cli.py]=source/go1_hier_sim/go1_hier_sim/cli.py
  [runtime.py]=source/go1_hier_sim/go1_hier_sim/envs/runtime.py
  [geometry.py]=source/go1_hier_sim/go1_hier_sim/envs/geometry.py
  [config.py]=source/go1_hier_sim/go1_hier_sim/envs/config.py
)
restore_active_sources() {
  rc=$?; trap - EXIT INT TERM
  for part in cli.py runtime.py geometry.py config.py; do
    cp "$root/reports/backup_$part" "${live[$part]}"
  done
  rm -f source/go1_hier_sim/go1_hier_sim/strict_gate_v1322.py
  rm -f source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py
  for part in cli.py runtime.py geometry.py config.py; do
    expected=$(/root/env_isaaclab/bin/python - "$root/configs/manifest.json" "$part" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['baseline_hashes'][sys.argv[2]])
PY
)
    actual=$(sha256sum "${live[$part]}" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || exit 90
  done
  exit "$rc"
}
for part in cli.py runtime.py geometry.py config.py; do
  cp "${live[$part]}" "$root/reports/backup_$part"
done
trap restore_active_sources EXIT INT TERM
declare -A candidate=(
  [cli.py]="$root/source/go1_hier_sim/go1_hier_sim/cli.py"
  [runtime.py]="$root/source/go1_hier_sim/go1_hier_sim/envs/runtime.py"
  [geometry.py]="$root/source/go1_hier_sim/go1_hier_sim/envs/geometry.py"
  [config.py]="$root/source/go1_hier_sim/go1_hier_sim/envs/config.py"
)
for part in cli.py runtime.py geometry.py config.py; do
  cp "${candidate[$part]}" "${live[$part]}"
done
```
- [ ] **Step 3: Run full project test suite under candidate overlay, then 512-env PhysX smoke on all four real families** with receipt and SHA restoration; no training on any failure.
- [ ] **Step 4: Replay old checkpoint only as development diagnosis** with strict one-per-env trace; assert 512 rows, 512 unique IDs, per-row required/credited counts and terminal flags, and analyze by gap count/first failure. Do not call seed 2028 a final holdout again.
- [ ] **Step 5: Run a bounded 64-env A/B pilot** with exactly one reward variable changed (entrance-only helper), same initial checkpoint/geometry/dev seed, inspect support-term nonzero without standing or void credit, true per-gap credits and failure positions; stop if support is absent or safety degrades. Record hashes and logs.
- [ ] **Step 6: If pilot supports the hypothesis, train isolated entrance with 50 critic-only + 600 actor updates.** Use new run ID and checkpoint directory. Evaluate once on a newly verified unused final seed using deterministic actions, strict 512 unique env slots and complete trace. Accept only ≥90% with hard failures/each actual gap counted; else stop. Do not automatically train other teachers until entrance passes and its evidence is reviewed.
- [ ] **Step 7: Commit scripts/tests/doc changes only; archive receipt, tests, trace, gate JSON, checkpoint SHA and restoration hash.** Report any 90%-failure as failure, not completion.

## Self-review gates

- [ ] Confirm spec coverage: real geometry/hard checks preserved; entrance-only safe support and authoritative paid event; strict 512 slots/trace; isolation, parity, fresh holdout and failure stop.
- [ ] Confirm no placeholder or undefined interface names and that task ordering is collector → scorer → integration.
- [ ] Freshly verify every claimed pass from command exit codes and saved outputs; do not infer that PhysX initialization alone proves trained crossing.
