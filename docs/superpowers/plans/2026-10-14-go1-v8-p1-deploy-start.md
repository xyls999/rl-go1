# Go1 v8 Dense Terrain / Map Attention P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a separately versioned v8 dense 11-family course and warm-startable map-attention actor, repair known gap telemetry and start exactly one new p1 training run without promoting the curriculum.

**Architecture:** Reuse stage1 observation 1073/action 12 and RSL GRU128. New attention is a gated residual beside the **v7b v1 `ActorCore.encoder` 1024→128→32** (verified directly in checkpoint), not `hier_v2.EncoderActorCore`; legacy checkpoint loading is preserved. Geometry changes occupy only the v8 copy of the active course source; old checkpoints/reports/stop flag remain immutable. A reward/telemetry module gates long-gap signals using live geometry and state, not a deployment-time oracle label.

**Tech Stack:** Isaac Lab 2.3.2/Isaac Sim 5.1, Python 3.11, PyTorch 2.7, rsl-rl 3.1.2, NumPy/trimesh, pytest. See `sim/AGENTS.md` and `sim/agent.md` before deploy.

**Spec:** `docs/superpowers/specs/2026-10-14-go1-v8-dense-terrain-map-control-design.md`.

## Global Constraints

- 11 families, 512 training/evaluation environments, seed2026, commands 0.35–0.8 m/s, official `stage1-v2` gate in **one deterministic batch**, 11/11 families each ≥90% to promote; startup is not success.
- One Isaac simulator at a time, two training ranks, 64MB `/dev/shm`. `lp_phased_stop.flag` stays. No watchdog, no automatic p2, no overwriting v7a/v7b/s22 artefacts.
- No real Go1 sensor inventory: teacher scan only in simulation, no privileged actor labels; actual depth student/physical commissioning excluded from p1-start claim.
- Workspace lacks Git: save SHA-256 before/after manifest and copied originals to timestamped `reports/v8_backup_*`, don't pretend to commit or create a Git worktree.
- New configs, state, logs, run-id and checkpoint directory; original `scripts/torchrun_sim.sh` SHA-256 `9a1f496c6f1b575f2150aaa28c5ebca42b53dc3e6bf3d1080a297ad471d88a55` stays unchanged.

---

### Task 1: Back up active remote source and establish baseline

**Files:** Read from `/config/Desktop/sim/source/go1_hier_sim/go1_hier_sim/{envs/{geometry,runtime,config}.py,hier_v2/model_v2.py,checkpoints.py,cli.py,training/on_policy.py}`, and `configs/rewards_stage1_gaps_v7b_supported_crossing.toml`. Create local `E:/tmp-issac/v8_work/` copies and remote timestamped `reports/v8_backup_<UTC>/` copies, `reports/v8_deploy_manifest.json`.

**Interfaces:** Output SHA paths for every file that later tasks may modify. Consume `sim/AGENTS.md` and full `sim/agent.md`.

- [ ] Step 1: Ensure no active `scripts/sim.py train/evaluate` or curriculum driver; stop flag exists and `model_899.pt` hashes to `fd04f3a6f1c5d03c6d6992be722a6924ab2ca9015112a2bdc6aa8242c2d4e631`.
- [ ] Step 2: Copy remote current source into staging and `reports/v8_backup_<UTC>/`; record original SHA for each path and all v7b state/report/torchrun before changing any code.
- [ ] Step 3: Execute remote baseline `./sim.sh test` and preserve stdout in `reports/v8_baseline_tests.log`. If failing, stop, diagnose and do not deploy.

### Task 2: Build denser valid course with a measurable high step

**Files:** Modify staged `envs/geometry.py`, create `tests/test_v8_geometry.py`.

**Interfaces:** `build_tile('combined',difficulty,np.random.default_rng(seed),False)` returns `mesh.metadata['course_profile']`; each `block` has `kind/start/end`, tyre blocks have `tyres`, stairs have `riser_heights/tread_depths`.

- [ ] Step 1: Write failing tests that for seeds `[2026,2027,2030]`, difficulty `[0.0,0.1]`: first two tyre blocks each have ≥16 tyres (4 lateral ×4 forward) at difficulty≤0.3 and ≥12 tyres at difficulty>0.3; last has ≥12; first three gaps `width<=.24`, next three `.29<=width<=.42`, each gap preserves its original pad, all ends `<=33.0`, all rises return to datum, final staircase has at least one rise ≥0.16m and that riser's tread ≥0.45m. A separate test keeps all 11 families' `build_tile` paths cookable and nondegenerate. Run `./sim.sh test` against test + unmodified staging and see correct failing assertions (not import errors).
- [ ] Step 2: Use `mixed_tires(...,4 if difficulty<=.3 else 3)` for the first two tyre bands; keep middle ordinary stairs at 3 risers per direction while the final staircase retains 8, redistributing former final tyre rows to fit before 33.0. Introduce `riser_heights` sequence with one ≥0.16m final stair rise and a ≥0.45m matching tread; construct descent symmetrically. Update block metadata and avoid a zero-volume end plate. Keep pad .65 short and 1.10 long.
- [ ] Step 3: Run the targeted tests green, then all tests; generate example metadata table (x ranges, widths, step rises, collider faces) for seed2026 and save `reports/v8_geometry_profile.json`. One PhysX smoke must include an actual high stair tile; block run if cooking fails or feet/body have no collision.

### Task 3: Correct gap instrumentation and bound actual long-gap reward

**Files:** Create `envs/gap_diagnostics_v8.py`, `tests/test_v8_gap_diagnostics.py`; modify staged `envs/runtime.py`; create `configs/rewards_stage1_gaps_v8_dense_attn.toml` copied from v7b with only justified, documented new weights/fields.

**Interfaces:** Pure helper `foot_in_void(foot_z: torch.Tensor, depth_m: float)->torch.Tensor[N]`; `gap_clear_event` consumes observed positive lip/window, long-gap width on training side, actual four-feet flight and safe post-lip landing, returns per-env one-shot boolean. Runtime never exposes ground-truth width/family to actor. Preserve `get_metrics` keys and gate protocol.

- [ ] Step 1: Write failing tests: `(foot_z.min < -.30) & ((foot_z.max-foot_z.min)>.30)` or all feet below datum flags; a 0.50m stair crest with all feet ≥0 does not flag. `gap_clear_event` rejects short gap, normal trot on flat, flight with no landing and repeated event; accepts one long-gap clearance with stable touchdown and ≤ one bounded payout.
- [ ] Step 2: Implement helper and replace the old negative `foot_spread`; wire termination-step debug logs with measured foot min/max, contact state, local x and orientation. Connect bounded gap-clear only to genuinely registered long-gap window and a crossed landing, not to all gap-family ordinary flight. Keep short-gap crossing supported; cap bonus and do not mask real failure codes.
- [ ] Step 3: Run targeted red/green tests plus all suite. Validate 16-env short/long episodes with `GO1_DEBUG_GATE=1`, confirm no plain-flight payouts and no nonfinite reward; new TOML must be parsed by strict loader. This does not certify improved long-gap skill.

### Task 4: Versioned actor-only attention without fake sensor confidence

**Files:** Create `hier_v2/map_attention_v3.py`, `tests/test_map_attention_v3.py`; modify `hier_v2/model_v2.py`, `training/on_policy.py`, `cli.py`, `checkpoints.py` in staging.

**Interfaces:** `MapAttentionActorCore(stage='locomotion')` reads `[N,1073]` and recurrent `[1,N,128]`, outputs `[N,12]` and `[1,N,128]`; `HierActorCriticV3` inherits the v1 `HierActorCritic` PPO API with independent v1 `CriticCore`, matching v7b checkpoint exactly. New model state includes `map_adapter.*`, making `detect_actor_variant` return `v3`; old `encoder.*` remains byte-for-byte shape-compatible; legacy states retain v1/v2 routing. `GO1_POLICY_VARIANT=map_attn_v3` opt-in **only** for new stage1 train; evaluation loads actor by checkpoint state automatically; incompatible warm-start keys still fail.

- [ ] Step 1: RED tests: zero residual v3 forward equals the checkpoint-verified v7b v1 with identical inherited weights, invalid all-512 scan remains finite, masked cells cannot affect attended latent, temporal `[T,N,D]` and single `[N,D]` both shape-correct, `load_actor` round-trips v1, v2 and v3 distinct variants; unrecognised `GO1_POLICY_VARIANT` fails before simulation.
- [ ] Step 2: Build local 2-channel 16×32 scan CNN to 8×16 tokens; apply valid masking, global pooling and 4-head query from 49 nonprivileged prefix; project a 32-D residual into the old 32-D scan latent with **zero-initialized** final layer, leaving old v1 ActorCore encoder/GRU/head unchanged at initialization. All-invalid fallback explicitly unmask one zero dummy token. Keep critic v1 independent.
- [ ] Step 3: Update runner class registry; strictly whitelist missing `actor.map_adapter.*` in `--initialize-from` **v7b v1** checkpoint, preserve missing `log_std` only, reject all unexpected keys. Give new adapter a bounded separate optimiser learning rate; preserve warm-start slow actor and critic rates, metadata `policy_variant=map_attn_v3`, original interface manifest1073/12 and real export graph. Run target and full tests green plus checkpoint roundtrip test. Never label finite-hit indicator a calibrated variance.

### Task 5: Preflight, deploy, smoke and start one p1 run

**Files:** Create independent `configs/curriculum_phases_v8_dense_attn.json`, `reports/v8_deploy_manifest.json`, `reports/v8_p1_start.log` and `reports/v8_p1_status.json`; modify only source/config files above on remote after backup. Keep `scripts/torchrun_sim.sh` unchanged.

**Interfaces:** Use unchanged `scripts/run_phased_curriculum.py --base-id 2026-10-14_go1-v8-dense-attn --config configs/curriculum_phases_v8_dense_attn.json --max-stages 1`, `GO1_POLICY_VARIANT=map_attn_v3`, `GO1_REWARD_CONFIG_PATH=<v8.toml>`. New independent v8 state/sampler/results paths and run ID. Driver may gate at end but must never automatically run p2.

- [ ] Step 1: RED test asserts config has exactly 11-family roster, 512 envs, speed [0.35,0.8], max_stages=1, new state/sampler/checkpoint IDs, missing/modified stop flag and dirty sim lock block start. Set `init_checkpoint` to immutable v7b model_899, specify 150 critic warmup, 900 max iterations, same two-rank topology; tests green.
- [ ] Step 2: Copy staged files to remote one writer at a time after SHA backup; `./sim.sh test` complete, `./sim.sh check --stage stage1 ...` small/no camera, one small train smoke under a **throwaway separate run ID**, verify generated checkpoint has `policy_variant=map_attn_v3`, finite actor/critic, 1073/12. Ensure smoke cannot overwrite actual p1 or v7b artefacts.
- [ ] Step 3: Preflight exact v7b model/state/stop/torchrun hashes and no active sim. Start only `--max-stages 1` under `flock reports/v7_sim.lock` with `setsid`/unbuffered stdout, capture PID/PGID and unique log; verify log shows first nonzero PPO iteration and new checkpoint/output directory before calling it **started**. On any error, preserve partial diff/artifacts, no fallback into old driver, no p2, no gate promotion.
- [ ] Step 4: Save timestamped start evidence: immutable deployed SHA manifest, test log, sim validation, command/PID, log excerpt with iteration, output path, GPU topology; tell user *only after verified*, explicitly not passed 11-family gate and not real-world validated. No watchdog/periodic user-facing reports.

### Observed v8 p1 start (2026-10-15 UTC)

- Backups: `reports/v8_backup_20261014T171746Z/manifest.json`; remote deployed-source SHA manifest: `reports/v8_deploy_manifest.json` (`500cf60eeab96a3dfcf0e71fc1ab37440fac6a975a53841abe2684295799a613`). v7b `model_899.pt` still hashes to `fd04f3a6f1c5d03c6d6992be722a6924ab2ca9015112a2bdc6aa8242c2d4e631`; `scripts/torchrun_sim.sh` is unchanged.
- RED→GREEN unit regression cycles completed for gap telemetry/event, long-gap bounds, actor attention, full JIT graph, exact v7b warm-start, reward and v8 curriculum config. Remote `./sim.sh test`: **214 passed in 26.09s**, `reports/v8_export_green.log`. PhysX checks: `reports/v8_physx_check16.log` and `reports/v8_physx_check512.log` both contain `CHECK PASSED`; this only verifies cooking and finite initial stepping, not traversability/torque at the 0.17 m rise. The 16-env v7b-on-v8 gap diagnostic report is `reports/evaluate_stage1_2026-10-14_go1-v8-gap-diagnostic16.json` (0/16 completion; not a gate).
- Independent two-rank 16-env PPO smoke completed 3 iterations; `reports/v8_train_smoke16.log`, `checkpoints/stage1/2026-10-14_go1-v8-attn-smoke16/model_2.pt`. Checkpoint metadata `policy_variant=map_attn_v3`, `capability_validated=false`, finite actor/critic; full graph export roundtrip at `reports/v8_smoke_export/locomotion.json` passes. Actual v7b policy loading into zero-residual v8 actor gave **bit-exact** synthetic action and GRU state.
- Started one detached p1 driver PID **113415**, run `2026-10-14_go1-v8-dense-attn-s01-p1-d1` with **two ranks, 512 env/rank, 11 families, 900 iterations and first 150 critic-only warmup**; `--max-stages 1`, no watchdog or p2, stop flag kept. `reports/v8_p1_status.json` captured active rank processes and **PPO iteration 11/150 with 589824 aggregate timesteps** at 2026-10-15T00:06:24Z. Training log `v35_phased_2026-10-14_go1-v8-dense-attn-s01-p1-d1.log`; initial checkpoint `checkpoints/stage1/2026-10-14_go1-v8-dense-attn-s01-p1-d1/model_0.pt`. No v8 deterministic 512-env per-family gate yet; decision remains **hold**.
- Deliberate gaps: no actor update before the 150 critic warmup finishes; no observed stable high-step traversal, long-gap jump/landing, 11-family ≥90%, camera student or real Go1 validation. Do not treat the unit tests/smoke as evidence of those skills.

### Task 6: Later M3 continuation after p1 measurements (separate implementation plan)

**Files:** No code in p1-start deliverable. Research/update `docs/superpowers/specs/2026-10-14-go1-v8-dense-terrain-map-control-design.md` only after p1 gate/flight telemetry and real sensor inventory.

- [ ] Step 1: Use p1's actual 512-environment gate, not one-env video, to decide whether a separate jump policy with actual `jump-checkpoint` and safe hand-over is required; if so, train/validate it before claiming long gaps solved.
- [ ] Step 2: Train true camera-to-map student at 20Hz only with actual depth frames, timestamp alignment, height and calibrated uncertainty where data supports it; verify all-unknown and stale-sensor stop, then re-run all 11 families with student driving. Camera model weight and real Go1 sensor calibration are independent deliverables.
