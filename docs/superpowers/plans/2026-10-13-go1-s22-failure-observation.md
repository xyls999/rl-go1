# Go1 s22 Failure Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop s22 at its latest saved `model_550.pt` checkpoint per the user's changed instruction, prevent s23, and record that partial checkpoint's behavior on all 11 terrain families, including multiple `combined` failures, before altering its behavior.

**Architecture:** The watchdog honors `lp_phased_stop.flag`, but its running driver does not. An initial temporary next-launch guard was installed while s22 was training. The user then explicitly authorised stopping immediately: preserve `model_550.pt`, stop only the identified s22 launcher/workers and driver, restore the launcher byte-for-byte, retain the watchdog stop flag, verify no Isaac process remains, and only then run sequential (never concurrent) single-instance diagnostic recordings for all 11 families. s22 has **no completed gate**; s21 remains the last valid deterministic gate.

**Tech Stack:** bash, Python 3, PyTorch/Isaac Lab, ffmpeg for offline review.

**Spec:** User approval in chat, 2026-10-13: pause after s22 and observe the failure before changing the course or jump policy; `docs/2026-10-11-rc-course-rewrite-design.md` §11–12 specifies the immutable mixed layout and 11-family gate.

## Global Constraints

- Remote workspace `/config/Desktop/sim`, host `abc@10.200.212.241:8022`; local working directory `E:/tmp-issac`. Neither workspace is a Git repository; preserve original launcher backup and hashes instead of claiming commits.
- Two A10 GPUs and `/dev/shm` 64 MB: **never** run a second Isaac simulation concurrently with training/evaluation.
- Preserve the user-approved complete `combined` tile, 11-family curriculum state, actor observation/reward tensor shapes, checkpoint, and speed range 0.35–0.8 m/s.
- User override at 16:27 UTC: s22 stopped mid-training at `model_550.pt`; **no s22 gate/history may be claimed**. The last completed deterministic gate is s21. Diagnostic videos must be labelled `s22_partial_model550` rather than s22 final.
- Do not claim that an old `d2/d3_combined_flat` video or an annotated terrain video reproduces current `combined` failure.

---

### Task 1: Stop s22 now at saved model_550 and block auto-relaunch

**Files:** Temporarily modify remote `scripts/torchrun_sim.sh`; create remote `lp_phased_pause_after_s22.flag`, `lp_phased_stop.flag`, and `scripts/torchrun_sim.sh.before-s22-pause` backup. Restore the launcher byte-for-byte afterward.

**Interfaces:** Watchdog reads `lp_phased_stop.flag`; driver calls `scripts/torchrun_sim.sh train` only after it writes the previous stage's state; deterministic gate uses `sim.sh`.

- [x] **Step 1:** Read UTC time, current driver/watch/torchrun PIDs, latest state entry, exact s22 progress, and launcher SHA-256. Abort if s22 is already complete and s23 has launched; that case requires a separate safe boundary. At 16:08 UTC s22 was 273/900; original SHA-256 `9a1f496c6f1b575f2150aaa28c5ebca42b53dc3e6bf3d1080a297ad471d88a55`.
- [x] **Step 2:** Save a timestamped byte-for-byte launcher backup and place one guard immediately after `sim_workspace` initialization: `if [[ -f "${sim_workspace}/lp_phased_pause_after_s22.flag" && "${1:-}" == train ]]; then echo 's22 pause: refusing next train' >&2; exit 75; fi`. Use atomic replacement, retain existing permissions, verify `bash -n` and compare the unguarded remainder with the backup. Never run an unguarded wrapper test that would launch Isaac.
- [x] **Step 3:** Create both pause and watchdog stop flags. Guard test exited 75 without launching Isaac; s22 torchrun PID 58930 and workers 59028/59029 remained active at 16:11 UTC. Watchdog logged STOP and exited at 16:11 UTC; driver PID 50419 remained active.
- [x] **Step 4:** Per user override, stop s22 at 551/900 after `model_550.pt` was saved (SHA-256 `d047ecea7accf0443165e7651b65572bd39f467e5c8e94744f917f73e66ffbac`). Graceful SIGTERM triggered an Isaac mutex assertion in the workers; verified process identities and used targeted SIGKILL. At 16:32 UTC no train/eval/driver/watch process remained and GPU memory dropped to its non-simulation baseline. `lp_full_roster_state.json` still ended at s21, as required for an incomplete s22.
- [x] **Step 5:** Restored `scripts/torchrun_sim.sh` byte-for-byte (SHA-256 `9a1f496c6f1b575f2150aaa28c5ebca42b53dc3e6bf3d1080a297ad471d88a55`), passed `bash -n`, removed temporary pause guard flag, retained `lp_phased_stop.flag` to prevent watchdog relaunch. No s22 or s23 deterministic gate may be inferred.

### Task 2: Record the present checkpoint on all 11 families

**Files:** Create eleven `reports/diagnostic_s22_<family>_plan.json` plans and corresponding `media/full_roster_s22_partial_model550_<family>.mp4`, evaluation report/log and manifest entries. Add two extra `combined` variants if the first clip does not clearly identify its failure.

**Interfaces:** Each `GO1_TASK_PLAN` is a one-instance copy of an actual s22 instance of that family, selecting difficulty closest to 0.1 (ties by original index). A one-family recording uses `DISPLAY=:1 GO1_TASK_PLAN=reports/diagnostic_s22_<family>_plan.json ./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless --device cuda:0 --num-envs 1 --episodes 1 --episode-seconds 133 --seed 2026 --follow-env 0 --show-direction-arrows --video-seconds 133 --record-video media/full_roster_s22_partial_model550_<family>.mp4 --checkpoint checkpoints/stage1/2026-10-12_go1-full-roster11-s22-p1-d1/model_550.pt`. The single-environment reseed makes this a representative clip, not a byte-identical replay of a 512-env gate tile.

- [x] **Step 1:** Created 11 one-instance plans and a manifest in `reports/diagnostic_s22_partial_model550_*` from actual s22 plan indices 0, 47, 94, 141, 188, 238, 282, 328, 374, 420, 466. Each selected difficulty is 0.1; speed bounds are 0.35–0.8. Confirmed checkpoint SHA-256 `d047ecea7accf0443165e7651b65572bd39f467e5c8e94744f917f73e66ffbac`, stopped watchdog/driver/train, checked no Isaac workers, and verified X display socket before the combined recording started at 16:40 UTC.
- [x] **Step 2:** Eleven families finished serially by 18:26 UTC, extra full-length `combined_full` finished 18:36 UTC; no concurrent train/eval PIDs at 18:49 UTC, STOP flag retained. Exactly 12 canonical/extra MP4s and matching one-episode reports copied locally and verified by `s22_failure_videos/video_decode_verification.txt`; first three short MP4s are 40.02 s, remaining eight plus extra are 132.98 s. An unconditional 2000-step evaluator cap caused an initial `slope_down` no-velocity-samples failure; its failed 40 s clip was archived, then test-first corrected **only video-mode** cap. Original CLI backed up with SHA-256 `491147bc0ca59191ae6adad3168ae1c8e886e2a210ce2ea8cd628603f82f6f3e`; **169 passed, 70 subtests passed**. Every launch used a one-entry plan, checkpoint hash check, stop flag, no-other-sim check and single-simulator `flock`. Complete delivery: `go1_s22_partial_model550_all_videos_and_analysis.zip` (34 entries, 12 MP4, CRC verified).
- [x] **Step 3:** Fully decoded all 12 MP4s; verified report SHA/protocol/one-episode result/duration and made `s22_failure_videos/video_inventory.json`. Inspected combined first-fall sequence and full-video contact sheets plus representative gap/stair frames; cross-checked 15 raw `FALL_EVENT`s with seed-2026 exact geometry in `combined_exact_blocks.json`. `s22_failure_videos/analysis_zh.md` gives all 11 links, checkpoint/index and supported causes. Reports' default-zero `mean_failure_x` was not used as a location. Exact per-success termination frames/steps are unavailable from current report and not invented.
- [ ] **Step 4 (optional deeper diagnosis, not part of video delivery):** Instead of two other instances, recorded the same combined instance for a full 133 s with `GO1_DEBUG_FALL=1 GO1_DEBUG_GAP=1`: 15 exact body-fall events, 11 at short-gap blocks, three at first tyres, one at the tyre→slope seam. This locates where and how the trunk fails, but the defective `foot_spread = min - max` classifier means `GAP_FALL=0` cannot exclude a foot entering a void. Any foot-contact/root-cause telemetry correction must be a separate test-first, user-approved change to runtime, followed by another isolated replay.

### Task 3: Evidence review before behavior change

**Files:** `reports/diagnostic_s22_combined_observation.md` (new observation note only).

- [x] **Step 1:** All delivered dog videos use `s22/model_550.pt`, not the distinct-route `d3_combined_flat` old policy (16/16 in its own obsolete setup) or a terrain-only showcase. The latest comparable deterministic 11-family gate remains s21 `hold`: 494/883, combined 0/346; no s22 gate exists.
- [x] **Step 2:** `analysis_zh.md` maps all 15 complete mixed-video fall events to actual plan geometry: 11 low-body events on short-gap/landing bands at course_x1.87–3.29, 3 attitude failures on first tyres, 1 at tyre→slope seam; no reach to long gaps (x12.778+) or final narrow stairs (x18.351+). The gap-fall classifier's `min(foot_z)-max(foot_z)>positive_threshold` branch is impossible, so zero `fell_into_gap` does not prove no feet entered gaps. Independently documented the LP-ACRL jump wiring deficit; the early ~0.082 m gaps are non-jump obstacles, so that deficit is not proven to cause these early events.
- [x] **Step 3:** The delivered `analysis_zh.md` prioritizes test-first termination/foot-contact telemetry, then early short-gap trunk support and first-tyre attitude diagnosis, then independent verification of width-based long-gap jump; no training, reward, route, checkpoint, or jump policy was hot-modified. The watchdog STOP flag remains; no s22 gate or ≥90% certification is claimed.
