# Go1 v7 reward isolated execution (2026-10-14; in progress)

Approved scope: v7a eliminates paid false `leap_clear` on the LP-ACRL reward config; v7b retains v7a and adds bounded `supported_short`, `safe_segment`, `tire_stall` targeting generated per-environment combined geometry. Existing combined layout, 11 families, 0.35–0.8 m/s command range, 512 environments, policy/critic observations and original v5/v6 reward TOMLs remain unchanged. Original s22 model_550 is the shared parent; both runs reset optimizers via `--initialize-from`.

## Verifications before training

- Original remote files backed up before replacement: `reports/v7_backup_20261014/`; current and original SHA-256 in `reports/v7_deploy_manifest.json` on `/config/Desktop/sim`. Local isolated source: `E:/tmp-issac/v7_work/`.
- Red/green CPU tests: initially caught missing v7a config and v7b helpers; reviewer then caught unchanged-coordinate safe bonus and velocity-only stall counter reset; two additional red tests caught each. Revised high-water test caught repeated oscillation after visiting a farther tyre position. After fixes, staged remote unittests: **11 passed**. Full remote `./sim.sh test -q`: **181 passed, 76 subtests passed in 16.97s** (`reports/v7_cpu_tests_highwater.log`).
- Serial one-environment GPU smoke after final review: `V7_SMOKE_OK a samples 4 supported_positive 0 safe_paid 0`; `V7_SMOKE_OK b samples 4 supported_positive 2 safe_paid 0`; `V7_SMOKE_OK b_long samples 40 supported_positive 0 safe_paid 0`. Saved remote logs `reports/v7{a,b,b_long}_smoke_highwater.log`. Smoke is not a deterministic 512-environment gate.
- Reviewer final read-only verdict: `APPROVE` after checking measured post-crossing distance, tire high-water and per-env reset. Reviewer did not run tests.
- Geometry extracted from actual meshes: at difficulty 0 short gaps start/end 1.30/1.43, 2.08/2.21, 2.86/2.99; first tire 3.64–4.88. At difficulty 0.6 first tire may be 7.70–8.94 because the middle-block order is randomized; bounds therefore come from each tile's real `course_profile.blocks` rather than fixed tire x.

## Running

- Remote serial supervisor PID `93778`, launch `2026-10-14T08:23:20Z`, log `reports/v7_serial_driver.log`; lock `reports/v7_sim.lock`. It launches **v7a train→deterministic gate→v7b train→deterministic gate**, serially; an infrastructure failure, absent report, or plan mismatch stops before v7b. `--max-stages 1` forbids automatic next-phase escalation.
- V7a plan SHA-256 **d4a4c2fd3b96ebe5bc554a5d0088405c39352d1701ddffa5c6419d8901607e3e**, byte-identical to original v6 stage-one plan. At last observation, v7a was in warmup (`Learning iteration 65/150`), `Episode_Reward/leap_clear: 0.0000`, no reward-shape errors or traceback. No gate result exists yet; no elevation may be claimed.

## Interruption and current status (2026-10-14, after Docker GPU reattachment)

Host VFIO GPUs `65:00.0` and `ca:00.0` were released from a shut-down VM and are now NVIDIA-bound; the restarted container sees four A10s. The container PID 1 started `09:26:10 UTC`; v7a training log records `torch.distributed.elastic` sending both worker processes SIGTERM at `09:25:27 UTC`. Isaac Sim emitted a shutdown-time `Fatal Python error: Segmentation fault`. This is an interrupted simulation, not a reward-test failure and not a finished stage. Last printed learning iteration was **876/900**; last saved checkpoint `checkpoints/stage1/2026-10-14_go1-v7a-isolated-s01-p1-d1/model_850.pt` SHA-256 `36da7b172b716a0273a355f00a898e0874c422d6ccdf0c5893374e041dcf1af0`, with v7a reward-config `leap_clear.weight=0.0` and optimizer data present. No v7a deterministic gate report, v7a state, v7b state, or v7b training exists. Original deployment hashes, old states and s22 checkpoint still verify; stop flag remains present and no simulator or curriculum driver is active. Preserve this partial run; do not claim hold/advance or silently restart the optimizer.

## Isolated restart after interruption (running)

Following the container's four-A10 reattachment, the old v7a partial run and all original files were preserved. Remote post-restart staged unit tests: 11 passed; `./sim.sh test -q`: **181 passed, 76 subtests passed** in `reports/v7_cpu_tests_after_four.log`. Three serial one-environment GPU smoke tests (v7a combined, v7b combined and v7b long-gap) passed; see `reports/v7{a,b,b_long}_smoke_after_four.log`. A first restart preflight refused to overwrite the interrupted run's `lp_reward_v7a_results.txt` (SHA-256 `f0a5a6612622a1845ee15f47df4efcf9daa7c0c2f50aae54a9d6cc6485582204`); **no simulation launched** on that attempt. Created separate `configs/curriculum_phases_v7a_retry1.json` with distinct `lp_reward_v7a_retry1_{state,acrl_state,results}` paths, leaving the old partial log, checkpoint, state/result files and v6 control untouched.

Remote serial retry supervisor PID `6427` started at `2026-10-14T09:42:11Z`, `reports/v7_serial_retry1b.log`; v7a run ID `2026-10-14_go1-v7a-retry1-s01-p1-d1`. Its plan is byte-identical to v6 (`d4a4c2fd3b96ebe5bc554a5d0088405c39352d1701ddffa5c6419d8901607e3e`), and uses fresh optimizer initialized from s22 model_550. The launcher is deliberately still **two training ranks** even though the container can see four A10s; changing distributed topology in the middle of the controlled comparison is excluded. The runner will run v7a training/gate and then v7b training/gate serially, stopping after one stage per variant. No retry gate result has been observed yet.

## Official gates and final state (2026-10-14 12:09 UTC)

The serial driver finished v7a followed by v7b and stopped after one stage each. Both gates are **hold**; the log's `DONE` marker means only that both runs terminated and their reports were verified. Local copies of reports, states, plan, serial/driver logs and manifest: `E:/tmp-issac/v7_work/results/`. Independent verification summed the **raw `environments` records** by the byte-identical 512-instance plan SHA-256 `d4a4c2fd3b96ebe5bc554a5d0088405c39352d1701ddffa5c6419d8901607e3e`, rather than reconstructing numerators from rounded family rates.

| Family | v7a success/episodes | v7b success/episodes |
|---|---:|---:|
| combined | 0/390 | 0/398 |
| gaps_long | 46/73 (63.01%) | 44/65 (67.69%) |
| gaps_short | 47/52 | 47/49 |
| slope_down, slope_mixed, slope_up | 47/47 each | 47/47 each |
| stairs_down_long, stairs_mixed, stairs_narrow_8_mixed | 46/46 each | 46/46 each |
| stairs_up_long | 45/46 | 45/46 |
| tire_field | 42/46 | 45/46 |
| **Aggregate** | **459/886 = 51.81%, 9/11 families ≥90%, hold** | **460/883 = 52.10%, 9/11 families ≥90%, hold** |

- v7a exact report: `reports/evaluate_stage1_2026-10-14_10-53-47_gate-2026-10-14_go1-v7a-retry1-s01-p1-d1.json`; model `checkpoints/stage1/2026-10-14_go1-v7a-retry1-s01-p1-d1/model_899.pt` SHA-256 `3495698e0a63adeb886ff138eeb81b209676f86727f1f96966af43206925c31b`.
- v7b exact report: `reports/evaluate_stage1_2026-10-14_12-09-24_gate-2026-10-14_go1-v7b-isolated-s01-p1-d1.json`; model `checkpoints/stage1/2026-10-14_go1-v7b-isolated-s01-p1-d1/model_899.pt` SHA-256 `fd04f3a6f1c5d03c6d6992be722a6924ab2ca9015112a2bdc6aa8242c2d4e631`.
- Each exact `stage1-v2` report has `passed=false`, seed 2026, a checkpoint SHA matching the model, and `episodes=512` denoting planned environments; individual environment episode counts sum to 886/883. Both state files record target level 1 and `hold`.
- In last 100 logged training iterations v7a `Episode_Reward/leap_clear=0.000000`; v7b `supported_short=+0.001554`, `safe_segment=+0.000389`, `tire_stall=−0.017021`, `leap_clear=0.000000` as global means. The new positive signals exist, but cannot be inferred to repair combined from these averages; no v7b failure-location video has been recorded.
- Fresh integrity check printed `INTEGRITY_OK 10 deployed_files 3 preserved_hashes 2 states_hold stage_target_1 stop_flag_present` and `SIM_PROCESS_COUNT=0`. No higher stage or additional training was started. The interrupted original v7a run and original v5/v6/s22 artifacts remain preserved.

**Conclusion:** Removing the false flight bonus and adding v7b short-gap/tyre shaping did not satisfy the simultaneous 11-family ≥90% condition; `combined` remains 0 and `gaps_long` remains below threshold. The two variants used the same sampled plan and source s22 initialization, but reward effects must not be inferred uniquely from small differences in repeated-episode fractions. Further work requires new failure-position/contact evidence and a separate decision, not blind reward amplification or automatic stage advancement.
