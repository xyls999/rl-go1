# V1330 gait/gap-aware reward candidate

**Status: one authorized training run completed; `trained-unverified`.** The candidate is isolated from V1329 and V1325–V1328. The actor interface remains 1073 observations / 12 actions; observation profile and task plan are byte-identical to V1329.

## Reward changes

- Raise the one-shot `fallen` cost from `-12` to `-16`.
- Reduce graded/entry `gap_intrusion` shaping to `weight=-8`, `entry_multiplier=1.5`; retain the intrusion-depth signal.
- Enable the existing duration-growing stall and backward-motion costs (`stall_grow=-1`, cap `2`; `backward_motion=-1`).
- Add a soft fore/aft swing envelope (`0.32 m`) and mild sustained four-foot-airborne penalty away from gap windows (`0.06 s` grace, `0.12 s` ramp). Both are masked in the simulator-derived gap approach/crossing/landing window.
- Add a progress-gated leading-foot lift reward only for short gaps, using a fixed `0.22 m` target. Horizontal gap width selects the crossing style; it does not scale vertical foot height. The existing scorer still permits supported step-over for short gaps and requires flight plus stable far-side landing for long gaps.

These are unvalidated shaping values, not claimed performance gains. V1329's env203 diagnostic ended in a severe tilt without an explicit gap-fall flag; this candidate is a hypothesis, not proof of the cause.

## Authorized execution scope

- Exactly one **random-initialized** training run, **512 environments × 650 iterations**, with a fresh seed audited on the remote workspace.
- Training only. **Diagnostic evaluation, holdout, formal gate, retry, and follow-on runs are not authorized.** The runner revokes active execution permissions after training completion or a launched-run terminal failure.
- Remote execution uses password-only SSH and the existing exclusive simulator lock; source overlays are hash-pinned and restored before a successful receipt is accepted.

## Training-only result

The fresh-seed random-init run (`go1-v1330-gait-gap-aware-s01-p1-d1`, seed `1427789363`) completed 650 iterations on 512 environments and recorded 118,106 training episodes. Its training-stage summary reports completion `0.7925%`, survival `1.3869%`, mean achieved speed `0.6646 m/s` versus mean commanded speed `0.8604 m/s`. These are training metrics only; they do **not** establish held-out or generalization performance. No diagnostic evaluation, holdout, formal gate, retry, or follow-on run was executed. The run authorization was revoked and the remote source overlay restored.

## Files and evidence

- Reward profile: `configs/rewards_stage1_gaps_v1330_gait_gap.toml`
- Isolated source overlay: `source/go1_hier_sim/go1_hier_sim/`
- Training-only fail-closed runner: `scripts/verify_and_run_v1330.sh` and `scripts/production_v1330.py`
- One-shot authorization and run contract: `configs/manifest.json`, `configs/train_execution.json`, `reports/authorized_run_manifest.json`
- Unit/config/runner tests: `tests/`
- Local implementation checks: `reports/final_local_validation.txt`
- Training metrics and artifact hashes: `reports/training_summary.md`
- Remote preflight/run/reconciliation record: `reports/remote_training_evidence.txt`

No additional execution is authorized by this run contract. Any evaluation, holdout, gate, retry, or follow-on requires separate authorization.
