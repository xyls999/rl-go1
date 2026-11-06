# V1330 training-only result

**Status:** `trained-unverified`. One authorized training run completed; no evaluation or gate was run.

## Run contract and outcome

- Run: `go1-v1330-gait-gap-aware-s01-p1-d1`
- Initialization: random; audited fresh seed `1427789363`
- Training: 512 environments × 650 iterations; 118,106 episodes
- Training-stage completion rate: `0.0079250842` (0.7925%)
- Training-stage survival rate: `0.0138688974` (1.3869%)
- Mean achieved speed: `0.6646438 m/s`; mean commanded speed: `0.8604084 m/s`
- No diagnostic evaluation, holdout, formal gate, retry, or follow-on run occurred. The checkpoint is **not** independently validated and these training aggregates do not establish generalization or improved gap-crossing capability.

## Safety and restoration checks

The post-run receipt records `status: trained-unverified`, `restored: true`, `diagnostic_ran: false`, `formal_gate: false`, `final_gate_passed: false`, and `final_holdout: false`. A separate read-only check under the exclusive remote lock confirmed the 512-environment/650-iteration report, checkpoint/log/report hashes, all 14 live-source baselines restored (including removal of the V1330-only helper), no leftover overlay backup, no training process, and idle GPUs (39/4/4/4 MiB reported). Active run/training/diagnostic/gate/retry/follow-on authorizations are all false.

## Local artifacts and SHA-256

| Artifact | SHA-256 |
|---|---|
| Checkpoint `checkpoints/stage1/go1-v1330-gait-gap-aware-s01-p1-d1/model_649.pt` | `d72baffb9c641847acccb0f871e2f9eae92b27114d844d629d8a043140c15a51` |
| Training log `reports/go1-v1330-gait-gap-aware-s01-p1-d1_full_train.log` | `726e6c4e4ad3363a846ecf1ab7c46718582b153e6d96fbf3990c5eee52d3b0b1` |
| Stage report `reports/stage_go1-v1330-gait-gap-aware-s01-p1-d1.json` | `48957ca1484487d87b345bf3352bea6f8b96af9aee191437198296253131282e` |
| Training receipt `reports/go1-v1330-gait-gap-aware-s01-p1-d1_full_train.receipt` | `850569ab16db8e66d06d786b71f597d62eb17219bbd63cd2f79b02031ffa3448` |
| Fresh-seed audit `reports/training_seed_audit.txt` | `c988998103fc672722ed2c33d686ab8d0b1c0f4933602ee287f4486ce6ff3835` |
| Archived one-shot authorization manifest | `d89ba4496a9921aad057bde2518d2ec21e2efa93db2c3cd3902fe21178772ad4` |
| Current revoked manifest `configs/manifest.json` | `da1217d081bfa8ba84c11f4451956f004fe8aad407bf363d1fe1fce1da431668` |

The curated remote execution and post-run reconciliation record is in `reports/remote_training_evidence.txt`.
