# Go1 v11 Unified 11-Family LP-ACRL Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task-by-task. This workspace is not Git; replace commits with versioned backups and SHA manifests. User directed implementation without repeated approval prompts.

**Goal:** Train one Stage-1 actor across every one of the fixed 11 LP-ACRL families, including combined, without a leap specialist.

**Architecture:** Extend reward-only mixed tyre high-water progress and the full-roster stage planner's explicit per-family quota; preserve actor schema and geometry. Warm-start one unfrozen actor from the unvalidated v8 walker. Never launch two Isaac suites or automatic p2.

**Tech Stack:** Python3.11, PyTorch2.7, Isaac Lab2.3.2, pytest; remote `/config/Desktop/sim`.

**Spec:** `docs/superpowers/specs/2026-10-16-go1-v11-unified-roster11-design.md`.

## Global Constraints

- Exactly the roster `TERRAIN_FAMILIES` (11), commands0.35–0.8m/s, 1073 actor observations/12 actions/GRU128; no privileged actor, gate or deployment input.
- No `GO1_V10_OBSERVABLE_LEAP`, jump checkpoint or frozen walker handoff in v11. Preserve old data and `lp_phased_stop.flag`; no watchdog/automatic p2.
- Training reward may use privileged truth; net-force contact ambiguity remains unresolved, so no safety or ability claim from PPO. Only same-session512 deterministic evaluation 11/11 each≥90% can promote.

---

### Task 1: Mixed tyre progress reward (RED→GREEN)

**Files:** staged copies of `source/go1_hier_sim/go1_hier_sim/envs/mixed_transition_reward_v7.py`, `envs/runtime.py`, `envs/config.py`, `reward_config.py`; new `configs/rewards_stage1_gaps_v11_unified.toml`; tests in `v11_work/tests/test_v11_mixed_tire.py` staged to remote tests.

- [ ] RED test one environment: first tyre contact initializes high water at x4.0; moving to4.01 with upright root≥.27, >=1 contacting foot and forward speed>0.1 returns positive bounded `tire_progress`; repeat/retreat/return produce zero; new progress4.03 pays; fallen/collision/airborne/noncombined/outside band pays zero; reset clears. Run focused remote pytest and see intended failure.
- [ ] GREEN minimal high-water reward: compute delta against old per-env `tire_highwater` *before* its existing update, require first tyre visit and stable supported posture, pay `(delta.clamp(0,.02)/.02)` only inside tyre band. Keep `safe_segment` one-off and `tire_stall` unchanged. Add optional `tire_progress` reward schema/positive sign, config `RewardTermCfg` and weight routing in all contract branches, runtime term and v11 TOML `weight=1.0`; `reward_config` must leave old tables parseable. Run focused + full unit suite.

### Task 2: Full-roster per-family quota and phase plan (RED→GREEN)

**Files:** staged `scripts/run_phased_curriculum.py`; new `configs/curriculum_phases_v11_unified.json` from v8 with distinct state/result/base id, p1 speed_levels `[0,1,2,3,4]`, family priorities `combined:2.5,gaps_long:2.0`, others1; new `v11_work/tests/test_v11_roster_plan.py`.

- [ ] RED tests: 512-slot training plan with weight2.5/2.0/1.0 gives all11 named families, exact512 slots, combined>gaps_long>each other, no family absent, preserves each original slot's difficulty/speed, deterministic rounding; zero/unknown priorities reject. Independent gate plan must preserve speed/difficulty yet balance11 families46–47 slots each (old `min_family_episodes=40` stays). Test focused remote import of driver.
- [ ] GREEN: extend `rebalance_family_plan` with optional explicit family priority using largest-remainder integer quotas; default (no priorities) remains exactly old equal roster behavior. `main()` supplies priorities from v11 JSON for training but writes a separate equal-roster512 gate plan for deterministic evaluation and `gate_decision`; source task difficulty/speed assignments stay intact and both plan fields and instances agree. Validate both generated512 plans and5 speed levels; run full unit suite. No reduction of actual tile count or mixed geometry.

### Task 3: Versioned deployment and serial unified p1

**Files:** remote `reports/v11_backup_<UTC>/`, `reports/v11_unified_manifest.json`, `sim/agent.md`, local `v11_work/reports/`.

- [ ] Compare staged sources to remote originals, backup only changed remote files, hash all sources/rewards/config/old checkpoint/stop flag. Copy changed files under lock; run `./sim.sh test`, 16-env×2 PPO smoke with **all11** (plan16 includes each family), confirm no leap checkpoint/opt-in and all actor updates. Fail closed on nonfinite/physics/PPO errors.
- [ ] Create independent v11 p1 Stage-1 run of 900 iterations using one actor warmstarted from v8 `model_899.pt` SHA `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`; two GPU ranks×512 each with existing backend proven on this machine, no concurrent Isaac, no watchdog/p2, stop flag retained. Verify actual PPO iterations and family assignment; preserve logs/checkpoint metadata and status.
- [ ] After training exits and only then run original deterministic 512-env/11-family gate once; report per-family numerator/denominator, long-gap flight/landing and mixed first failures. Only 11/11 each≥90% promotes; otherwise `hold` and do not call it solved.
