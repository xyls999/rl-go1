# Stage-1 reward contract (V13: soft gait balance)

`configs/rewards_stage1.toml` is the source of truth. Isaac Lab integrates each weighted term over the 0.02 s control step, and the parsed contract is stored in every Stage-1 checkpoint.

## Why V12 was removed

V12 required every foot to produce at least two touchdowns in each fixed two-second window, limited the touchdown-count spread, and constrained each foot's contact fraction. This was a brittle heuristic: valid stair climbing often uses deliberately asymmetric support, contact sensors can chatter at a riser, and counts say nothing about diagonal gait phase. V12 also changed the scale of several event terms by dividing them by the control step without retuning their weights. Warm-start training therefore changed the objective far too abruptly and destroyed the useful V5 gait.

## V13 design

V13 returns to the reward scale used by the best V5 checkpoint and adds only bounded, soft gait regularizers:

* `trot_gait` ports Isaac Lab Spot's continuous gait timing idea. FL+RR and FR+RL air/contact clocks are synchronized within each diagonal pair and anti-synchronized across pairs. It replaces the old binary contact bonus.
* `air_time_variance` follows Isaac Lab Spot: variance of the four completed air intervals plus variance of the four completed stance intervals, both clipped at 0.5 s. This discourages one leg from having a systematically different duty cycle without demanding an exact touchdown count.
* `foot_contact_cycle` is a small, bounded continuous cost after 0.45 s of uninterrupted swing. It catches a leg that never lands, which completed-interval variance alone cannot see.
* `flat_trot_pattern` follows the FWWB guard: while moving, penalize an exactly-two-foot support state only when the pair is non-diagonal.
* A soft height-scan `terrain_gate` relaxes trot pattern, timing, lateral-foot, and overdue-air constraints around rough/stair edges. Stair footholds are therefore not forced to look like flat-ground trot.
* Swing clearance is measured relative to the highest currently supporting foot, not the same foot's previous touchdown height. This prevents stair elevation from masquerading as clearance.
* `foot_clearance` is reduced from 0.75 to 0.10, `small_steps` from 0.75 to 0.10, and explicit `step_frequency` shaping is disabled. These terms must not dominate task completion or create a profitable high-leg/jitter gait.

## Active gait-related weights

| Term | Weight | Purpose |
|---|---:|---|
| `trot_gait` | 0.50 | smooth diagonal timing score |
| `foot_clearance` | 0.10 | bounded touchdown clearance quality |
| `small_steps` | 0.10 | weak step-length preference |
| `step_frequency` | 0.00 | disabled; no frequency farming |
| `four_feet_grounded` | -0.25 | discourage prolonged four-foot shuffling |
| `foot_contact_cycle` | -0.50 | bounded cost for a foot held airborne |
| `air_time_variance` | -0.60 | equalize completed air/stance duty cycles |
| `flat_trot_pattern` | -0.20 | reject non-diagonal two-foot support on flat terrain |
| `foot_lateral` | -1.00 | prevent crossing/excessive lateral spread |

Task tracking and completion remain the dominant positive signal. Energy, action-rate, slip, body-motion, collision, fall, and boundary terms retain the best V5 scale so the new gait terms do not redefine the optimization problem.

## References

* NVIDIA Isaac Lab Spot `GaitReward` and `air_time_variance_penalty`.
* NVIDIA Isaac Lab Unitree Go1 rough locomotion configuration.
* Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, CoRL 2021/2022.
* Margolis and Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*, CoRL 2022.
* The local `E:\fwwb` terrain-aware clearance and flat-trot-pattern rewards.

Change one reward family at a time and inspect per-term magnitudes before a long run. A changed reward contract must use policy-only warm start; never resume optimizer state from a different contract.

The old V5 actor is unusually sensitive to parameter updates: controlled smoke
tests showed that a relative actor-weight change near `1e-4` can sharply reduce
completion. Warm-start mode therefore uses a fixed optimizer schedule,
separate actor/critic learning rates (`3e-8` and `3e-6`), and supports
`--critic-warmup-iterations`. These controls are intended for diagnostics and
very gradual adaptation; a clean V13 run is preferred when optimizing the new
gait objective to convergence.
