# Gap / discontinuous-terrain locomotion: how the literature trains for it

Research note for the Go1 Stage-1 course (Isaac Lab 2.3.2 / Isaac Sim 5.1), where the open question is
whether to keep **two separately trained policies (walker + jump specialist) with a hand-written runtime
trigger**, or move to the published recipes. Every claim below is linked to a primary source
(arXiv abs / PMLR / official repo / official doc). Where a number or claim could not be verified from a
primary source, it is marked **unverified**.

Compiled 2026-09-18. "Robot length" numbers are taken exactly as stated by each paper (usually trunk/body
length, not leg span); do not mix them.

## Comparison table

| Work | Year | Robot | Terrain | Single policy vs specialists | Routing? | Training stages | Key finding |
|---|---|---|---|---|---|---|---|
| [Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*](https://arxiv.org/abs/2010.11251) | 2020 | ANYmal | rough (hills, steps, stairs), blind | **single generalist** | none | teacher (privileged) → student (DAgger) | Explicitly rejects "elaborate state machines that trigger motion primitives"; adaptive terrain curriculum via particle filter targeting traversability 0.5–0.9 |
| [Miki et al., *Learning robust perceptive locomotion…in the wild*](https://arxiv.org/abs/2201.08117) ([Science Robotics](https://www.science.org/doi/10.1126/scirobotics.abk2822)) | 2022 | ANYmal | wild incl. stairs | **single student** | none | teacher RL (privileged) → student distill (belief encoder) | "does not require any special mode for stairs"; degrades gracefully to proprioception when exteroception fails |
| [Margolis et al., *Learning to Jump from Pixels* (DIC)](https://arxiv.org/abs/2110.15344) | 2021 | Mini Cheetah (28 cm tall, 38 cm long) | "gap world", variable-width gaps | hierarchical: RL high level + model-based low level | none at policy level | RL high-level from depth, sim→real | Crosses **66 cm in sim**, **16 cm real with onboard depth**, **26 cm real with mocap/ground truth** |
| [Rudin et al., *Advanced Skills by Learning Locomotion and Local Navigation End-to-End*](https://arxiv.org/abs/2209.12827) | 2022 | ANYmal | gaps, boxes, obstacles | **single end-to-end** | none | PPO end-to-end, position+time reward | Velocity tracking limits the solution space; goal+time reward lets one policy learn jumping/climbing and a new gait |
| [Agarwal et al., *Legged Locomotion in Challenging Terrains using Egocentric Vision*](https://arxiv.org/abs/2211.07638) | 2023 | medium-sized quadruped (exact model **unverified**) | stairs, curbs, stepping stones, gaps | **single student** | none | scandots RL (phase 1) → depth DAgger distillation (phase 2) | Single phase-1 policy covers all terrain; **100% success on gaps ≤ 26 cm**, 94% stepping stones; blind baseline 0% on gaps |
| [Hoeller et al., *ANYmal Parkour*](https://arxiv.org/abs/2306.14874) ([Science Robotics](https://www.science.org/doi/10.1126/scirobotics.adi7566)) | 2023 | ANYmal | parkour: walk, jump gaps, climb, crouch | **5 specialists + learned high-level router** | **learned** router at 5 Hz selects among **frozen** 50 Hz skills | per-skill RL, then hierarchical RL for the router | The router is *trained*, capability-aware, and the only published design with explicit skill selection; up to 2 m/s |
| [Zhuang et al., *Robot Parkour Learning*](https://arxiv.org/abs/2309.05665) ([PMLR v229 p73](https://proceedings.mlr.press/v229/zhuang23a.html), [site](https://robot-parkour.github.io/)) | 2023 | A1 and **Go1** | climb, leap gaps, crawl, squeeze, run | 5 skills → **distilled into ONE vision policy** | none at runtime | 2-stage RL (soft→hard dynamics constraints) → **DAgger distill into single policy** | Single end-to-end vision policy; **leap gaps 0.60 m (1.5× robot length) at 80%**; Go1 supported by same pipeline |
| [Caluwaerts et al., *Barkour*](https://arxiv.org/abs/2305.14654) ([blog](https://research.google/blog/barkour-benchmarking-animal-level-agility-with-quadruped-robots/), [repo](https://github.com/google-deepmind/barkour_robot)) | 2023 | custom quadruped | 0.5 m broad-jump board (> body length), A-frame, weave | **two baselines: specialists+switch vs distilled Transformer** | explicit switching baseline | specialist RL + high-level nav; and distillation to "Locomotion-Transformer" | Switching and generalist score similarly, **generalist has smoother transitions**; jump specialist only **38%** on the 0.5 m broad jump |
| [Cheng et al., *Extreme Parkour with Legged Robots*](https://arxiv.org/abs/2309.14341) ([site](https://extreme-parkour.github.io/), [repo](https://github.com/chengxuxin/extreme-parkour)) | 2024 | A1 | extreme parkour (gaps, tilted ramps, handstand) | **single neural net policy** | none | scandots RL (phase 1) → distillation to depth + heading (phase 2) | Single policy; long jump across gaps **2× robot length**, high jump 2× height |
| [PIE, *Parkour with Implicit-Explicit Learning*](https://arxiv.org/abs/2408.13740) | 2024 | Lite3 | gaps, stairs, steps | **single policy** | none | **one-stage** end-to-end RL | Gaps up to **1 m (3× robot length)**; curriculum gaps up to 1 m |
| [*Agile, Continuous, Terrain-Adaptive Jumping*](https://arxiv.org/abs/2409.10923) | 2024 | **Go1** | human-sized stairs, sparse stepping stones | hierarchical single motion policy + model low-level | none | heightmap predictor + RL motion policy + model leg controller + terrain curriculum | Go1 single discontinuity: **0.8 m horizontal / 0.6 m vertical**; continuous jump of a 14-step staircase in 4.5 s |
| [*Curriculum-Based RL for Quadrupedal Jumping*](https://arxiv.org/abs/2401.16337) | 2024 | **Go1** | obstacles (learned without reference trajectories) | **single policy** | none | curriculum: in-place → forward/diagonal → over obstacles | **90 cm forward jump**, single policy, no reference trajectory |
| [SoloParkour](https://arxiv.org/abs/2409.13678) | 2024 | Solo-12 | leap/climb/crawl | **single policy** | none | privileged RL → off-policy pixel RL (warm-start from privileged) | 35 cm gap (78% of length); DAgger/BC baselines fail on leap terrains |
| [WMP, *World Model-based Perception*](https://arxiv.org/abs/2409.16784) | 2024 | A1 | stairs, gap, climb, tilt, crawl | single student (world-model) | none | world model + policy; same curriculum as Rudin 2022 | A1 gap **85 cm (~2.1× robot length)**; documents the student-vs-teacher performance gap of plain distillation |
| [HOVER](https://arxiv.org/abs/2410.21229) | 2024 | humanoid (19-DOF) | n/a (control *modes*, not terrain) | **single masked multi-mode generalist** | **training-time command mask** (random per episode), no external switch | oracle motion imitation → DAgger distillation | Single masked policy **beats per-mode specialists**; separate per-mode policies are "not compatible with each other"; "seamless transitions between control modes" |
| [Acrobotics](https://arxiv.org/abs/2509.02727) | 2025 | quadruped | generalist parkour | **single generalist** | none | generalist RL | Rivals mixture-of-experts specialists with **25% as many agents** |
| [JumpER](https://arxiv.org/abs/2507.01243) | 2025 | monopedal quadruped | wide gaps, stepping stones, irregular stairs | **single policy** | none | staged curriculum; **objective switched velocity-tracking → goal-reaching** | Velocity-tracking RL converges to "spin in place" local optimum; goal-reaching objective fixes it; gaps ≤ 60 cm |
| [*Now You See That* (RSS 2022)](https://www.roboticsproceedings.org/rss22/p027.pdf) | 2022 | humanoid | stairs, platforms, wide gaps | single (multi-critic) | none | privileged RL → DAgger distill | Notes training separate specialists before distillation "incur[s] substantial training overhead"; wide gaps > 45 cm real |

## 1. State of the art for discontinuous terrain: which architecture?

**Dominant answer: a single end-to-end policy (often via a teacher→student distillation), not a
runtime-routed set of specialists.** The only prominent published *routed* system is ANYmal Parkour
(Hoeller 2023), and its router is itself a **learned high-level policy**, not a hand-written trigger.

- Every parkour/gap system listed in the table lands on one policy *at deployment*: Zhuang 2023 explicitly
  states "we distill these skills into a single vision-based parkour policy"; Cheng 2024 trains "a single
  neural net policy"; Agarwal 2023 trains one phase-1 policy for all terrain; PIE 2024 is "one-stage
  end-to-end"; SoloParkour trains "a single policy … on a curriculum of variations of these terrains";
  Curriculum-based Jumping 2024 and Agile Continuous Jumping 2024 both give Go1 a single motion policy.
- The specialists-then-distill pattern is common, but the **distillation target is one policy**:
  Zhuang 2023 (5 skills → DAgger → 1 policy), Barkour 2023 (specialists → Transformer generalist),
  HOVER 2024 (per-mode → masked generalist). HOVER explicitly reports the generalist "not only supports
  diverse control modes but also outperforms policies trained individually for each mode."
- The only explicitly *routed* design, ANYmal Parkour (Hoeller et al. 2023), trains a catalog of skills
  and then "using a high-level policy to select and control those skills across the terrain", with the
  low-level skills **frozen** during router training and the router operating at 5 Hz over 50 Hz skills.
  Crucially the router sees a learned perception latent and is *aware of skill capabilities*, i.e. it is
  trained to route, not hand-coded.

## 2. Robustness of runtime switching vs a single policy

The literature consistently reports that **hand/runtime switching between independently trained policies is
the fragile part**, and single (distilled/masked) policies transition more smoothly:

- Lee 2020 frames the whole problem against classical controllers "based on elaborate state machines that
  explicitly trigger the execution of motion primitives and reflexes," which "have increased in complexity
  but fallen short of the generality and robustness of animal locomotion." Their alternative is one
  generalist student.
- Miki 2022: "our controller does not require any special mode for stairs, and can traverse stairs natively
  in any direction and any orientation." Mode-free single policy is presented as the robustness win; when
  exteroception degrades it "smoothly transitions to proprioceptive locomotion" *within the same policy*.
- HOVER 2024 states that separately developed control modes "are not compatible with each other," and its
  contribution is a single masked policy that "enables seamless transitions between control modes."
- Barkour 2023 is the most direct evidence for your exact question. The Google DeepMind write-up says:
  "the single generalist locomotion transformer policy and the specialist policies with the policy
  switching mechanism achieve similar performance. The locomotion transformer policy has a slightly lower
  average Barkour score, but **exhibits smoother transitions between behaviors and gaits**."
- Rudin 2022 (Advanced Skills) argues that decomposing navigation into "path planning, path following and
  locomotion … limit the robot's capabilities since the individual tasks do not consider the full solution
  space," and shows one end-to-end policy learns jumps/climbs the decomposed version does not.
- *Now You See That* (RSS 2022) notes specialists-before-distillation "incur[s] substantial training
  overhead" — relevant because your two-model design also pays a separate-specialist cost.
- *Control-rate / OOD hand-over latency*: no paper we found quantifies trigger latency for a hand-coded
  policy switch. Margolis 2021 does quantify the real consequence of distribution shift: DIC crosses 66 cm
  in sim but only 16 cm with onboard depth and 26 cm with ground-truth state — i.e. the hand-over boundary
  is where sim-to-real (and OOD) errors concentrate. **No primary source supports a hand-written void-lip
  trigger as a robust mechanism**; treat your 100% first-void failure as consistent with the reported
  fragility of OOD switching.

## 3. Terrain curriculum and privileged-teacher → distillation pipeline

**Terrain curriculum (legged_gym / Isaac Lab "terrain levels"):**
- Each sub-terrain is a row (difficulty) × column (terrain type) grid. From
  [`legged_gym`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py):
  `move_up = distance > terrain_length/2`; `move_down = distance < ||command||·T·0.5` and not moving up;
  `terrain_levels += move_up - move_down`, clipped at 0; robots solving the top level are sent to a
  **random** level. Config defaults: `num_rows=10`, `num_cols=20`, `max_init_terrain_level=5`,
  `terrain_length=terrain_width=8` (see the [config file](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)).
- Rudin 2022 (Minutes) describes the same curriculum: step height 5→20 cm, slope 0→25°, level up when the
  robot crosses the terrain border, level down if it moved less than half the required distance, loop back
  from the top level to preserve diversity ([arXiv 2109.11978](https://arxiv.org/abs/2109.11978)).
- Isaac Lab generalizes this: environment origins are sub-terrain origins; `update_terrain_levels()` moves
  them to harder terrain; `max_init_terrain_level` defaults to `num_rows-1` when `None`; difficulty varies
  linearly over rows with per-column random jitter ([API docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html),
  [how-to](https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html)). Built-in configs
  already expose `gap_width_range` (`MeshGapTerrainCfg`) and `stone_distance_range` (`HfSteppingStonesTerrainCfg`),
  so gap width can be the curriculum variable directly.
- Lee 2020 gives an alternative automatic curriculum: a **particle filter** over terrain parameters that
  keeps the distribution of traversability in **[0.5, 0.9]** (neither trivial nor impossible); it is applied
  during both teacher and student training ([arXiv 2010.11251](https://arxiv.org/abs/2010.11251)).
- JumpER 2025 adds an important twist: with pure velocity-tracking RL the policy can converge to a local
  optimum that "spin[s] in place to avoid falling down" and never attempts the gap; switching the objective
  to **explicit goal-reaching** fixes it ([arXiv 2507.01243](https://arxiv.org/abs/2507.01243)). This is
  exactly the "walker stops at the void" failure you observed.

**Privileged teacher → distillation (two-stage):**
- Pattern: train a teacher with cheap privileged info (scandots / height scan / obstacle geometry), then
  train a student that sees only deployable inputs, supervised by DAgger on the teacher's actions.
- Miki 2022 (SciRob): teacher RL with ground-truth terrain → student with an attention-based recurrent
  belief encoder that fuses proprioception + exteroception; deployed zero-shot.
- Agarwal 2023: phase 1 RL with **scandots** (cheap pseudo-depth); phase 2 supervised distillation to a
  depth+proprioception RNN at 50 Hz. "Since supervised learning is orders of magnitude more sample
  efficient than RL," this makes training tractable.
- Cheng 2024: phase 1 scandots RL + ROA adaptation; phase 2 distillation to depth **and** a learned heading
  predictor (with a teacher/student heading mixture to prevent drift).
- Lee 2020: teacher RL + proprioceptive student (TCN) via DAgger; adaptive curriculum runs in both phases.
- Zhuang 2023: uses the same two-phase scaffolding but the "teacher" is a set of skills distilled by DAgger
  into one vision policy.

## 4. One policy that both walks AND jumps/leaps

The literature is nearly unanimous: **keep one low-level policy and let it learn the maneuver.**

- Miki 2022 and Lee 2020 show one policy handling stairs/slopes/rough with no mode input.
- Rudin 2022 (End-to-End) shows one policy learning to jump gaps and climb boxes because the objective is
  goal+time, not velocity tracking; the reward "is not influenced by the trajectory the robot took."
- Zhuang 2023 / Cheng 2024 / Agarwal 2023 / SoloParkour / PIE: all single policies over walk + leap +
  climb + crawl, differing only in how skills are acquired (soft-dynamics RL, scandots, curriculum).
- ANYmal Parkour (Hoeller 2023) *does* separate skills, but the separation is at the **navigation level**
  (which skill to activate) and each skill is a full low-level controller with a position/heading/time
  command; the router is learned and runs at 5 Hz. This is the closest thing to your "jump specialist," and
  it is fundamentally different from a hand-written height-scan trigger.
- The "mode input / mask" idea appears mainly for **humanoid whole-body control** (HOVER 2024; MaskedMimic
  in graphics), where the modes are command spaces (root tracking vs joint-angle tracking). HOVER's mask is
  a *training-time* random mask, not a runtime switch. There is no primary-source example of a runtime
  terrain trigger selecting a walk vs jump policy that is reported as robust.
- Go1-specific single-policy jumps: Curriculum-based Jumping 2024 (90 cm forward jump, one policy) and
  Agile Continuous Jumping 2024 (Go1, 0.8 m horizontal / 0.6 m vertical, 14-step staircase) both use a
  single motion policy; the latter adds a model-based low-level tracker because the *sim-to-real* gap for
  high-speed pixel-to-action is large, not because it swaps skills.

## 5. How wide a gap can a Go1-class (~0.3 m tall) robot clear, and how to parameterize the curriculum

Published, verifiable numbers (keep the robot's own length/height definition attached):

| Robot | Gap / jump | Source |
|---|---|---|
| Mini Cheetah (28 cm tall, 38 cm long) | **66 cm in sim**; **16 cm real (onboard depth)**; **26 cm real (ground-truth state)** | [Margolis 2021](https://arxiv.org/abs/2110.15344) |
| Medium quadruped (model **unverified**) | **gaps ≤ 26 cm, 100% success**; blind 0% | [Agarwal 2023](https://arxiv.org/abs/2211.07638) |
| A1 / Go1 (low-cost) | **leap 0.60 m = 1.5× robot length, 80%** | [Zhuang 2023](https://arxiv.org/abs/2309.05665) |
| A1 | long jump across gaps **2× robot length** | [Cheng 2024](https://arxiv.org/abs/2309.14341) |
| Custom quadruped | broad jump 0.5 m board (> body length), **38% success** | [Barkour 2023](https://arxiv.org/abs/2305.14654) |
| Solo-12 | leap **35 cm = 78% of length** | [SoloParkour 2024](https://arxiv.org/abs/2409.13678) |
| A1 | gap **85 cm (~2.1× robot length)** | [WMP 2024](https://arxiv.org/abs/2409.16784) |
| Lite3 | gaps **1 m (3× robot length)** | [PIE 2024](https://arxiv.org/abs/2408.13740) |
| Go1 | single discontinuity **0.8 m horizontal / 0.6 m vertical**; continuous 14-step staircase | [Agile Continuous Jumping 2024](https://arxiv.org/abs/2409.10923) |
| Go1 | **90 cm forward jump** (jump policy, no reference) | [Curriculum-Based Jumping 2024](https://arxiv.org/abs/2401.16337) |

Interpretation for your hand-tuned **0.37–0.78 m**:
- **0.37 m** is within the demonstrated single-policy range (≈ Agarwal's 26 cm real / Barkour's 0.5 m board
  territory; well under Zhuang's 0.60 m leap). Reasonable for training.
- **0.78 m** is at or beyond the published *extreme* frontier. Zhuang 2023 reports 0.60 m (1.5× length) at
  80%; A1 2× length (Cheng 2024) and A1 85 cm (WMP) exist but are either specialized parkour policies or
  world-model/teacher-student systems, and the Go1 0.80–0.90 m results are **jump-focused policies** with a
  model-based low-level tracker (Agile Continuous Jumping) or a dedicated jump curriculum with no reference
  (Curriculum-Based Jumping). A generalist walking+jumping PPO policy reaching 0.78 m is **not demonstrated
  in the sources reviewed** — treat it as a stretch goal, not the curriculum baseline.
- Parameterize by **fraction of body length**, not absolute meters: train gap width as a curriculum from
  ~0.1 m to ~0.4 m (≈ 0.25–1.0× a Go1 trunk length), and reserve >1× length for a later difficulty column
  once the walking policy already crosses the small gaps. Use `MeshGapTerrainCfg.gap_width_range` /
  `HfSteppingStonesTerrainCfg.stone_distance_range` with the row-based curriculum so difficulty escalates
  automatically ([Isaac Lab API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html)).
- Key structural lesson across sources: **a gap is not a foothold problem, it is a planning-horizon
  problem.** Margolis 2021, Agarwal 2023 and WMP 2024 all emphasize that the policy must *remember*
  (history/recurrent state) because the front camera cannot see the terrain under the hind feet. Your
  height-scan actor already has that information; the recurrent actor is the right tool.

## Recommendation for our setup

**Remove the runtime trigger and the two-model design.** No primary source supports a hand-written void-lip
trigger that hands control to a separately trained jump policy; the reported behaviour is exactly what you
see (brittle hand-over, OOD at the first void, no transfer because the specialist saw a different gap
distribution). The literature's answer is a **single policy trained on the full course**, with skills
acquired by curriculum (Zhuang 2023, Cheng 2024, Agarwal 2023, SoloParkour 2024, Hoeller 2023's learned
router being the routed exception — and even there the router is trained, not scripted).

### What to delete from the code
1. The walker↔jump policy switch and its height-scan "void lip" detector.
2. The fixed N-step "hand over" control window and the action-masking that zeroes the walker's gradient.
3. The separately trained jump specialist checkpoint (keep its weights only as an optional *initialization*
   for the single policy, not as a runtime module).
4. Any reward that assumes the void is a separate task; fold it into the one course reward.

### Ordered plan
**Stage 0 — terrain plumbing (Isaac Lab).** Rebuild the course as grid sub-terrains (row = difficulty,
column = type) with a column containing gaps and one containing stepping stones, using
`MeshGapTerrainCfg` (`gap_width_range`) and `HfSteppingStonesTerrainCfg`
(`stone_width_range`, `stone_distance_range`) so the built-in row-based curriculum drives gap width
([docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html)). Keep slopes and
stairs in other columns. Set `curriculum=True` and `max_init_terrain_level` low (e.g. 0–1).

**Stage 1 — terrain curriculum on the full course, ONE policy.** If you want zero-new-machinery, use the
existing Isaac Lab curriculum: level up when the robot crosses the sub-terrain border, level down if it
moves less than half the expected distance ([legged_gym rule](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py);
[Rudin 2022](https://arxiv.org/abs/2109.11978)). Gap width grows as the difficulty variable from ~0.1 m to
~0.4 m. Do **not** start at 0.37–0.78 m; that is the end of the curriculum, not the beginning.

**Stage 2 — reward/objective fix (this is why your walker stopped).** Use a **goal+time (position) objective
plus forward progress**, not velocity tracking alone. Rudin 2022 shows velocity tracking limits what one
policy can learn; JumpER 2025 shows velocity-tracking RL converges to a "spin in place" local optimum at
gaps and that an explicit goal-reaching objective fixes it. Concretely: keep your forward-progress /
goal-distance shaping, add a terminal goal reward, and keep a *modest* orientation/action-rate penalty —
do not let stability penalties dominate (your earlier failures were stationary local optima).

**Stage 3 — optional but literature-standard: privileged teacher → DAgger distillation.** If/when you move
the deployment input from height scan to a depth camera, follow the standard two-stage pipeline: teacher RL
with privileged height scan, then DAgger-distill into a single recurrent student that takes
depth+proprioception ([Miki 2022](https://arxiv.org/abs/2201.08117), [Agarwal 2023](https://arxiv.org/abs/2211.07638),
[Cheng 2024](https://arxiv.org/abs/2309.14341), [Lee 2020](https://arxiv.org/abs/2010.11251)). If you stay
in sim with height scan as the actor input, **skip distillation** — the height-scan actor is already the
privileged signal, and Zhuang 2023/Agarwal 2023 show you can train the single policy directly on it.

**Stage 4 — if you still want a leap that the walking policy cannot reach.** Do not add a runtime trigger.
Two literature-compatible options, in order of preference:
- (a) **Single masked/generalist policy**: train one policy on the full course with the gap column pushed to
  the hardest width, as Barkour's distilled Transformer and HOVER's masked generalist do; accept the
  curriculum will spend many samples on the hardest column. This is the "single policy" answer and needs no
  router.
- (b) **Trained high-level router over frozen skills** (ANYmal Parkour pattern): only if the leap truly
  cannot be represented by the walking policy. Then the router must be **learned** from perception and
  trained to be aware of skill capabilities ([Hoeller 2023](https://arxiv.org/abs/2306.14874)); it replaces
  your hand trigger entirely. Skills stay frozen at 50 Hz, router at ~5 Hz.

**Stage 5 — evaluation and acceptance.** Evaluate on the *full* course (all columns), report completion,
falls, collisions and per-terrain success separately; require ≥50% completion as the acceptance gate, then
certify. Mirrors the evaluation protocols in the cited works (Barkour scoring; Zhuang/Cheng per-skill success
rates; Agarwal per-terrain success).

### Concrete parameter guidance
- Gap curriculum: 0.10 → 0.40 m (≈ 0.25–1.0× Go1 trunk length) for the single walking+jumping policy; the
  0.78 m end of your current range is a stretch goal only, matching jump-specialist baselines
  ([Zhuang 0.60 m/80%](https://arxiv.org/abs/2309.05665), [Go1 0.8–0.9 m jump-only](https://arxiv.org/abs/2401.16337)).
- Keep the recurrent actor (history is required because a front-mounted camera/scan cannot see under the
  hind feet, [Agarwal 2023](https://arxiv.org/abs/2211.07638)), and prefer many parallel envs with per-env
  level up/down over a single hand-tuned course ([Rudin 2022](https://arxiv.org/abs/2109.11978)).
