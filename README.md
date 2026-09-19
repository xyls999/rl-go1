# RL-Go1 · 分层四足机器人地形运动与速度决策

> English abstract — A two-layer hierarchical controller for a **Unitree Go1** quadruped built on
> **Isaac Lab 2.3.2 / Isaac Sim 5.1**. **Layer 1 (locomotion)** turns a three-dimensional base-velocity
> command into 12 joint-position actions at 50 Hz and is trained on a single, frozen course direction.
> **Layer 2 (speed decision)** reads a simulated terrain-relief map and outputs the velocity command
> (`vx, vy, wz`) at 10 Hz for the frozen Layer-1 policy. Both layers are trained with PPO (RSL-RL) on a
> procedurally randomised mixed-terrain course (slopes, stairs, blind ledges, rough ground, bumps), and
> each layer is gated by an `N`-cell acceptance matrix measured with 256 parallel environments.
> Obstacle avoidance is explicitly **out of scope**; the project targets terrain traversal.

本仓库是 Go1 分层视觉/地形运动控制项目的训练代码、配置、测试与文档。
**没有任何手工步态脚本**：步态由 PPO 自行涌现，本项目只定义契约、奖励、赛道与验收协议。

---

## 目录

- [0. 项目速览](#0-项目速览)
- [1. 目标与范围](#1-目标与范围)
- [2. 分层架构与输入输出](#2-分层架构与输入输出)
- [3. 奖励函数设计](#3-奖励函数设计)
- [4. 赛道与课程设计](#4-赛道与课程设计)
- [5. 训练与评测协议](#5-训练与评测协议)
- [6. 阶段成果](#6-阶段成果)
- [7. 视频素材](#7-视频素材)
- [8. 复现步骤](#8-复现步骤)
- [9. 仓库结构](#9-仓库结构)
- [10. 已知限制与后续工作](#10-已知限制与后续工作)

---

## 0. 项目速览

| | 第一层 · 运控 (Locomotion) | 第二层 · 速度决策 (Speed) |
|---|---|---|
| 策略文件 | `ActorCore("locomotion")` | `ActorCore("navigation")` |
| 观测维度 | **1073**（critic 1080） | **193**（critic 200） |
| 动作 | **12** 关节位置偏移，50 Hz | **3** 速度指令 `vx, vy, wz`，10 Hz |
| 地形输入 | 高度扫描 512 点 + 置信度 512 | 地形起伏图 72 点 + 置信度 72 |
| 指令来源 | 训练时在冻结赛道方向上采样（含速度层给出的指令） | 策略自身输出，经平滑/时延后交给冻结的第一层 |
| 评价 | 16 格矩阵（难度 × 速度）逐格越线完成率 | 4 格速度带越线完成率 |
| 状态 | 已在旧赛道冻结并认证；新赛道重训中 | 旧赛道全部通过；新赛道重训中 |

演示视频（见 [第 7 节](#7-视频素材)）：

- `media/layer1_speed_tracking_long_course.mp4` — 第一层单方向速度跟踪，跨越新长赛道
- `media/two_layer_dense_easy.mp4` / `media/two_layer_dense_medium.mp4` — 两层联动跨越密集混杂赛道

---

## 1. 目标与范围

**目标**：让 Go1 在**随机化、不平整、连续**的赛道上以较快速度稳定通过全程并到达终点。

**明确的范围界定**：

- ✅ 分层训练：低层运控（电机级）与高层速度决策（任务级）分开训练、低层训好后**冻结**。
- ✅ 单方向：第一层的速度指令表达在**回合开始时冻结的赛道方向**上，而不是机体当前朝向。
- ✅ 由策略自学步态，不使用手写 trot/gait clock 奖励。
- ❌ **不做避障**：第二层读取地形起伏图，只决定 `vx/wz`。第二层配置中的壁面减速与绕障项
  （`wall_proximity_brake`、`obstacle_evasion`）权重为 **0**，场景也不生成离散障碍盒。
- ❌ 不做多机器人协同、不做真实硬件 UWB（仿真中的目标点是模拟的相对 XY 量测）。

---

## 2. 分层架构与输入输出

### 2.1 调用链

```text
第二层策略 (10 Hz)
  └─ action[3] ──► command_from_action ──► 鲁棒性增益 ──► CommandRouter(0.2 s 平滑, 失效即停)
        │                          │
        │                          └─► 指令历史队列（把“第一层看到的指令”延迟若干控制步）
        │
        └─► cmd3 = [vx, vy, wz] ──► _course_speed_scale（下坡按前方高度差限速）
                                   │
                                   ▼
第一层策略 (50 Hz) ──► action[12] ──► 关节目标 = 默认位姿 + 0.25 · action ──► 电机
```

- 物理步长 **200 Hz**，运控 **50 Hz**，视觉/地形 **20 Hz**，导航 **10 Hz**（`contracts.py`）。
- 第二层的 `vx/vy` 是机体坐标系速度指令（允许转弯跟随航线）；第一层**训练时**的指令则表达在
  冻结的赛道基下（`FixedWorldVelocityCommand` 用 `_forward_axis` 把采样指令投影到该基）。

### 2.2 第一层：运控策略

**输入 1073 维**（`contracts.LOCO_OBS_DIM`）：

| 段 | 维度 | 内容 |
|---|---:|---|
| `p42` 本体感觉 | 42 | 角速度 3（×0.25）+ 投影重力 3 + 12 关节相对位置 + 12 关节速度（×0.05）+ 上一动作 12（clip±3 /3） |
| `cmd3` | 3 | `vx, vy, wz`（除以 `[0.8, 0.3, 1.0]` 归一化） |
| 步态相位 | 2 | `sin/cos(2πφ)` |
| 固定航向编码 | 2 | 初始赛道方向在机体坐标系的 `sin/cos`（让策略知道“哪边是赛道前方”） |
| 高度扫描 | 512 | 前方 3.1 m × 1.5 m 网格（0.1 m 分辨率）相对机身高度，`clip(±1 m)` |
| 扫描置信度 | 512 | 射线是否有效（无效即“未知”，不是自由空间） |

**网络拓扑**（`models.py::ActorCore`）：

```text
[前缀 49] ─────────────────────────────────────────────┐
[地形 1024] ── 编码器 Linear(1024→128) ELU Linear(128→32) ELU ──► 潜在 32 ┤
                                                                    ▼
                                        concat(49, 32) = 81 ──► GRU(81→128)
                                                                    ▼
                            head Linear(128→256) ELU Linear(256→128) ELU Linear(128→12)
```

**输出 12 维**关节位置偏移；实际目标为 `默认关节角 + 0.25 × action`，再按软限位截断。
运控输出**只**用于驱动电机，不包含任何特权仿真状态。

### 2.3 第二层：速度决策策略

**输入 193 维**（`contracts.NAV_OBS_DIM`）：

| 段 | 维度 | 内容 |
|---|---:|---|
| `p42` 本体感觉 | 42 | 同上（与第一层一致） |
| `goal4` | 4 | 目标相对位置 `xy/10`、有效性、归一化时效 `age/0.5` |
| 上一指令 | 3 | 上一控制步的 `cmd3`（归一化） |
| 地形起伏图 | 72 | 6×12 网格的地形高度（相对机身，`clip(±1 m)`） |
| 起伏置信度 | 72 | 该格是否已知（相机视锥内且近期可见；未知 ≠ 可通行） |

**网络拓扑**（`models.py::ActorCore("navigation")`）：

```text
[前缀 49] ─────────────────────────────────────────────┐
[地形 144] ── 编码器 Linear(144→128) ELU Linear(128→32) ELU ──► 潜在 32 ┤
                                                                    ▼
                                        concat(49, 32) = 81 ──► GRU(81→128)
                                                                    ▼
                            head Linear(128→128) ELU Linear(128→64) ELU Linear(64→3)
```

**输出 3 维**，映射为速度指令：`vx = (a₀+1)·0.4 ∈ [0, 0.8]`、`vy = a₁·0.3`、`wz = a₂ ∈ [-1, 1]`。
指令经 `CommandRouter` 做 0.2 s 一阶平滑；目标失效/过期时指令强制归零（安全状态）；
`_course_speed_scale` 再依据**前方高度落差**（由高度扫描实时计算）按比例限速，下坡自动减速。

> **变体**：`hier_v2/model_v2.py` 提供另一套 fwwb 风格编码器（`512→256→32` + L2 归一化，
> `EncoderActorCore`）。它仅在显式传入 `--reward-contract loco_v2`/`nav_v2` 时启用；
> **本仓库发布的 checkpoint 全部使用上面的 `ActorCore/CriticCore` 拓扑**。

### 2.4 Critic（非对称特权评价器）

Critic 使用**独立的地形编码器**（不与 actor 共享权重），并在 GRU 之前拼接 7 个特权量：
真实机体线速度 3 + 四足接触标志 4。

| 层 | actor 观测 | critic 观测 | GRU 输入 |
|---|---:|---:|---:|
| 第一层 | 1073 | 1080 | 49+32+7 = 88 |
| 第二层 | 193 | 200 | 88 |

`privileged_state` 只用于训练期价值估计；actor 部署路径不读取任何特权状态。

### 2.5 感知输入

- **高度扫描（第一层）**：Isaac Lab `height_scanner`，网格 `size=(3.1, 1.5)`、`resolution=0.1`、
  安装偏移 `(1.05, 0, 20)`（相对机身，向前 1.05 m、z 抬高 20 m 向下投），50 Hz 更新，512 点。
- **地形起伏图（第二层）**：由同一 512 点扫描**稀疏采样**为 6×12 = 72 格，输出"高度 + 置信度"。
  置信度来自相机视锥覆盖与约 2 s 的可见性记忆：**未知空间不计为自由空间**。
- 模拟相机为 80×60 前向相机、俯角 20°、Z 深度 0.2–6 m（视觉蒸馏阶段使用；`stage3a/3b`、`stage4`）。

### 2.6 契约与版本

`contracts.interface_manifest()` 是**机器可校验的唯一接口规范**，`SCHEMA_VERSION = 5`：

| schema | 变化 | 兼容策略 |
|---:|---|---|
| 3 → 4 | 观测前缀新增固定航向编码（47 → 49） | `migrate_locomotion_heading_state`：旧列原位搬移、新列零初始化（保持策略函数不变） |
| 4 → 5 | 第二层几何从"障碍距离扫描 144"改为"地形起伏图 144"（6×12×2） | `_is_schema4_relief_migration`：仅在**唯一差异**是键名 `obstacle → terrain_relief` 时允许搬迁；维度变化一律拒绝 |

checkpoint 加载时会比对 `interface_manifest()`；不匹配且不属于上述迁移即报错，避免静默降级。
详见 [`docs/interfaces.md`](docs/interfaces.md)。

---

## 3. 奖励函数设计

奖励以 `configs/rewards_stage1_v20_paper.toml`（第一层，42 项）与 `configs/rewards_nav_v2.toml`
（第二层，15 项）为**唯一真源**；`reward_config.py` 做严格模式校验（未知项、缺项、符号错误即拒绝启动），
并且奖励配置会写入 checkpoint 元数据，续训时校验一致。设计灵感来自 Rudin et al.,
*Learning to Walk in Minutes* (CoRL 2021)。

### 3.1 第一层（42 项）

**A. 前进与任务目标**

| 项 | 形状 | 权重 | 说明 |
|---|---|---:|---|
| `track_linear_velocity` | `exp(-‖v_xy−c_xy‖²/std²)` | 1.00 | 世界/赛道基速度跟踪（std 0.30） |
| `track_yaw_velocity` | `exp(-(ωz−c_wz)²/std²)` | 0.50 | 偏航角速度跟踪（std 0.35） |
| `forward_velocity` | `exp(-(v_fwd−0.70)²/std²) − 0.50·max(−v_fwd,0)` | 0.65 | 固定方向前进速度；**倒退受罚** |
| `forward_progress` | `clamp(Δs/(dt·0.60), −1, 1)` | 1.20 | 沿固定方向的位移增量 |
| `goal_closeness` | `1 − d/34` | 0.20 | 接近终点的稠密信号 |
| `course_checkpoint` | 一次性 1.0 ×3 | 1.20 | 25 % / 50 % / 75 % 里程碑 |
| `goal_reached` | 一次性 1.0 | 80.0 | 到达终点条带（稀疏，防稀疏奖励不触发） |
| `straight_line` | `exp(-‖横向偏移‖²/std²)` | 1.00 | 保持直线（std 0.75） |
| `velocity_direction` | `exp(-atan2(v_lat, v_fwd)²/std²)·gate` | 1.50 | 速度方向与赛道方向一致（min_speed 0.10） |
| `cross_track_l2` | `min(((‖lat‖−0.10)/0.75)², 4)` | −1.50 | 横向越界平方代价 |
| `heading_alignment` | `exp(-(航向误差)²/std²)` | 0.60 | 机体朝向与赛道方向一致 |
| `upright` | `clamp(−g_z, 0, 1)` | 0.20 | 抗翻倒（仅作锚点，不冻结步态） |

**B. 姿态与高度**

| 项 | 形状 | 权重 | 说明 |
|---|---|---:|---|
| `base_height` | `exp(-(h−0.34)²/0.05²)`，低于 0.285 m 加罚 0.5 | 0.60 | 机身高度 |
| `base_vertical_velocity_l2` | `v_z²` | −0.40 | 抑制弹跳 |
| `base_roll_pitch_rate_l2` | `(ωx²+ωy²)·1.5²` | −0.08 | 抑制滚转/俯仰抖动 |
| `base_orientation_l2` | `(roll²+pitch²)/0.35²` | −0.35 | 机身水平 |

**C. 步态相关（多数默认 `0`，交给 PPO 自学）**

| 项 | 权重 | 说明 |
|---|---:|---|
| `foot_clearance` | 0.08 | 落地瞬间的抬脚高度（0.075 m，地形上 0.145 m） |
| `feet_air_time` | 0.20 | 首次落地时结算的有效腾空时间（上限 0.45 s） |
| `feet_slide` | −0.015 | 支撑相打滑（deadband 0.04 m/s，地形缓解系数 0.35） |
| `foot_lateral` | −0.10 | 足端横向外扩/内收/跨体（防外八、内八） |
| `four_feet_airborne` | −0.50 | 四足同时离地（抑制跳跃） |
| `foot_load_balance` | −0.10 | 四足负载长期不均（EMA） |
| `phase_contact` / `phase_clearance` / `trot_gait` / `small_steps` / `step_frequency` / `four_feet_grounded` / `foot_contact_cycle` / `air_time_variance` / `flat_trot_pattern` | 0.0 | 保留实现与参数，默认关闭（手写步态先验会限制地形适应） |

**D. 平滑、能耗与安全**

| 项 | 权重 | 说明 |
|---|---:|---|
| `action_rate_l2` / `action_acceleration_l2` | −0.015 / −0.0008 | 动作平滑 |
| `joint_torque_l2` / `joint_power_l1` / `joint_acceleration_l2` | −1e−5 / −1e−5 / −2.5e−7 | 力矩 / 功率 / 关节加速度 |
| `joint_deviation_l2` / `hip_abduction_deviation_l2` / `hip_abduction_velocity_l2` | −0.035 / −0.035 / −0.0006 | 关节偏离默认位姿（带死区） |
| `fallen` / `collision` / `out_of_bounds` | −5.0 / −1.50 / −2.0 | 摔倒 / 非足端持续碰撞 / 越界 |

> 碰撞语义：**只有躯干等非足端部位**接触力连续 5 个控制步（0.1 s）超过 25 N 才判碰撞；
> 腿部擦碰台阶不终止回合，而是通过分级代价整形。

### 3.2 第二层（15 项）

| 项 | 权重 | 说明 |
|---|---:|---|
| `track_lin_vel_xy` / `track_ang_vel_z` | 0.15 / 0.10 | 对自身输出指令的执行跟踪 |
| `path_progress` | 4.0 | 沿"起点 → 终点"投影的推进量（max_speed 0.80） |
| `bearing_progress` | 0.25 | 朝目标方位的角向推进 |
| `reach_goal` | **300.0** | 到达终点（阈值 0.60 m，条带起点 33.10，半宽 4 m） |
| `settled_goal` | 100.0 | 到达后停下（线速度 ≤0.18、角速度 ≤0.30） |
| `goal_braking` | −2.0 | 接近终点时超速未制动 |
| `stuck_duration` | −1.5 | 长时间原地不动 |
| `unsafe_tilt` / `tilt_recovery` | −8.0 / 0.60 | 危险倾角 / 倾角恢复 |
| `wall_proximity_brake` / `obstacle_evasion` | **0.0** | 明确不做避障（保留实现，权重置零） |
| `navigation_time` | −0.002 | 每步时间成本 |
| `undesired_contacts` / `termination` | −1.0 / −250.0 | 非期望接触 / 任务失败 |

**分层原则**：第二层奖励**不含任何关节级项**（它无法直接影响关节），关节级优化完全属于第一层。

### 3.3 设计取舍与教训

1. **不写死步态**：曾尝试 gait clock、对角同步、抬腿高度、步频等手工项，权重最终清零；
   实测手工项会把策略推向"原地高频抬腿"或"跳跃"这类刷分行为。
2. **奖励塑形会造成站立局部最优**：脚底打滑惩罚过重（线性、无死区）时，策略学会"站住不动"；
   现在只惩罚**接触相**的切向滑移，且带 0.04 m/s 死区与地形缓解。
3. **四脚同时着地也会被误判为好步态**：`trot_gait` 现要求"恰好两脚接触且为对角"，
   并新增四足同时离地惩罚。
4. **稀疏终点必须真正可达**：终点实现为**矩形条带**（33.10–34.10 m × 半宽 4 m），
   判定条件为**两只前脚同时进入**，避免"机身中心越线"或从旁侧绕开地形即算成功。
5. **前后项量级要对齐**：`small_steps` 曾误乘 `1/dt` 放大 50 倍，导致策略用碎步刷分。

---

## 4. 赛道与课程设计

赛道由 `envs/geometry.py` **确定性**程序化生成（单调、不可跨越的随机化），每个环境一块
**40 m × 40 m** 的 tile（环境间距 44 m），特征区覆盖 **5.30 m → 35.00 m（≈29.7 m）**，
最后是终点前的水平段与终点条带。

**5 类地形块**（每块带 x 方向两端渐变，保证与基准面平滑衔接）：

| 类型 | 参数 | 作用 |
|---|---|---|
| `slope_block` | 上坡再下坡；坡度 `U(5°, 5°+25°·d)`，坡长 0.90–1.60 m，可带侧倾 | 上下坡 |
| `stairs_block` | 3–5 级；单级 `U(0.04, 0.04+0.12·d)` m，踏面 `0.40±0.03` m，平台 0.30–0.60 m；下行用**随机重排**的等高集合 → 高矮交错且严格回落到基准 | 上下楼梯 |
| `ledge_block` | 爬升 → 短平台 → **单级 ≤0.16 m 的合法下落** → 回基准 | 盲台阶/落差 |
| `rough_block` | 1.0–2.2 m 起伏面，幅度 ×1.8 | 不平整地面 |
| `bumps_block` | 2–5 个高斯凸包，峰 0.04–0.06+0.06d m，宽 0.16–0.30 m | 连续凸起 |

**生成策略**：先放置一个"必含全部 5 类"的起始组合（随机顺序），再用短块填充至接近特征区末端，
剩余用起伏面补齐 —— 因此**每个 tile 都同时包含坡、楼梯、盲台阶、粗糙地与凸包**（13–15 段），
且每 tile 至少 2 段楼梯、1–2 段坡。随机性由 tile 的 RNG 派生，同一 seed 可完整复现。

**物理包线**（由测试强制，防止生成不可跨越地形）：

| 约束 | 上限/下限 | 依据 |
|---|---|---|
| 坡度 | ≤ **30°** | 最差静态摩擦 0.6 → `atan(0.6) ≈ 31°` |
| 单级台阶 | ≤ **0.16 m** | 腿长 0.25+0.25 m，站立髋高 ≈0.365 m |
| 踏面深度 | ≥ **0.35 m** | 0.6 m 躯干跨步需要 |
| 终点前平地 | ≥ **1.10 m** | 保证越线判定与停下 |
| 每 tile 三角面 | **≤ ~21 k** | 256 环境 ≈5.4 M 面；超过约 10 M 面会导致 PhysX **cooking 失败** → 地形网格缺失 → 全环境摔倒 |

> 赛道网格分块与面数预算是**硬约束**：曾因单 tile 62 k 面（256 环境 16.1 M 面）触发
> `cooking failure`，表现为"completion = 0、fallen = 1.0"而诊断日志只在启动时留下 2 行警告。

---

## 5. 训练与评测协议

### 5.1 PPO 超参（`configs/ppo_v20_paper.json`）

```json
{ "num_steps_per_env": 48, "gamma": 0.99, "learning_rate": 3e-4, "clip_param": 0.2,
  "lam": 0.95, "num_learning_epochs": 5, "num_mini_batches": 4, "entropy_coef": 0.005,
  "min_noise_std": 0.05, "max_noise_std": 1.0,
  "warm_start_noise_std": 0.45, "warm_start_learning_rate": 3e-4,
  "warm_start_actor_learning_rate": 5e-5, "warm_start_clip_param": 0.1,
  "warm_start_entropy_coef": 0.002, "value_loss_coef": 1.0, "max_grad_norm": 1.0,
  "desired_kl": 0.01, "save_interval": 50 }
```

- 批量：**256 环境**，双卡 `torchrun --nproc_per_node=2`（数据并行，梯度同步）。
- 热启动：`--initialize-from <ckpt>` 载入策略权重但重置优化器/噪声；`--critic-warmup-iterations N`
  先只训 critic（actor 冻结），避免早期优势估计噪声破坏已有步态。
- 鲁棒性课程：`--robustness on` 打开 sim-to-real 课程（指令增益、时延、观测噪声）。

### 5.2 评测协议（关键）

> **评测环境数必须等于训练批量（256）。**
> GPU PhysX 的接触求解依赖同时模拟的环境数量：同一 checkpoint 在难度带 0.82–0.85 上测得
> **1 环境 100 %、4 环境 100 %、16 环境 0 %、256 环境 8.2 %**。用 1/4 环境探针会**系统性高估**通过率。
> 因此所有门禁与报数都固定 `--num-envs 256`。

- 完成判定：**两只前脚同时进入终点矩形条带**且回合结束（`finish_crossed`），
  同时满足 `fallen=0`、`out_of_bounds=0`；碰撞单独统计。
- 评测为确定性策略（`act_inference`，无探索噪声）。

### 5.3 冻结门（16 格矩阵）

`scripts/eval_gate_stage1.sh` 定义**难度 × 速度** 16 格（默认难度 `0.25/0.5/0.75/1.0` ×
速度 `0.3/0.5/0.8/1.0`），每格 **256 回合**，验收阈值：

```text
completion ≥ 0.90  ∧  collision ≤ 0.10  ∧  fallen ≤ 0.05
```

全部格子通过后，`scripts/gate_record.py` 生成一份门禁记录，`certify_checkpoint` 会**重新核算**
每格并写入 `envelope`，得到 `*.validated.pt`。低层只有拿到该记录才允许被冻结并交给上层使用。

---

## 6. 阶段成果

> 下表所有通过率均为**确定性策略 + 256 环境 + 256 回合**的实测结果（除标注"视频"者为 1 回合演示）。
> "赛道版本"标注该结果对应的赛道长度/形态，因为赛道在项目过程中被加长、加难。

### 6.1 第一层（运控）

| 赛道 | 检查点 | 评测 | 结果 |
|---|---|---|---|
| 短赛道（特征 9.3 m，终点 19.1–20.1） | `2026-09-19_09-45-00_go1-stage1-terrain-a9/model_699.validated.pt` | 16 格 × 256 回合 | **16/16 PASS**，最弱格 90.23 %（难度 1.0 × 速度 0.3） |
| 密集混杂 v1（21 m） | 同上（a9） | 3 点 × 256 回合 | 0.20–0.45@0.3 → 88.67 %（摔倒 10.94 %）；0.85–1.00@0.5 → 55.08 %；@1.0 → 36.72 %（碰撞 26.17 %）→ **超出旧分布** |
| 密集混杂 v1（21 m） | `2026-09-19_16-00-00_go1-stage1-terrain-a10b-hard/model_599.pt` | 2 点 × 256 回合 | 0.85–1.00@0.5 → **92.19 % PASS**；@1.0 → 87.89 %（碰撞 8.59 %、摔倒 3.52 %） |
| 密集混杂 v2（30 m，含盲台阶） | `2026-09-19_18-00-00_go1-stage1-terrain-a10c-long`（训练中） | 视频 1 回合 | 0.25–0.50 / 指令 0.35–0.55 m/s → **100 %**（碰撞 0、摔倒 0） |

第一层认证冻结记录：`envelope = {min_completion 0.90, max_collision 0.10, max_fallen 0.05}`、
`validation_protocol = stage1-gate-v1`、`metadata.capability_validated = true`。

### 6.2 第二层（速度决策，与冻结第一层联动）

| 赛道 | 检查点 | 评测 | 结果 |
|---|---|---|---|
| 短赛道 | `2026-09-19_11-45-00_go1-stage2-speed-v1/model_350.pt` | 4 格 × 256 回合 | 0.20–0.40 → **99.61 %**；0.40–0.60 → **100 %**；0.60–0.80 → **100 %**；0.80–1.00 → **97.66 %**（碰撞 0.78 %）→ **4/4 PASS** |
| 短赛道（全带） | 同上 | 0.20–1.00 × 256 回合 | **99.61 %**（碰撞 0.39 %、摔倒 0、超时 0） |
| 密集混杂 v1（21 m） | 同上 | 2 带 × 256 回合 | 0.20–0.60 → 90.23 %；0.60–1.00 → 62.89 %（碰撞 12.5 %、摔倒 25.4 %）→ 需按新赛道重训 |
| 密集混杂 v1（21 m） | 同上 + 冻结第一层 a9 | 视频 1 回合 ×2 | 0.20–0.50 → **100 %**；0.45–0.80 → **100 %** |

---

## 7. 视频素材

| 文件 | 内容 | 赛道 / 参数 | 结果 |
|---|---|---|---|
| `media/layer1_speed_tracking_long_course.mp4` | **第一层单方向速度跟踪**（只跑运控，蓝色箭头=固定方向指令，绿色=实际速度） | 密集混杂 v2（30 m），难度 0.25–0.50，指令 0.35–0.55 m/s，75 s | completion **100 %**、collision 0、fallen 0 |
| `media/two_layer_dense_easy.mp4` | **两层联动**：第二层按地形起伏决策速度，第一层执行 | 密集混杂 v1（21 m），难度 0.20–0.50 | completion **100 %**、collision 0、fallen 0 |
| `media/two_layer_dense_medium.mp4` | 同上，更难带 | 密集混杂 v1（21 m），难度 0.45–0.80 | completion **100 %**、collision 0、fallen 0 |

预览帧位于 `media/frames/`。视频由 Isaac Sim **内部 RGB 录制**（`--record-video`）生成，
相机固定跟随第 0 号环境，画面中不含桌面/编辑器 UI。

---

## 8. 复现步骤

### 8.1 依赖

- Isaac Lab **2.3.2** / Isaac Sim **5.1**，Python 3.11（示例路径 `/root/env_isaaclab`）
- PyTorch **2.7.0+cu128**，`rsl-rl-lib 3.1.2`
- 2 × NVIDIA A10（23 GB）；单卡亦可运行，减小批量即可
- `source/go1_hier_sim/` 为可编辑安装的 Python 包（`pyproject.toml`）

```bash
export ISAACLAB_PATH=/config/Desktop/IsaacLab
cd /config/Desktop/sim            # 本仓库根目录
./sim.sh test                     # 全部单元测试（98 项）
```

### 8.2 生成赛道并自检

```bash
# 打印 12 个随机种子的赛道剖面（坡度/单级台阶/块类型/面数预算）
PYTHONPATH=source/go1_hier_sim /root/env_isaaclab/bin/python - <<'PY'
import numpy as np
from go1_hier_sim.envs.geometry import build_tile
for seed in range(12):
    tile = build_tile("course", 1.0, np.random.default_rng(seed), obstacles=False)
    p = tile.metadata["course_profile"]
    print(seed, sorted(p["feature_kinds"]), len(p["blocks"]), max(p["step_rises"]), len(tile.faces))
PY
```

### 8.3 训练

```bash
# 第一层（课程化）：先宽难度，再硬难度；--initialize-from 承接上一阶段
GO1_RUN_ID=<run-id> ./scripts/torchrun_sim.sh train --stage stage1 --terrain course --headless \
  --num-envs 256 --iterations 900 --episode-seconds 150 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --speed-min 0.30 --speed-max 1.00 \
  --initialize-from <previous.pt> --critic-warmup-iterations 40 --robustness on \
  --run-name <run-id>

# 第二层：冻结第一层（必须用 validated checkpoint），只训速度层
GO1_RUN_ID=<nav-run-id> ./scripts/torchrun_sim.sh train --stage stage2 --terrain course --headless \
  --num-envs 256 --iterations 1500 --episode-seconds 120 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --locomotion-checkpoint <layer1.validated.pt> \
  --run-name <nav-run-id>
```

### 8.4 评测 / 门禁 / 录制视频

```bash
# 单元验证（务必 256 环境）
./sim.sh evaluate --stage stage1 --terrain course --headless --device cuda:0 \
  --num-envs 256 --episodes 256 --episode-seconds 150 \
  --course-difficulty-min 0.85 --course-difficulty-max 1.00 --speed-min 0.4 --speed-max 0.6 \
  --checkpoint <ckpt>

# 16 格冻结门（一个格子一次调用，可断点续跑）
./scripts/eval_gate_stage1.sh <ckpt> 256 0.3,0.5,0.8,1.0 0.25,0.5,0.75,1.0 cuda:1 150 256 1.0

# 内部视频录制（渲染仅支持 cuda:0）
./sim.sh evaluate --stage stage2 --terrain course --headless --device cuda:0 \
  --num-envs 1 --episodes 1 --episode-seconds 120 --video-seconds 45 \
  --course-difficulty-min 0.20 --course-difficulty-max 0.50 \
  --locomotion-checkpoint <layer1.validated.pt> --checkpoint <layer2.pt> \
  --follow-env 0 --show-direction-arrows --record-video media/demo.mp4
```

---

## 9. 仓库结构

```text
.
├── README.md                  # 本文件
├── agent.md                   # 项目长期记忆：已验证事实、接口版本、议题与教训
├── AGENTS.md                  # 工作区约定（改动前先读 agent.md）
├── sim.sh                     # 统一入口：test / check / train / evaluate / export
├── configs/
│   ├── rewards_stage1_v20_paper.toml   # 第一层奖励契约（42 项，唯一真源）
│   ├── rewards_nav_v2.toml             # 第二层奖励契约（15 项）
│   ├── rewards_loco_v2.toml            # 可选 loco_v2 契约（hier_v2 变体）
│   ├── rewards_stage1.toml             # 历史默认契约（兼容回退）
│   └── ppo_v20_paper.json              # PPO 超参
├── docs/
│   ├── architecture.md         # 架构契约（维度/频率/冻结规则）
│   ├── interfaces.md           # 观测/动作契约与 schema 迁移细则
│   ├── rewards_stage1.md       # 第一层奖励定义与调参约束
│   ├── results.md              # 全部实测结果与出处
│   └── training.md             # 训练/评测命令手册与踩坑记录
├── media/                      # 演示视频与预览帧
├── scripts/
│   ├── eval_gate_stage1.sh     # 16 格冻结门
│   ├── gate_record.py          # 门禁记录生成
│   ├── torchrun_sim.sh         # 双卡训练入口
│   ├── resume_v20_course.sh    # 课程化续训流程（分阶段）
│   ├── bootstrap_runtime.sh    # 项目局部运行时恢复
│   └── sim.py                  # 薄封装（转发到 go1_hier_sim.cli）
├── source/go1_hier_sim/go1_hier_sim/
│   ├── contracts.py            # 版本化接口规范（schema 5）
│   ├── models.py               # actor/critic 网络（ActorCore / CriticCore）
│   ├── envs/{config,geometry,environment,runtime}.py   # 场景、赛道、环境、运行时
│   ├── hier_v2/                # fwwb 风格可选契约：奖励、模型变体、鲁棒性课程
│   ├── training/{on_policy,distillation}.py            # PPO runner 与视觉蒸馏
│   ├── checkpoints.py          # 严格 checkpoint + 认证（certify）
│   ├── gate.py                 # 冻结门协议/记录
│   ├── export.py               # TorchScript 导出（完整部署图）
│   └── cli.py                  # 命令行入口
└── tests/                      # 98 项单元测试（契约/几何/奖励/门禁/模型/视频）
```

---

## 10. 已知限制与后续工作

1. **评测环境数依赖**：GPU PhysX 接触求解随批量变化，必须用训练批量（256）评测；
   更小的批量会系统性高估通过率。这是仿真器特性，已写入协议。
2. **赛道网格面数预算**：单 tile ≤ ~21 k 面。加长或加密赛道时若超过约 10 M 面（256 环境），
   PhysX cooking 会失败且只在启动日志留下警告，表现为"全环境摔倒"。
3. **第二层尚无独立门禁协议**：`certify_checkpoint` 的 stage2 路径仍沿用旧的 `terrain=="mixed"`
   判定，需要为速度层定义专门的验收协议（当前只做逐格评测）。
4. **`hier_v2` 编码器变体未被发布 checkpoint 使用**：只有显式传 `--reward-contract loco_v2|nav_v2`
   才会启用 `EncoderActorCore`；`_ppo_config` 读取的是**原始 CLI 参数**，而环境侧按 stage 推导契约，
   两者在未显式传参时不总是同步 —— 需要统一（见 `docs/training.md` 的已知不一致）。
5. **赛道仍在迭代**：当前地形（40 m tile、30 m 特征区、含盲台阶）正在重训两层；
   第 6 节标注了每个数字对应的赛道版本。
6. **不做避障**：第二层的壁面/绕障奖励权重为 0，场景不生成障碍盒；如需避障需另立契约。
7. **无 CI / 无 LICENSE 决策**：本仓库当前只要求在 Isaac 环境内跑 `./sim.sh test`
   （98 项）；纯 CPU 的 CI 与许可证选择待定。

---

## 引用

- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning.* CoRL 2021 (PMLR 164:91–100).
- Isaac Lab — <https://github.com/isaac-sim/IsaacLab>；RSL-RL — <https://github.com/leggedrobotics/rsl_rl>
