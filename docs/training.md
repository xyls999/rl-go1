# 训练与评测手册

本文件记录**可复制的命令**与**必须遵守的坑**。所有命令以仓库根目录为工作目录。

## 0. 环境自检

```bash
./sim.sh test                       # 103 项单元测试：契约 / 几何 / 奖励 / 门禁 / 模型 / 视频 / 起伏图
./sim.sh check --stage stage1 --terrain course --headless --num-envs 4 --device cuda:0
```

`check` 会打印观测与奖励维度，并跑一次真实 physics step；**它不会写 checkpoint**。

## 1. 训练

### 1.1 第一层（运控，课程化）

**v3 赛道的实际训练路线**（两阶段，第二阶段的产物即冻结层）：

```bash
# 阶段 1：宽难度（0.20–1.00）、速度 0.30–1.00
GO1_RUN_ID=2026-09-20_14-00-00_go1-stage1-terrain-v3-broad \
./scripts/torchrun_sim.sh train --stage stage1 --terrain course --headless \
  --num-envs 256 --iterations 900 --episode-seconds 150 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --speed-min 0.30 --speed-max 1.00 \
  --initialize-from checkpoints/stage1/<v2-hard>/model_799.pt \
  --critic-warmup-iterations 40 --run-name 2026-09-20_14-00-00_go1-stage1-terrain-v3-broad

# 阶段 2：硬难度收束（0.85–1.00）、速度 0.35–0.60 → 完成后冻结
GO1_RUN_ID=2026-09-20_16-30-00_go1-stage1-terrain-v3-hard \
./scripts/torchrun_sim.sh train --stage stage1 --terrain course --headless \
  --num-envs 256 --iterations 800 --episode-seconds 150 \
  --course-difficulty-min 0.85 --course-difficulty-max 1.00 \
  --speed-min 0.35 --speed-max 0.60 \
  --initialize-from checkpoints/stage1/2026-09-20_14-00-00_go1-stage1-terrain-v3-broad/model_899.pt \
  --critic-warmup-iterations 40 --run-name 2026-09-20_16-30-00_go1-stage1-terrain-v3-hard
```

通用模板：

```bash
INIT=checkpoints/stage1/<previous>/model_<n>.pt
GO1_RUN_ID=2026-09-19_18-00-00_go1-stage1-terrain-a10c-long \
GO1_REWARD_CONFIG_PATH=$PWD/configs/rewards_stage1_v20_paper.toml \
GO1_PPO_CONFIG_PATH=$PWD/configs/ppo_v20_paper.json \
./scripts/torchrun_sim.sh train --stage stage1 --terrain course --headless \
  --num-envs 256 --iterations 900 --episode-seconds 150 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --speed-min 0.30 --speed-max 1.00 \
  --initialize-from $INIT --critic-warmup-iterations 40 --robustness on \
  --run-name <run-id>
```

- `--critic-warmup-iterations N`：**先只训练 critic**，actor 权重与动作均值逐位冻结，避免早期优势噪声破坏已有步态（热启动必用）。
- `--initialize-from`：载入策略权重、**丢弃**旧优化器与探索噪声（`log_std` 重新初始化为 `warm_start_noise_std`）。
- `--episode-seconds`：必须 ≥ 路线长度 / 最低速度（30 m / 0.30 m/s ≈ 113 s → 用 150 s）。
- 长跑建议 `nohup setsid ... &` 并把日志写到文件，再周期性 grep `Learning iteration`。

### 1.2 第二层（速度决策）

**v3 赛道的实际训练路线**（两阶段，第二阶段的产物为当前采用模型）：

```bash
# 阶段 1：宽难度 0.20–1.00
GO1_RUN_ID=2026-09-21_13-45-00_go1-stage2-speed-v3-broad \
./scripts/torchrun_sim.sh train --stage stage2 --terrain course --headless \
  --num-envs 256 --iterations 1200 --episode-seconds 150 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --locomotion-checkpoint checkpoints/stage1/2026-09-20_16-30-00_go1-stage1-terrain-v3-hard/model_799.pt \
  --allow-uncertified-locomotion \
  --run-name 2026-09-21_13-45-00_go1-stage2-speed-v3-broad

# 阶段 2：难带收束 0.85–1.00
GO1_RUN_ID=2026-09-21_17-00-00_go1-stage2-speed-v3-hard \
./scripts/torchrun_sim.sh train --stage stage2 --terrain course --headless \
  --num-envs 256 --iterations 800 --episode-seconds 150 \
  --course-difficulty-min 0.85 --course-difficulty-max 1.00 \
  --locomotion-checkpoint checkpoints/stage1/2026-09-20_16-30-00_go1-stage1-terrain-v3-hard/model_799.pt \
  --allow-uncertified-locomotion \
  --run-name 2026-09-21_17-00-00_go1-stage2-speed-v3-hard
```

- **stage2 没有速度带**：不要传 `--speed-min/--speed-max`（`make_config` 只对 stage1 生效，
  且 `speed_max > 0.8` 会被直接拒绝）。第二层的包线只有"难度"一维。
- `--locomotion-checkpoint` 默认以 `require_validated=True` 加载（只接受带门禁记录的 checkpoint）；
  用未认证但已实测过的第一层时必须显式加 `--allow-uncertified-locomotion`。
- 单卡/双卡实测每轮耗时接近（多进程数据并行只增大每轮样本量，不减少 wall-clock）：
  stage2 约 12–14 s/轮（256 环境 × 每卡）。

## 2. 评测（单元验证）

```bash
./sim.sh evaluate --stage stage1 --terrain course --headless --device cuda:0 \
  --num-envs 256 --episodes 256 --episode-seconds 150 \
  --course-difficulty-min 0.85 --course-difficulty-max 1.00 \
  --speed-min 0.4 --speed-max 0.6 \
  --checkpoint <ckpt>
```

报告写入 `reports/evaluate_<stage>_<timestamp>.json`，关注字段：

```text
completion_rate  success_rate  collision_rate  fallen_rate  out_of_bounds_rate
timeout_rate     velocity_xy_rmse  yaw_rmse
```

stage2 报告另有**指令稳定性**字段（用于判断速度层是否"更稳"）：

```text
mean_command_delta_l1         # 相邻控制步的 |Δvx| 平均（m/s per 0.1 s）
command_step_large_fraction   # |Δvx| > 0.1 m/s 的步数占比
```

`passed` 字段是**便利标记**，判定口径为：

```text
stage1: terrain == "course" ∧ completion ≥ 0.95 ∧ velocity_xy_rmse ≤ 0.15 ∧ yaw_rmse ≤ 0.25
stage2: terrain == "course" ∧ success_rate ≥ 0.90 ∧ collision_rate ≤ 0.05
```

它**不包含** `fallen ≤ 0.05`（门禁比它更严），也不检查环境数、回合数或 cooking failure。
**请以报告里的显式指标为准，不要用 `passed` 下结论。**

## 3. 冻结门（16 格）与认证

```bash
# 一个格子一次调用；可断点续跑（每次只打印/追加一行）
./scripts/eval_gate_stage1.sh <ckpt> 256 0.3,0.5,0.8,1.0 0.25,0.5,0.75,1.0 cuda:1 150 256 1.0
```

参数顺序：`<checkpoint> <episodes> <speeds-csv> <difficulties-csv> [device] [episode-seconds] [env-count] [max-speed]`。

- 每格 `half = speed-0.1`、`upper = min(max_speed, speed+0.1)`，因此速度带是**以格点为中心的 ±0.1 区间**。
- 验收：`completion ≥ 0.90 ∧ collision ≤ 0.10 ∧ fallen ≤ 0.05`。
- 跑完整矩阵后会写 `reports/gate_stage1_<ts>.json`，再执行脚本末尾打印的 `certify_checkpoint` 即得 `*.validated.pt`。

**第二层验收**（只按难度遍历；stage2 无速度带）：

```bash
./scripts/eval_gate_stage2.sh <locomotion-ckpt> <navigation-ckpt> 256 0.5,0.75,0.9 cuda:0 150 256
```

参数顺序：`<locomotion> <navigation> <episodes> <difficulties-csv> [device] [episode-seconds] [env-count]`。
判定口径与第一层相同（completion ≥0.90 ∧ collision ≤0.10 ∧ fallen ≤0.05），并额外打印指令稳定性指标。

## 4. 录制演示视频

```bash
./sim.sh evaluate --stage stage2 --terrain course --headless --device cuda:0 \
  --num-envs 1 --episodes 1 --episode-seconds 120 --video-seconds 45 \
  --course-difficulty-min 0.20 --course-difficulty-max 0.50 \
  --locomotion-checkpoint <layer1-validated>.pt --checkpoint <layer2>.pt \
  --follow-env 0 --show-direction-arrows --record-video media/demo.mp4
```

- **渲染只支持 `--device cuda:0`**（其它设备会报 "renderer supports camera workloads only on cuda:0"）。
- `--record-video` 先写 `<stem>-step-0.mp4` 再自动重命名；`--follow-env 0` 让相机跟随第 0 环境；
  `--show-direction-arrows` 叠加"指令速度（蓝）/实际速度（绿）"箭头。
- 抽帧检查：`ffmpeg -y -ss 15 -i media/demo.mp4 -frames:v 1 -q:v 2 /tmp/frame.jpg`。

## 5. 必读的坑

1. **评测环境数必须等于训练批量**。GPU PhysX 接触求解随模拟批量变化：同一 checkpoint 在
   难度 0.82–0.85 上 1/4 环境 100 %、16 环境 0 %、256 环境 8.2 %。**用 1/4 环境探针会系统性高估。**
2. **网格面数预算**。单 tile 面数 × 环境数 ≈ 总面数；超过约 10 M 面时 PhysX cooking 失败，
   表现为`completion = 0 / fallen = 1.0`，而日志里只有启动时两行 `cooking failure`。
   当前 v3 配置：**≤16.6 k 面/tile → 256 环境 4.26 M 面**。**加长赛道后必须重新检查面数。**
3. **`--num-envs 512` 不可用**（会触发上述 cooking 失败）；本项目所有报数都用 256。
4. **速度上限**：`contracts.MAX_FORWARD_COMMAND = 1.0`，同时约束 `CommandRouter` 限幅与
   `envs/config.py` 的 `speed_max` 校验。传更大的速度会在启动时报错（不会静默截断）。
   **注意 `speed_max` 校验对 stage2 也会生效，但 stage2 根本不使用速度带** —— 第二层评测不要传速度参数。
5. **不要把射线未命中当成"高度 0"**。`raycast_mesh` 对未命中的射线返回 `inf`：任何 `inf * 0` 或
   `inf − inf` 都会把整张几何/起伏图变成 NaN，而 NaN 会一路传进 PPO 的梯度（表现为训练在随机轮次崩、
   报 `contains NaN or infinity`）。规则：**先 `torch.isfinite(...).all(-1)` 屏蔽，再做参考值/均值/梯度**，
   并且分母一律 `clamp_min`；返回给相机或可见性记忆的点也要有限化。
6. **不要再改 `normalize_command` 的 `[0.8, 0.3, 1.0]` 与 manifest 的 `goal_xy = 10.0`**：
   它们是已发布 checkpoint 的归一化契约，改动会使两代模型失效（1.0 m/s 会被归一化为 1.25，
   属于已知且可接受的代价）。
7. **回合时长**：赛道加长到 30 m 后，门禁/评测的 `episode-seconds` 必须 ≥150 s，
   否则"未完成"会被记成超时。
8. **热启动不要沿用旧优化器**：`--initialize-from` 已丢弃优化器与噪声；
   实测若用高学习率/高噪声直接续训，几轮内策略就会坍缩（奖励上升但完成率归零）。
   实测安全的热启动区间：`warm_start_noise_std ≤0.15`、actor lr ≤2e-6、clip ≤0.02。
9. **生成器的填充阈值内外必须一致**：`geometry.py` 外层 `while` 与内层 `break` 用了不同阈值时
   会无限循环（pytest 卡死、100 % CPU）。
10. **`reports/`、`logs/`、`checkpoints/` 不入库**（体量大且与机器绑定）。要复现结果请对照
    `docs/results.md` 的**赛道版本表**与本文的命令重跑。
11. **容器 IP 会变**：训练/评测用 `setsid nohup` detached 运行，SSH 中断不影响其继续，但会丢失观测；
    恢复后先 `pgrep -af 'scripts/sim.py train'` 并 `grep 'Learning iteration' <log>` 重新对表。

## 6. 已知代码不一致（待修）

- `cli.py::_ppo_config` 依据**原始 CLI 参数** `args.reward_contract` 选择策略类
  （`HierActorCriticV2` vs `HierActorCritic`），而环境侧 `_resolve_contract` 会按 stage 推导契约
  （stage2 → `nav_v2`）。因此**未显式传 `--reward-contract` 时，策略类与奖励契约可能不同步**：
  已发布的 checkpoint 全部是 `ActorCore/CriticCore`（`hier_v2` 的 L2Norm 编码器变体未被使用）。
  修复方向：让 `_ppo_config` 使用与 `make_env` 相同的 `_resolve_contract(args)` 结果。
- `certify_checkpoint` 的 stage2 分支仍要求 `terrain == "mixed"`，需要为速度层定义独立门禁协议。
