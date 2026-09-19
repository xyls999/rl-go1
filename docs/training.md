# 训练与评测手册

本文件记录**可复制的命令**与**必须遵守的坑**。所有命令以仓库根目录为工作目录。

## 0. 环境自检

```bash
./sim.sh test                       # 98 项单元测试：契约 / 几何 / 奖励 / 门禁 / 模型 / 视频
./sim.sh check --stage stage1 --terrain course --headless --num-envs 4 --device cuda:0
```

`check` 会打印观测与奖励维度，并跑一次真实 physics step；**它不会写 checkpoint**。

## 1. 训练

### 1.1 第一层（运控，课程化）

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

```bash
./scripts/torchrun_sim.sh train --stage stage2 --terrain course --headless \
  --num-envs 256 --iterations 1500 --episode-seconds 120 \
  --course-difficulty-min 0.20 --course-difficulty-max 1.00 \
  --locomotion-checkpoint checkpoints/stage1/<a9-validated>.pt \
  --run-name <nav-run-id>
```

`--locomotion-checkpoint` 以 `require_validated=True` 加载：**只接受带门禁记录的 checkpoint**。

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

`passed` 字段只对 `--terrain mixed` 有意义，**不要**用它判断课程地形结果（见已知不一致第 3 条）。

## 3. 冻结门（16 格）与认证

```bash
# 一个格子一次调用；可断点续跑（每次只打印/追加一行）
./scripts/eval_gate_stage1.sh <ckpt> 256 0.3,0.5,0.8,1.0 0.25,0.5,0.75,1.0 cuda:1 150 256 1.0
```

参数顺序：`<checkpoint> <episodes> <speeds-csv> <difficulties-csv> [device] [episode-seconds] [env-count] [max-speed]`。

- 每格 `half = speed-0.1`、`upper = min(max_speed, speed+0.1)`，因此速度带是**以格点为中心的 ±0.1 区间**。
- 验收：`completion ≥ 0.90 ∧ collision ≤ 0.10 ∧ fallen ≤ 0.05`。
- 跑完整矩阵后会写 `reports/gate_stage1_<ts>.json`，再执行脚本末尾打印的 `certify_checkpoint` 即得 `*.validated.pt`。

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
   当前配置：≈21 k 面/tile → 256 环境 ≈5.4 M 面。**加长赛道后必须重新检查面数**。
3. **`--num-envs 512` 不可用**（会触发上述 cooking 失败）；本项目所有报数都用 256。
4. **速度上限**：`contracts.MAX_FORWARD_COMMAND = 1.0`，同时约束 `CommandRouter` 限幅与
   `envs/config.py` 的 `speed_max` 校验。传更大的速度会在启动时报错（不会静默截断）。
5. **不要再改 `normalize_command` 的 `[0.8, 0.3, 1.0]` 与 manifest 的 `goal_xy = 10.0`**：
   它们是已发布 checkpoint 的归一化契约，改动会使两代模型失效（1.0 m/s 会被归一化为 1.25，
   属于已知且可接受的代价）。
6. **回合时长**：赛道加长到 30 m 后，门禁/评测的 `episode-seconds` 必须 ≥150 s，
   否则"未完成"会被记成超时。
7. **热启动不要沿用旧优化器**：`--initialize-from` 已丢弃优化器与噪声；
   实测若用高学习率/高噪声直接续训，几轮内策略就会坍缩（奖励上升但完成率归零）。
8. **`reports/`、`logs/`、`checkpoints/` 不入库**（体量大且与机器绑定）。要复现结果请按
   第 6 节的赛道版本与命令重跑。

## 6. 已知代码不一致（待修）

- `cli.py::_ppo_config` 依据**原始 CLI 参数** `args.reward_contract` 选择策略类
  （`HierActorCriticV2` vs `HierActorCritic`），而环境侧 `_resolve_contract` 会按 stage 推导契约
  （stage2 → `nav_v2`）。因此**未显式传 `--reward-contract` 时，策略类与奖励契约可能不同步**：
  已发布的 checkpoint 全部是 `ActorCore/CriticCore`（`hier_v2` 的 L2Norm 编码器变体未被使用）。
  修复方向：让 `_ppo_config` 使用与 `make_env` 相同的 `_resolve_contract(args)` 结果。
- `certify_checkpoint` 的 stage2 分支仍要求 `terrain == "mixed"`，需要为速度层定义独立门禁协议。
