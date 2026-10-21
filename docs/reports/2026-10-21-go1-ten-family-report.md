# Go1 十族跑酷策略：当前效果说明（2026-10-21）

本文件描述当前交付物的**实际能力、视频对应表现、评测数据、以及 combined（混合地形）未达标的原因与证据**。

- 策略：`checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt`
- 架构：1073 维观测（本体感知 + 48×64 深度扫描特征 + 指令）、12 维关节目标、GRU(128)、`map_attn_v3`
- 训练：Isaac Lab，512 并行环境，PPO；十族混合课程 + 三次精修（v20→v26）
- 评测：确定性 argmax，`seed=2026`，指令带 0.35–0.8 m/s，单回合 133 s

## 1. 十族能力（单族门禁）

| 地形族 | 完成率 | 说明 |
|---|---|---|
| `stairs_up_long` 连续上长楼梯 | 通过 | 稳定上下，无停顿 |
| `stairs_down_long` 连续下长楼梯 | 通过 | 步态收敛，无跌落 |
| `stairs_mixed` 上下混合楼梯 | 通过 | 反向切换时步频自动调整 |
| `stairs_narrow_8_mixed` 窄踏面楼梯 | 通过 | 踏面 0.35 m，逐级过渡 |
| `slope_up` 上坡 | 通过 | 坡度上限内稳定 |
| `slope_down` 下坡 | 通过 | 无前倾失控 |
| `slope_mixed` 上下坡交替 | 通过 | 过渡处不减速 |
| `tire_field` 连续轮胎阵 | **0.907** | 十族中最弱项，偶发踩轮胎边缘打滑 |
| `gaps_short` 短沟（可步越） | **1.000** | 步越节奏稳定 |
| `gaps_long` 长沟（需起跳） | **0.935** | 起跳—落地衔接稳定 |
| `combined` 混合障碍 | **0.000** | 见第 3 节 |

十族同门（10/10）历史最优：**98.61%**（tire 0.907 为短板）。

## 2. 11 族同门 512 槽验收（含 combined）

一次性 512 环境、11 族混合、确定性评测（`reports/v49_11fam.exit`）：

```
overall completion = 0.799
stairs/slopes (8 families) = 1.000
gaps_short = 1.000
gaps_long  = 0.960
tire_field = 0.854   (该抽样带下)
combined   = 0.000
```

结论：**十族可交付；combined 未达标**。

## 3. combined 为什么没做到（证据链）

受控实验（同一评测器、同一权重 v26，128 槽确定性）：

| 场景 | 完成率 | 摔倒率 |
|---|---|---|
| `course`：同一条混合路线，**零沟** | **1.000** | 0.0% |
| `gaps_short_dense`：纯窄沟、**间距 0.65 m（与入口同密度）** | **1.000** | 0.0% |
| `gap_course`：纯窄沟路线 | **1.000** | 0.0% |
| `combined`：混合路线 + 入口三沟 | **0.000** | ~100% |
| `combined` + 只留 **1 条** 11 cm 浅沟 | **0.000** | ~100% |

即：**混合地形本身能走，纯沟本身能走，二者组合即全灭**；问题不是沟宽/深/间距（已分别扫过：宽 ×0.25/×0.6、0.29–0.42 宽沟、深 0.115 浅沟、虚拟支撑面、间距 0.65/0.80/1.10），而是**"沟出现在混合场景"这一感知—行为组合**。

训练侧尝试（真实 combined 门禁）：

| 方案 | 轮数 | combined 门 | 摔倒率 |
|---|---|---|---|
| 纯 PPO（v30b） | 900 | 0 | 83% |
| 入口几何软化（v32a/v33a） | 400 | 0 | ~100% |
| 软终止 → 硬微调（v36a→v36c） | 1200 | 0 | 78.7% → 72.8% |
| 软终止 + 窄沟 credit（v38/v46） | 2800 | 0 | 72.7% / 73.6% |
| 特权观测教师（v47） | 800 | 0 | 81.1% |
| 虚拟支撑面（v48） | 800 | 0 | 97.2% |
| 单沟组合课程（v51） | 500 | 0（自身地形也 0） | 100% |
| 阶梯混合课程（v53） | 1000 | 0 | 76.7% |

与 P10 论文的差距：论文的两个核心机制我们只做了近似或未做——
1. **穿透式软动力学**（允许穿透 + 穿透体积/深度 × 前进速度惩罚 + 自动难度课程）；
2. **特权教师**（距离/高度/宽度 + 障碍类型 one-hot）——我们代码里虽有 `teacher_locomotion_obs`（精确地形几何），但直接给 actor 使用仍无效（v47）。

文献侧同类结论：Acrobotics(2025) 明确指出 Extreme Parkour 的策略"单个障碍可低概率完成，连续 5 个障碍 30 次尝试零成功"；Setup Policies(2021) 用"衔接策略"把随机序列成功率从 1.9% 提到 71.2%；ANYmal Parkour(Science Robotics 2024) 用"单障碍技能 + 高层选择器"处理组合。

## 4. 视频对照（`docs/media/`）

当前 v26 权重、单环境、25 s、指令 0.35–0.8 m/s 录制的实际表现：

| 视频 | 应看到的现象 |
|---|---|
| `v26_stairs_up_long.mp4` | 匀速连续上 8 级台阶，躯干保持水平，无碰撞停顿 |
| `v26_stairs_down_long.mp4` | 连续下行，落步干净，无滑落 |
| `v26_stairs_mixed.mp4` | 上→下切换处步频自动加快，无停顿 |
| `v26_stairs_narrow_8_mixed.mp4` | 0.35 m 窄踏面精确落足，边缘不踩空 |
| `v26_slope_up.mp4` | 上坡增速，姿态角跟随坡度 |
| `v26_slope_down.mp4` | 下坡减速，无前倾 |
| `v26_slope_mixed.mp4` | 坡向交替处无跳变 |
| `v26_tire_field.mp4` | 连续踩轮胎顶通过；偶发 1–2 次打滑（对应 0.907） |
| `v26_gaps_short.mp4` | 小步越过 0.08–0.24 m 短沟，不减速 |
| `v26_gaps_long.mp4` | 起跳越过 0.29–0.42 m 长沟，落地后立即恢复步态 |
| `v26_combined.mp4` | **失败样例**：进入入口第 2 条窄沟后绊倒（对应 0%） |

## 5. 复现

```bash
# 十族 / 11 族确定性验收（远端）
cd /config/Desktop/sim
GO1_TASK_PLAN=$PWD/reports/lp_gate_plan_2026-10-18_go1-v28-eleven-mixture-s01-p1-d1.json \
./sim.sh evaluate --stage stage1 --terrain lp_acrl --headless \
  --num-envs 512 --episodes 512 --episode-seconds 133 \
  --speed-min 0.35 --speed-max 0.8 --seed 2026 \
  --checkpoint checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt \
  --episode-trace reports/v49_11fam_trace.jsonl --run-name v49-v26-11family

# 单族录像
bash scripts/v54_videos.sh
```

## 6. 结论与后续

- **可交付**：十族能力（stairs/slopes 全通、gaps 0.94–1.00、tire 0.907）。
- **未达标**：`combined`；已定位为"组合泛化"而非几何参数，并给出完整消融证据。
- **后续（按文献）**：
  1. `scandots` 局部高度采样 + 特权教师（Humanoid Parkour 2024 / Extreme Parkour 2023 路线）；
  2. 分层选择器（ANYmal Parkour 2024）或衔接策略（Setup Policies 2021）；
  3. 穿透式软动力学（P10 核心机制）。
