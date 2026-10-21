# LP-ACRL 分阶段难度课程训练计划 (v1)

## 0. 一句话

在 **8 类地形 × 4 难度档 × 5 速度档 = 160 个任务实例**上训练单个 12 维关节策略：
**外层**按「全体完成率 ≥ 90% 才升难度档，升档后低难度按固定比例穿插防遗忘」推进，
**内层**用 LP-ACRL 进度采样在每个档位内自动分配「哪个族/哪个实例该多练」。

## 1. 事实基础（已核实，非假设）

| 项 | 值 | 来源 |
|---|---|---|
| 地形族（8） | `stairs_up_long, stairs_down_long, stairs_mixed, slope_up, slope_down, slope_mixed, gaps_only, combined_flat` | `envs/task_curriculum.py::TERRAIN_FAMILIES` |
| 难度档 | 4 档，`difficulty_value(level)` 映射到 `[0, difficulty_max]` | `task_curriculum.py` |
| 速度档 | 5 档，`speed_interval(level, 0.35, 0.80)` 子区间 | 同上 |
| Tile / 路线 | `TILE_SIZE 40.0`、`COURSE_FEATURE_END 35.0`、`COURSE_GOAL_OFFSET 34.0` | `envs/geometry.py` |
| 完赛区 | **x 33.1–34.1 m，\|y\| ≤ 4.0**（与 40 m tile 对齐，一族一份） | `configs/rewards_stage1_gaps_v5.toml [goal_reached]` + geometry `finish_start_offset 33.10` |
| 完成率定义 | 阶段内已完成回合中 `success=True` 的比例 | `runtime` `_success` |
| 终止原因（5 类） | `success / fallen / collision / out_of_bounds / fell_into_gap` | `envs/config.py::task_termination`（报告已全部记录） |
| 论文辅助判据 | 存活 > 900 步；EPTE-SP < 30% | 已读论文（arXiv:2601.17428） |
| 阈值协议 | `--num-envs 32 --episodes 64 --seed 2026 --episode-seconds 133` | 既有验收协议 |

## 2. 两级循环

### 外层：难度档门控（用户规则）
- **升档**：本阶段完成率 ≥ **90%**，且样本充足（**≥200 回合**，且每族样本 ≥20 回合）
- **保档**：未达 90% → 留在当前档继续（内层 LP 会自动把采样集中到未达标的族）
- **降档保护**：完成率 < 30%，或「擦地+越界+掉沟」合计 > 50% → 退回上一档（避免无梯度空转）
- 每次只推进 **1 档**，改档后必须重跑该档完整阶段再判

### 内层：LP-ACRL 进度采样（已实现）
- `LPSampler`：每实例维护 `reward`（阶段平均回报）与 `lp = 新回报 − 旧回报`（平滑 0.5），
  概率 ∝ `softmax(lp / (β·std(lp)))`，`UNIFORM_FLOOR = 0.25` 保证永不归零
- 状态文件 `lp_acrl_state.json` 跨阶段累积（当前跑的那轮可作为热启动）

## 3. 防遗忘穿插（用户要求，最小改动）

允许集合 = `{目标档 L} ∪ 回放档`，混合权重按阶段自适应：

| 阶段位置 | 目标档 L | L−1 | ≤ L−2 | 说明 |
|---|---|---|---|---|
| 早期（L=1,2） | 0.70 | 0.20 | 0.10 | 低难度占比高，先把基础打稳 |
| 后期（L=3,4） | 0.80 | 0.15 | 0.05 | 聚焦难档，但仍强制回放 |

- 回放集合内部**均匀分配**（不参与 LP softmax），确保低难度每个族都持续被复习
- 硬约束：**回放总比例 ≥ 0.15**（低难度永不归零）

## 4. 阶段表

| 阶段 | 允许难度档 | 目标档 | 速度档 | 迭代 | 升档门槛 | 目的 |
|---|---|---|---|---|---|---|
| P0 | — | — | 低 | 已在 b1a3 完成 | 平地/混合赛道 ≥85.9% 已过 | 起始权重 |
| P1 | d0–d1 | d1 | 0–1 | 250 | 完成率 ≥90% | 楼梯族从 95 步提上来；定位擦地/越界死因 |
| P2 | d0–d2 | d2 | 0–2 | 250 | ≥90% | 中等楼梯/坡；gaps_only 保持回放低位 |
| P3 | d0–d3 | d3 | 0–2 | 300 | ≥90% | 全难度地形 |
| P4 | d0–d3 | d3 | 0–4 | 300 | ≥90% | 提速（验证速度绑定：报告附 cmd vs achieved） |
| P5 | d0–d3 | d3 | 0–4 | 300 | 完成率 ≥90% + 存活 >900 步占比达标 | 收敛 + 稳定（可降温 β） |
| P6 | d0–d3 | d3 | 0–4 | — | 64 回合 seed 2026 ≥90%，各族 ≥80% | 正式验收 + 录视频 |

## 5. 代码落地清单

### 已完成（可直接用）
- 8 族几何 + `lp_acrl` 模式 + 逐环境任务表（`build_scene(tasks=)`）
- `LPSampler`（含 `TERRAIN_FAMILIES`、`difficulty_value`、`speed_interval`、状态持久化）
- 逐环境回合回报记录 → `reports/stage_<run>.json`（**含 5 类终止原因**）
- 逐环境速度档命令采样（`FixedWorldVelocityCommand._resample_command`）
- 驱动 `scripts/lp_acrl_curriculum.sh` + 助手 `scripts/lp_acrl_stage.py`（plan / update / show）
- 论文 sim 优化：质量/质心/恢复系数/推挤/关节复位 + 电机模型 ±10% + 传感器延迟/噪声/偏置/dropout

### 待实现（工作量小，全部是纯逻辑 + 报告）
1. `BandCurriculum`（`task_curriculum.py`）：输入 `(目标档, 回放权重表, 允许集合)` → 输出每实例权重，
   与 `LPSampler` 概率相乘后归一化。
2. `lp_acrl_stage.py` 增加 `gate` 子命令：读 `stage_*.json` → 输出 `advance / hold / regress` +
   完成率、样本量、各族完成率、终止原因构成。
3. 驱动改读 `configs/curriculum_phases.json`（第 4 节表格的机器可读版），按 gate 结果自动切档。
4. 报告新增字段：`completion_rate`、`survival_rate`（steps ≥900）、`mean_cmd_speed` vs `mean_achieved_speed`。
5. （第二版，可选）EPTE-SP：逐回合统计「速度跟踪误差 > 容差」的步数占比。

### 命令（容器内）
```bash
# 单阶段手动
python scripts/lp_acrl_stage.py plan  --envs 256 --state lp_acrl_state.json --out reports/plan_s1.json \
    --difficulty-max 0.6 --speed-min 0.35 --speed-max 0.80 --target-level 1
GO1_RUN_ID=<run> GO1_TASK_PLAN=reports/plan_s1.json ./scripts/torchrun_sim.sh train ... 
python scripts/lp_acrl_stage.py gate   --plan reports/plan_s1.json --report reports/stage_<run>.json
python scripts/lp_acrl_stage.py update --state lp_acrl_state.json --plan reports/plan_s1.json --report reports/stage_<run>.json
# 全自动
GO1_LP_PHASES=configs/curriculum_phases.json bash scripts/lp_acrl_curriculum.sh
```

## 6. 已知瓶颈与针对处置（来自实测 Stage 1）

| 现象 | 数据 | 处置 |
|---|---|---|
| 楼梯族最快终止 | `stairs_mixed 94.7 步`、`stairs_up_long 99.2`、`stairs_down_long 110.2` | P1 目标档 d1 优先练楼梯；先看报告确认是「擦地(collision)」还是「越界」 |
| 坡道存活最长 | `slope_down 431 步`、`slope_mixed 222`、`slope_up 172` | 说明坡道不是瓶颈；不需额外加权 |
| gaps_only 回报最差 | 平均回报 `−64.5` | 早期放回放集合最低权重（避免主导），d≥2 再纳入目标档 |
| `fallen = 0` / `success = 0` | 24,347 回合全为 0 | 说明是碰撞/越界/掉沟终止 —— 已补记录，P1 报告即见分晓 |
| 速度绑定未验证 | 历史：策略无视指令（实测 0.958 vs 指令 0.482） | P4 提速时用报告里的 cmd vs achieved 直接判定 |
| 平均回合长度 174 步 | 约 3.5 秒 | 完成率门槛天然惩罚「活不长」，无需额外机制 |

## 7. 时间与资源

- 每阶段：250 迭代 × 256 envs ≈ **19 分钟**、≈ **2.4 万回合**（每 env 约 95 回合 → 完成率估计足够稳）
- 6 个阶段 + 验收 ≈ **2–2.5 小时**
- 硬件：2×A10，`torchrun --nproc_per_node=2`；`/dev/shm` 仅 64 MB → **绝不并发第二个训练/评测进程**

## 8. 当前运行如何处置

正在跑的是「纯 LP-ACRL、难度 0–0.6 平铺、无门控」的对照轮（Stage 2/10）。
- 其 `lp_acrl_state.json` 已积累每实例首轮回报 = 内层采样器的**热启动**，建议保留复用
- 建议：**Stage 2 跑完后停止**，按本计划从 P1 重启（复用 state），这样外层门控 + 防遗忘才真正生效

## 9. 验收标准（最终）

1. 最高档阶段（采样集中在 d2–d3）：**64 回合 seed 2026 完成率 ≥ 90%**（用户门槛）
2. 各族完成率 ≥ 80%（防止“只会在坡上跑”）
3. 论文辅助判据：存活 > 900 步的回合占比显著（目标 ≥ 70%）
4. 视频交付：单狗跟随视角，8 族各一段，含上下坡、上下楼梯、连续沟
