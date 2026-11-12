# Go1 V1333 长沟严格完成率：证据审计与改进研究

日期：2026-11-12。范围：只读审阅 A04/A03 训练证据、A04 源码与一手研究；未修改代码/配置，未启动训练、诊断、评估、holdout 或 gate。

## 结论先行

A04 有初步进展，但完成率仍低：按项目口径，训练 rollout 中 **693/47,695 = 1.453%** 到达终点且未摔倒、未掉沟；**1,222/47,695 = 2.562%** 无摔倒且未掉沟。报告汇总的 `fallen` 是 40,343、`fell_into_gap` 是 7,905、`collision` 是 23,378；这些失败标签可能重叠，不应相加。no-fall/no-gap 的补集为 46,473 个 episode，即约 97.4% 至少有摔倒或掉沟。可见的首要问题不是再把 airtime 权重调大，而是：

1. **优化奖励与项目定义的完成指标不一致。** 严格完成指标只在 `get_metrics` 中计算；实际 `goal_reached` 奖励使用更严格的旧 `_success` 条件。A04 的 693 次严格完成全部对应 `task_finish_reached`，但旧 `success` 仍为 0，说明该训练没有观察到旧终点奖励。
2. **摔倒/掉沟远比安全完成常见。** 单纯奖励空中前进没有直接解决稳定起跳、远岸落脚和落地后继续行走。
3. **新奖励信号很弱，且奖励并不等于完整、安全跨越。** `airborne_forward_progress` 在每轮日志样本均值为 0.027728、末值 0.0898、最大 0.1097。helper 只按特定窗口中的四足腾空正向距离记账；普通的近岸安全着陆不会触发其失败退款条件，因此存在“获得部分空中前进信用、但未确认清沟”的奖励错配风险，尚无 trace 证明策略实际利用了该路径。
4. **不要先盲目提速。** A04 的全 episode 平均命令速度/实现速度为 0.441/0.528 m/s；A03 为 0.443/0.522 m/s。它们不是起跳瞬间速度，因此既不能排除速度问题，也不足以支持全局提速。

建议顺序：先对齐目标和观测，再做阶段化、变量受控的跨沟实验；所有效果以冻结策略的独立评估为准。

## 项目证据

| 量 | A03 | A04 |
|---|---:|---:|
| 训练迭代 / 环境 | 300 / 512 | 900 / 512 |
| 训练 episode | 32,558 | 47,695 |
| 严格完成 | 0 (0%) | 693 (1.453%) |
| no-fall/no-gap | 524 (1.609%) | 1,222 (2.562%) |
| 旧 `success` / `completion_rate` | 0 | 0 |
| 训练均值命令/实现速度 | 0.443/0.522 m/s | 0.441/0.528 m/s |

A04 的 `task_finish_reached=693` 与 `task_completed=693` 相等；另有 529 个 episode 虽然无摔倒、未掉沟，但没有到达终点。环境槽的成功计数分布在 356/512 个槽位，不是单个槽位偶然贡献。它们仍是同一训练 rollout 统计，不是 47,695 个独立泛化样本。

A04 从 A03 `model_299.pt` 仅加载 policy/value 并重置 optimizer；相比 A03 又训练了三倍迭代。因此 A03→A04 的变化同时混合了训练预算、续训和奖励改动，不能当作 A04 新奖励的因果效果。A04 状态为 `trained-unverified`；没有独立评估、holdout 或正式 gate。

**证据文件**（远端 `/config/Desktop/sim`）：
- A04 receipt：`v128_work/v132_randomized/v1333/reports/long_gap_warmstart_result_a04.json`；其中 `status=trained-unverified`、`task_completion_rate_training_rollouts=0.014529824929237866`、`no_fall_no_gap_rate_training_rollouts=0.025621134290806164`、`capability_validated=false`。
- A04 stage report：`reports/stage_go1-v1333-longgap-air-distance-warmstart900-a04.json`。其中 `task_finish_reached=693`、`task_completed=693`、`fallen=40343`、`fell_into_gap=7905`、`collision=23378`、`out_of_bounds=1211`。episode 标记可能重叠；out-of-bounds 是另行指标，不纳入既定完成定义。
- A04 训练日志：`v128_work/v132_randomized/v1333/reports/go1-v1333-longgap-air-distance-warmstart900-a04_full_train.log`；900 个 `Episode_Reward/airborne_forward_progress` 更新样本均值 0.027728、末值 0.0898、最大值 0.1097。这是训练奖励信号，不是清沟率。
- A03 stage report：`reports/stage_go1-v1333-longgap-from-best-entrance-warmstart300-a03.json`。

## 源码审计：最值得先处理的两处

### 1. 训练奖赏没有直接对齐严格完成定义

- `envs/long_gap_completion_v1333.py:21` 定义 `task_completion_mask = finish_reached and not fallen and not fell_into_gap`。
- `envs/runtime.py:1758-1762` 把终点矩形进入状态锁存为 `_task_finish_reached`；`2851-2853` 只把严格完成标记放进 metrics。
- 同一文件 `1804-1815` 的旧 `_success` 另要求 gap 专项路线完成、终点后稳定行走等条件，并把 out-of-bounds 作为 hard failure；`2376` 的 `goal_reached` 奖励实际取 `_success`。配置中该项权重为 200，但 A04 旧 `success=0`，所以没有任何训练 episode 触发这项旧终点奖励。

这是当前最清晰的目标错配：既定 KPI 只看终点区域+不摔倒/不掉沟，而训练终点奖赏使用了更严的旧成功门。**实验建议**：严格保留两个指标，另做隔离奖励版本，只在严格完成条件首次成立时支付一次正奖励；不要改写/混淆旧 `success`，也不要擅自把 out-of-bounds 加进完成定义。奖励量级应依据现有 episode return 做消融，而不是直接猜一个大权重。

### 2. A04 空中距离信用不是安全着陆信用

- `envs/airborne_forward_progress_v1333.py:85-122` 要求长沟窗口、`local_x` 从近沿前 0.35 m 到远沿、四足同时腾空、正向移动、无侵入和 hard failure；每沟最多 `min(0.45 m, gap_width)`。
- `125-139` 仅在侵入或 hard failure 时对未确认信用退款。helper 没有“普通触地且尚未越过远沿”的退款条件；这不证明训练中出现了奖励投机，但说明单元测试覆盖不等于行为信用一定与清沟一致。
- `runtime.py:2585-2607` 接入该项；A04 配置权重 20、上限 0.45 m。奖励窗口的距离条件又在 helper 中截到远沿；落地虽有独立 `landing_stability` shaping，但前进距离奖励本身不要求远岸落地。

**实验建议**：先补齐逐 gap 的“触发—空中前进—是否跨过远沿—首触地点—稳定落地—退款/确认”统计。再比较当前奖励与“信用 provisional、只有远岸落地/清沟后确认，未清沟落地时撤销”的隔离变体；同时跟踪安全完成率，避免只提高空中移动或跳跃次数。

## 优先级实验路线

### P0：只读可证伪诊断（改奖励前）

以 A04 现有 stage report 可确认：大量 episode 有跌倒/掉沟，但汇总不能告诉我们它们发生在起跳前、腾空中、落地后还是后续沟段。未来获准的记录/回放应按 episode 与 gap 输出：沟宽/起跳距离、近沿命令与实际速度、空中距离/时长、远沿是否越过、首次触地位置、着陆姿态/角速度/足接触、碰撞/掉沟/出界、严格完成、各奖励项和 credit refund。先确保各标签与终止原因能闭合，不用挑一段诊断视频替代总体统计。

### P1：把奖励目标对齐，并隔离测试 A04 新项

从同一 A03 `model_299.pt` 和同一训练预算出发，至少对照：旧 A03 奖励；A04 原始空中距离信用；严格完成一次性奖励；严格完成奖励 + 远岸着陆确认的阶段 shaping。训练 seed/场景/更新预算尽量配对，多 seed 报告区间。该实验能区分“多训练 900 轮”与“奖励改动”的作用；A03 与 A04 现有结果不能。

若新增 shaped reward，考虑 potential-based 形式 `F(s,s')=γΦ(s')−Φ(s)`，并正确处理折扣、终止态与奖励缩放；任意“距离越来越近就给分”的 shaping 不享有策略不变性保证，可能改变策略目标。

### P2：从一个可控跳跃技能逐步过渡到连续全路线

把单沟完成分解成可训练目标：稳态助跑/起跳窗口、越过远沿、远岸指定落点、着陆后稳定，再逐步连接多沟完整路线。固定场景下先单独扫描 gap width 和 takeoff distance，再渐进引入其它随机项；课程晋级条件使用严格成功与安全指标，不要只按前进距离或跳得更远晋级。

这一方向有一手先例：Atanassov 等的四足跳跃工作从原地竖跳发展到前向/斜向跳，再到障碍跳，并以目标落地点、姿态和障碍尺寸为条件。作者官方 repo README 也把 Go1 向上跳与前向跳分成两阶段训练。其机器人、仿真、奖励和训练时长与本项目不同，**可借鉴课程结构，不应照抄其成功率/轮数**。

### P3：测量后再决定速度

把速度统计拆到 gap 阶段：入口、起跳、腾空、触地。A03/A04 全 episode 平均命令速度都约 0.44 m/s，未证明起跳速度足够或不足。若阶段数据确认冲量/起跳速度不足，再在相同 gap 几何、奖励和初始化上单独比较现有速度档；同时监控落地、摔倒、碰撞，不能只看 forward progress。

### 验收

后续每个候选先按严格 task completion 评估，另报 no-fall/no-gap、fallen、fell_into_gap、collision、out-of-bounds 和落地稳定；策略冻结后用未参与调参的独立种子/难度 holdout。预先由项目负责人定阈值。训练 rollout、reward mean、legacy success 和单回放都不代替独立评估。

## 一手来源

1. Atanassov et al., [Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design](https://arxiv.org/abs/2401.16337)，arXiv 摘要：明确描述从原地跳到前向/斜向跳，再到障碍跳；条件包括目标落地点、姿态和障碍尺寸。作者代码：[Curriculum-Quadruped-Jumping-DRL README](https://github.com/Vassil17/Curriculum-Quadruped-Jumping-DRL)，展示 Go1 向上跳训练后续训到 forward jump 的阶段安排。
2. Shafiee, Bellegarda & Ijspeert, [Viability Leads to the Emergence of Gait Transitions in Learning Agile Quadrupedal Locomotion on Challenging Terrains](https://www.nature.com/articles/s41467-024-47443-w), Nature Communications 2024：定义 viability 为存在动作可避免跌倒的状态；论文对 64 组 reward 权重做消融，在其特定机器人/terrain 上高 viability、低 CoT/峰值接触力权重组合取得最高 gap success。**不是 V1333 参数建议或可直接迁移的通过率。**
3. Ng, Harada & Russell, [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf), ICML 1999：potential-based shaping 的策略不变性条件；不能将任意正向进度奖励称为无害 shaping。

## 边界

本文是研究建议，不是代码改动或实验授权。A04/A03 训练 rollout 只能说明训练过程中发生了什么；目前没有独立评估/holdout 证明策略能力。下一步若要执行任何代码修改、训练或行为验证，应按既有授权规则另行明确指定范围。
