# Go1 混合赛道前段稳定落脚奖励设计（v1）

## 背景与证据

使用 s22 中途 `model_550.pt` 的 133 秒混合赛道录像有 15 个 `FALL_EVENT`：11 个在短沟/落台区段 `course_x=1.87–3.29 m`，躯干 z≈0.177–0.180 m；3 个在第一片轮胎 `x=3.96–4.16 m`、1 个在轮胎→上坡接缝 `x=4.89 m`，后四个由过度倾斜触发。低难度的前三个沟仅 0.0816–0.0821 m，按既有几何不需要跳。长沟和末段窄楼梯尚未到达。原 `base_height` 已在第一个低躯干事件前给出负分（采样步 150：未乘权重 `-6.721231`），不能假定单纯提高跌倒/高度罚分有效；原 `gap_intrusion` 在该次逐 25 步采样中均为 0。足部沟跌分类 `min(foot_z)-max(foot_z)>0.30` 的第一分支恒不成立，因此零 `GAP_FALL` 不能证明未踩沟。

## 不变约束

- 完整 11 族赛道、短沟/长沟/轮胎/末段窄楼梯布局不变；速度指令范围 **0.35–0.8 m/s**、观测/动作/神经网络参数形状不变。
- 保存旧状态 `lp_full_roster_state.json` 和 `s22/model_550.pt`，新奖励/新运行 ID/新课程状态隔离；奖励签名改变时只能 `--initialize-from`，不能 `--resume` 旧 reward_config。
- 两张 A10、`/dev/shm` 64 MB：只允许一套 Isaac 仿真；旧 watchdog 的 `lp_phased_stop.flag` 不删除。
- 验收只用**同一次完整确定性闸门**的 11 族逐族完成率≥90%，单实例视频或训练汇总不代替。

## 方案对比与选择

1. **全局提高 `base_height`、`fallen`、`upright` 权重（不采用）**：已有足够强的高度负信号，放大可能惩罚正常楼梯下行且只在即将跌倒时起作用。
2. **立即接入 jump checkpoint（不作为 v1）**：当前短沟仅约 0.08 m、应步行越过，长沟尚未在混合视频出现；现有 jump actor 还会在接入时出现已记录的 32/3 张量维度冲突，贸然切换无法解释早期失稳。
3. **推荐：新版本的“前段稳定落脚”有界奖励**：只在 11 族计划中 `combined` 的开头短沟→首片轮胎区间、且真实向前并发生落脚时，支付高度/姿态共同达标的少量正奖励；旧合同与旧 TOML 不变。先用一个 900 轮训练阶段及确定性闸门检验，不提前承诺有效。

## 新奖励项：mixed_transition_balance

在新的 `configs/rewards_stage1_gaps_v6_mixed_balance.toml` 里保留 v5 所有现有权重不变，额外加入可选 `[mixed_transition_balance]`：

```toml
[mixed_transition_balance]
weight = 1.5
start_offset = 1.20
end_offset = 5.60
minimum_height = 0.23
target_height = 0.31
minimum_upright = 0.55
target_upright = 0.90
```

一个新纯张量函数 `mixed_transition_balance_score(root_z, body_height, gravity_z, forward_progress, touchdown_events, combined_mask, course_x, cfg) -> (num_envs,)` 计算：

- 仅 `combined_mask` 且 `start_offset <= course_x <= end_offset` 且任意足从离地变为着地才支付。`combined_mask` 来自现有 11 族计划；没有计划时为 false。这个窗口覆盖不同难度下的前三条短沟、第一片轮胎与接缝，不在后段阶梯/长沟奖励弹跳。
- `safe_height = min(root_z, body_height)`；高度分 `clip((safe_height-0.23)/(0.31-0.23),0,1)`，所以落入沟内即使支撑面低也不会得到高分。姿态分 `clip(((-gravity_z)-0.55)/(0.90-0.55),0,1)`；速度门为该步原有 `forward_progress` 截断到 `[0,1]`，禁止原地踩踏刷分。
- 三分相乘并乘落脚/路段门，结果严格在 `[0,1]`；每次真实落脚最大奖励 1.5，无阶梯全局罚分或基于失败后重置的巨大负分。复用已有 `touchdown_events`、`body_height`、`forward_progress`，不增添 actor observation，不改变 checkpoint 接口。
- 用 `REWARD_FIELDS` 的**可选扩展**对新表严格校验；旧 50 项 TOML 原样返回 50 项、metadata 原样保留。新配置额外有第 51 项；`RewardsCfg` 注册新 term，缺省 weight=0，只有 v6 运行启用。其他地形该项逐环境严格为 0。

## 诊断与风险

- 不在 v1 改 `foot_spread` 终止规则：仅反转符号会把正常楼梯高差误判为沟跌；需先有接触/局部沟几何的针对性测试。单实例检查时开启 `GO1_DEBUG_FALL=1`，保留原始落点事件；新 run 的报告不得把 `mean_failure_x=0` 当真实坐标。
- 负奖励及全局步态项不重新调参，避免无法分辨收益来源。先比较新运行首闸门与 s21 历史，重点观察 `combined`、短沟、长沟、轮胎；若首闸门仍 hold，基于新录像和原始落点再做下一轮设计。
- LP-ACRL 的 long-gap 跳跃链路当前因 `hier_terrain="course"` 和未传 `--jump-checkpoint` 而失效，这是第二阶段风险，不把该问题当本次短沟跌倒的既定原因。

## 测试、部署、启动及回退

1. 先写纯张量单元测试（全零门、正确形状/范围、低躯干、倾斜、无落脚、无进展、混合/非混合掩码），可选奖励配置兼容测试（旧50项不变，新51项严格约束，坏字段拒绝），阶段驱动无确定性报告不得使用训练汇总升档的测试；观察红后才写实现。
2. 远端源文件事先按 SHA-256 留 byte-for-byte 备份，补最小实现；跑全量 CPU 单元测试和一环境 Isaac 奖励形状/策略载入的串行 smoke，再次确认无 Isaac 进程。
3. 新 `configs/curriculum_phases_mixed_balance_v1.json` 从 full 11 族配置复制，仅设置 `state=lp_mixed_balance_v1_state.json`、`sampler_state=lp_mixed_balance_v1_acrl_state.json`、`results_file=lp_mixed_balance_v1_results.txt`、`init_checkpoint=checkpoints/stage1/2026-10-12_go1-full-roster11-s22-p1-d1/model_550.pt`、`max_stages=1`、`require_deterministic_gate=true`；原驱动未指定 `results_file` 时沿用 `lp_phased_results.txt`；速度/难度/512环境/133秒/逐族90%闸门原样保留。阶段驱动仅在新 flag 下若 deterministic gate 丢失则**停止且不写升档结果**，旧配置行为不变。
4. `GO1_REWARD_CONFIG_PATH` 指向 v6，显式 `--base-id 2026-10-13_go1-mixed-transition-v1 --config ... --max-stages 1`，后台启动单一驱动：从旧 s22 模型**热启动且重置 optimizer**，训练满 900 轮后串行跑原 512 环境确定性闸门。首阶段之后停止驱动，由报告判断下一阶段而不无界自动循环；旧 watchdog STOP flag 保留。
5. 若启动或测试失败，不启动另一套仿真；保留日志和新状态，按备份与 SHA 恢复源文件。没有 Git 仓库，不宣称已提交。所有旧 checkpoint 和原始奖励 TOML 不覆盖。

## 决策授权

用户明确选择 B（测试通过后自动启动新训练），并要求不要再询问；在上述保持赛道/观测/速度/旧状态不变的范围内实施。跨越本设计的新跳跃策略、赛道更改或后续无界训练不在本次授权内。
