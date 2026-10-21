# v19 纯长沟上抬准备奖励：已批准的有界实验（进行中）

用户授权改代码继续训练。只更换v18失败的训练侧预起跳信号，默认关闭`GO1_V19_PRELAUNCH_REWARD=1`；不更改actor1073输入、12动作、GRU128、0.35–0.8目标、硬碰撞、旧清沟/稳落判定、原11族闸门。四足净力仅为支撑候选，不是精确落点，任何训练路线成绩不能证明真飞越、沟壁无擦碰或实机安全。外部一手文献调研子代理因缺网页工具未核验来源，以下方案仅以本仓库代码、冻结报告和真实PhysX回放为证据。

## 原因与改动

- v16权重单族原512槽门439/578（75.95%，HOLD），低速偏弱。逐步证据：失败env7在x≈0.70已见沟并打开窗口，x≈0.96已超速且vz为负；v18硬阈值vz≥0.30无信号，其单次`takeoff_alignment`在配置weight2、控制步长0.02s下最多给0.04回报。沟沿进入罚项一次性可达weight−80×40×0.02=−64量级，奖励差异严重。
- 隔离`v19_work/source/go1_hier_sim/go1_hier_sim/envs/prelaunch_v19.py`引入`PrelaunchGradientState`：纯长沟、已开启可观测扫描窗口、距离近沿0.55–0.15m、实际前进≥0.30m/s、命令≥0.35m/s、躯干z≥0.29m、直立≥0.90、两三足净力支撑且未倒/碰时，计算`clamp((vz+0.20)/0.60,0,1)`；**只给每段沟“历史最高分的正增量”**，累计最多1单位，原地/倒地/离窗或同段重复刷分无收益。actor看不到训练侧沟段元数据与接触候选。`runtime.py`将正增量按25倍加到旧`takeoff_alignment`，当前weight2、步长0.02s下每段累计回报≤1；`--episode-trace`只在启v19时追加`prelaunch_credited_gaps`和`prelaunch_credit_units`。checkpoint metadata标`reward_variant=prelaunch_v19,capability_validated=False`。旧v18规则保留但不可与v19同时开启。
- TDD：14个新纯张量测试先RED（未实现功能），后14 passed；原8槽真实评测启v19之前先RED（`reward integration missing`），部署之后9回合GREEN，失败env7首回得0.264单位；成功/失败、course_x逐项保持。远端`./sim.sh test -q` **318 passed、76 subtests passed**，日志`v19_work/reports/v19_prelaunch_full_tests.log` SHA `063b40e67a74afe15cd02ee7b4b03721fddcc9f3ba75b4ab0a5e8f3f99ba95da`。
- 冻结v16权重、**原512槽计划**的奖励只读回放：578完整回合，原始成功/失败、course_x、平均指令及实速逐回合完全未变；437/439成功回和104/139失败回得到上抬准备积分，积分总计1681.855，每回最多7.021、最多16段，单段≤1。每回`return`差等于积分，说明计量真实进入奖励而未变更动作。trace`v19_work/reports/v19_prelaunch_probe512.jsonl` SHA `ef52af9f514eadf734bea1e2a145d74085503f637be70b32cd3e0299087d67f8`。**不是新策略的能力提升；更不是跳跃安全。**
- 四卡×16环境×12迭代真实PPO/PhysX烟测退出0，`Episode_Reward/takeoff_alignment`最终打印0.0050–0.0085、`leap_clear`仍0.0000；日志SHA `deed3476a8f319d5ac499715ead716418f62ea113b24c6e52fcf7ade76fb57b7`。烟测checkpoint读回`reward_variant=prelaunch_v19`、`capability_validated=False`。

## 正在运行的短试验（结果未定）

- UTC2026-10-17T13:08:19Z唯一Isaac四rank各128纯`gaps_long`环境、200迭代/前50仅critic、热启冻结v16，Gloo，不启watchdog/自动p2。驱动`v19_work/scripts/start_v19_prelaunch_pilot.sh` SHA `082305863344b31d41581b72acaae71e67df937004dfea3ebf5e01fe397e002c`，远端`reports/v19_prelaunch_pilot4x128_{driver,train,gate}.log`、退出收据`reports/v19_prelaunch_pilot4x128.exit`。训练后同脚本串行用原512槽计划评测新checkpoint并写逐回合trace；这只是pilot门，不能代替十族学生或最终11族门。启动后父代理看到四个sim.py进程与iteration3/50；退出结果需重新核查。原`lp_phased_stop.flag`启动时SHA `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。
- 只有pilot原512槽指标、低速/难度分格、超速与真实飞行/落地候选均可解释，才决定是否投入900轮；若无真实飞行线索或奖励欺骗迹象，仍HOLD并继续触点/行为诊断，绝不直接升档。

## 追加（17:05 UTC）：对照配对、scale=4与900轮长训结果

- **试点(cap=1)**：78.65%（453/576，首槽421/512）、碰撞4。**同参legacy对照**（`start_v19_legacy_control.sh` SHA `fc043d4a1fff9a65ec19294ab6bee6094e65faaa4b797ada57967f9e9ab80757`，train/gate exit0，15:00:57Z，checkpoint SHA `0dac8b15...`，trace SHA `17713e63...`）：453/570、首槽420、碰撞3。**配对差异≈0：cap=1的增益就是多训200轮本身，不构成奖励杠杆；两臂都引入新碰撞。**
- **scale参数TDD**：`PrelaunchGradientState(scale=)`，credit/cap/步内上限均×scale，默认1保持试点行为；RED（TypeError）→16 passed，远端全量 **320 passed/76 subtests** 日志SHA `706a1609afe05d4120457b2b7ebe9652b8b1b59320a1ace0ce3695d1b5ae17e1`；runtime读`GO1_V19_PRELAUNCH_SCALE`，每沟回报≤0.5×scale。部署：prelaunch_v19 `a376ace2...`、runtime `57d56fef...`、cli不变。
- **scale=4试点**（15:27:52Z exit0，checkpoint SHA `c771e64b...`、trace SHA `a5e3ca99...`、报告SHA `d636c5bf...`）：479/567=84.48%、首槽425、**碰撞0**、跌倒88、速度1层93/118 vs 对照75/102；对v16配对31升/2降（对照25升/7降）。同算力净优势且不引入碰撞→授权长训。
- **900轮×4×512纯长沟 v19p4长训**（16:55:58Z train/gate exit0，checkpoint SHA `75968c8885cfc842d1edd4f46ba15f3076d4912d100dd02a51840e99f4adbdc3`，metadata `prelaunch_v19/capability_validated=False`）：族门 **completion 97.087%（≥90%达标）**、碰撞0/512、跌倒15/512（全部首沟前x<2低伏）、trace SHA `abef6826...`、报告SHA `c119a3b9...`；对v16配对101升/15降。第899轮`takeoff_alignment=0.4374`，但`leap_clear`900轮摘要全程0.0000：**无已验证真实腾空证据，达标≠跳跃安全认证**；实速1.055对指令0.486速度追踪不合格、难度仅0/0.1，均记录在案。
- 下一步属人工门范畴（学生十族蒸馏、11族同次512各90%、或难度升档p2），未获自动授权；服务器空闲等待指示。
