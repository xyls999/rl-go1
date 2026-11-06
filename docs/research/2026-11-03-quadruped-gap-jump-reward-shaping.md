# 四足跨沟/跳跃强化学习奖励塑形：一手论文与作者代码对照

> 2026-11-03。直接下载并阅读下列 arXiv PDF；另核对两个作者仓库的 `main` 源码。论文报告与仓库当前默认配置不一定对应同一实验 checkpoint。这里只研究，不改变本项目奖励或训练结果。

## 一手来源怎么做

| 来源 | 可核验的塑形方法 | 与本项目的边界 |
|---|---|---|
| [Cheng et al., *Extreme Parkour with Legged Robots*, arXiv:2309.14341](https://arxiv.org/pdf/2309.14341), §3.2, Eq. 2–3, pp. 5–6；§4.3.2, p. 10；[作者奖励实现](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py#L1226-L1288) 和[配置](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py#L293-L314) | 沿下一 waypoint 方向的世界系速度做**封顶进度** `min(<v,d>,v_cmd)`，不为无限加速多付；在距边缘约 5 cm 的接触脚上施加惩罚 `-Σ_i c_i M[p_i]`（PDF Eq. 3 的 4 是四足求和上限，不是额外乘以 4；作者代码中 `feet_edge=-1` 且仅对较高难度生效）。去掉边缘项的 NoClear 对照会在边缘踩得很近、不稳定甚至坠落。另有姿态、横向角速度、碰撞、脚撞竖面和动作/力矩平滑正则。 | 它的 A1 演示 0.8 m 长跳，不是本项目 0.08–0.24 m 多短沟与 0.35–0.8 m/s 指令；不能移植其系数、成绩。其触地边缘惩罚比单纯增加落沟终止惩罚更直接关联足点安全。 |
| [Atanassov et al., *Curriculum-Based RL for Quadrupedal Jumping: A Reference-free Design*, arXiv:2401.16337](https://arxiv.org/pdf/2401.16337), §IV-C/Table I, pp. 5–6 | **起跳前站立、腾空、落地**三个阶段分别定义奖励；空中跟踪水平/偏航速度和足端收腿净空，落地阶段跟踪目标位置/朝向及低角速度；落点位置/姿态、成功跳跃为单次奖励，连续项按阶段付。正奖励与惩罚不是简单相加，而是 `r_total=r_+ exp(-||r_-||²/σ)`；论文明确说不良缩放易使策略学会站着不跳。 | 这是**给定目标落点与姿态**的 Go1 跳跃任务，命令含目标/障碍尺寸和 jump toggle；本项目 1073 输入及多沟连续路线没有同一显式命令，不能照搬目标落点项或改硬判据。可借鉴“只在正确阶段给相应信号、落地后还需稳定”的设计。 |
| [Rudin et al., *Advanced Skills by Learning Locomotion and Local Navigation End-to-End*, arXiv:2209.12827](https://arxiv.org/pdf/2209.12827), §II-B–C, pp. 2–3 | 用终点位置+**持续时间窗口**奖励，而非全程速度追踪；作者说窗口须够长，才能逼迫机器人在目标附近稳定，否则会在末秒扑向目标、现实中摔倒。辅以关节加速度、力矩、碰撞、动作变化和足加速度惩罚；早期探索方向奖励随后移除，另有远离目标的停滞罚。作者指出过沟时恒定机身速度会限制必要的腾空/减速动作。 | ANYmal 的目标位置/时间任务，不等于本项目指定 0.35–0.8 m/s 速度带；不建议直接移除速度命令。启示是不能只看瞬间越过沟，需检验越沟后持续稳住。 |
| [Zhuang et al., *Robot Parkour Learning*, arXiv:2309.05665](https://arxiv.org/pdf/2309.05665), §3.1, p. 4；[Go1 leap 配置](https://github.com/ZiwenZhuang/parkour/blob/main/legged_gym/legged_gym/envs/go1/go1_leap_config.py#L90-L114) | 文中 `r_skill=r_forward+r_energy+r_alive`，速度目标、横向速度、偏航和机械能共同塑形；穿透罚与逐渐加难的课程帮助先探索，再在**硬物理约束**下微调。Go1 leap 当前源码 `tracking_world_vel=5`、`orientation=-0.1`，没有激活 `leap_bonous_cond`（被注释）。 | 当前配置的 leap 速度约 1.5–1.8 m/s、沟宽 0.3–0.8 m；`virtual_terrain=True` 属于论文软约束阶段，**本项目禁止改沟几何/碰撞/硬验收，不能照搬穿透课程**。它也没有证明“只加大姿态权重就能稳定落地”。 |
| [Yang et al., *Agile Continuous Jumping in Discontinuous Terrains*, arXiv:2409.10923](https://arxiv.org/pdf/2409.10923), §V-B–C, p. 4；§VI, pp. 4–5 | 连续跳跃的难点包括单次跳后落地姿态不准；训练把低层控制器的跟踪误差成本放进策略奖励，按逐级距离课程学习；执行端用 bounding 接触节奏和反馈控制姿态。 | 其低层有模型控制器，不是本项目直接 12 动作 PPO/GRU；可借其“连续跳要求落地后姿态准确”和接触相位诊断，不可声称它证明了本项目某个奖励系数。 |

补充：[PIE, arXiv:2408.13740, Table I, p. 3](https://arxiv.org/pdf/2408.13740) 在 parkour 中主要用线速度跟踪、姿态/横向角速度/碰撞/动作平滑等简单奖励，并**没有**表中专门的逐沟落地事件奖金。这说明跨沟成功未必必须复杂分段奖金，不构成“本项目改奖励必然有效”的证据。

## 对照 V132.2 失败证据（不是论文结论）

- [入口正式验收凭据](../../v128_work/v132_randomized/v1322/reports/v1322_go1-v1322-real-entrance-s01-p1-d1-holdout-seed1812341965-a01.receipt) 与同目录 JSON/JSONL：233/512 首次回合成功；279 失败中 fallen=219、collision=89、fell_into_gap=53（标记重叠），219 个 fallen 中 171 个已获得至少一个真实沟信用；五沟成功 18/112，二沟 105/146。多数失败局限于路线 x=2–6 m。仅凭终态不能确定是刚落地、下一沟起跳还是别的接触先失稳。
- [训练日志](../../v128_work/v132_randomized/v1322/reports/go1-v1322-real-entrance-s01-p1-d1_full_train.log) 最后一轮的已加权 `Episode_Reward`：`supported_short=0.0002`、`safe_segment=0.0014`，对比 `forward_progress=0.4350`、`landing_stability=0.0301`、`speed_error_l1=-0.0450`。这些是日志的归一化统计量，不能仅凭系数比判定因果，但表明新增逐沟信号在总量上很小。
- 现有 [entrance_support_v1322.py](../../v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/entrance_support_v1322.py) 奖励有支撑的接近前进，且水位线防往返刷分；[gap_score_v1321.py](../../v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/gap_score_v1321.py) 只在远岸两足支撑、立姿合格时逐沟记一次信用；[landing_stability_v131.py](../../v128_work/v132_randomized/v1322/source/go1_hier_sim/go1_hier_sim/envs/landing_stability_v131.py) 在普通触地且跳跃窗口内按姿态和角速度给分，**没有**要求该触地发生在一次真实逐沟信用后。runtime.py 有“Landing-region preference”的注释，但在紧随其后的 `terms` 中未见专门的沟沿触地惩罚；因此不能声称已实现 Cheng 的 5 cm edge penalty。

## 待验证的设计假设，不是已证实修复

1. **先诊断阶段与足点：** 只用开发 seed 2028 或其他非最终 seed，逐步记录每条沟的近沿、腾空、前/后足接触、真实 `paid`、随后的 0.3–0.5 s 姿态/横向角速度、实际与指令速度、沟沿接触、终止原因与每项加权奖励。把“信用后摔倒”拆成同一远岸落地失稳与下一沟失败两类。
2. 若证实脚踩沿/刚落地翻倒，可在**奖励专用真沟边界**附近计一次安全足点/短时稳定恢复信号，限定已越过远沿、两足在实地支撑、无硬失败、每沟上限且不改变 paid、全沟清越或终点判据。借鉴 Cheng 的边缘接触思路和 Atanassov 的落地阶段，但权重须通过同起点 A/B 在本项目标定；勿凭论文常数照抄。
3. 若先在下一沟过速/欠速失稳，可检验**仅在真实沟的接近/清越窗口**用封顶前进和按指令速度校正；绝不全程奖励无限速度，不改 0.35–0.8 m/s 命令范围。Rudin 的“完全取消速度追踪”与本任务约束冲突，不能照搬。
4. A/B 使用相同 checkpoint、64 环境开发计划、相同 seed/迭代/观测/扰动，唯一变量为一项奖励；正式验收需另审计**新 seed**，已用的 1812341965 和开发 seed 2028 不能再冒充 holdout。未验证前不宣称 90% 或启动后三位老师。
