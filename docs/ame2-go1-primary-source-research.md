# AME-2 对 Go1 课程的参考核查（只读研究；未批准实施方案）

日期：2026-10-14。来源：用户提供的 `C:/Users/Administrator/Downloads/main_zh_glm53.pdf`（共 23 页）、[论文 arXiv 2601.08485](https://arxiv.org/abs/2601.08485)、[官方项目站](https://sites.google.com/leggedrobotics.com/ame-2)、[官方最小发布](https://github.com/leggedrobotics/ame2_minimal)及用户给的[Kitjesen/ame2](https://github.com/Kitjesen/ame2)。以下将论文实验、已发布实现和针对 Go1 的建议严格区分。

## 已核实的一手事实

- 论文 PDF 第 4–5 页 §IV-A：控制器 50 Hz，关节 PD 目标动作；教师真值地图每点 `(x,y,z)`，学生神经地图另有不确定性 `u`。AME-2 编码器融合 CNN 局部地图特征、全局池化特征与本体感知，以全局/本体条件化多头注意力选择局部地图信息。PDF 第 5–6 页 §IV-B/C：教师使用真值地图，学生用在线神经地图和历史本体感知，训练结合 PPO、动作蒸馏与表征对齐。**这不是一项无需重新训练即可插入旧 Go1 权重的奖励参数。**
- PDF 第 9–10 页 §V：深度点投影局部栅格、预测高度与方差、里程计融合；第 10 页 Eq. (9) 用 β-NLL（β=0.5）训练不确定性。地图缺失与遮挡不应当作无障碍平地。PDF 第 15 页 §VII 消融：学生去掉不确定性输入、RL 或教师监督都会损害部分未见地形表现；这只说明论文设置下的对照，不构成当前 Go1 的结果。
- PDF 第 6–7 页 §IV-D、表 I：平缓地形上四脚同时离地是非期望事件；着地冲击/非足接触、打滑、自碰等纳入约束。论文附录 A（第 19 页）有需要跳的沟训练，但**没有可直接复制的 Go1 显式起跳时机、空中姿态或落脚控制器**。第 18 页 §VIII-E 还明确说多数失败发生在技能切换处，这和当前混合赛道值得对照，但不是因果证明。
- PDF 第 8 页 §IV-E 与第 18 页 §VIII-F：ANYmal-D 教师/学生分别训练约 80k/40k 迭代；整个 ANYmal-D 流程成本约 60 RTX-4090 GPU 日。附录 A 的 ANYmal-D 沟可达 1.1 m，**绝不能直接套给 Go1**。论文实机包含 ANYmal-D/TRON1；第 18 页也承认 2.5D 高程图、退化感知与技能切换限制。
- [官方项目](https://sites.google.com/leggedrobotics.com/ame-2)链接的 [leggedrobotics/ame2_minimal](https://github.com/leggedrobotics/ame2_minimal/blob/main/README.md) 当前仅发布 G1 教师主路径，学生/神经建图模型权重及训练管线不完整；例如 [livox_neural_map_sensor.py](https://github.com/leggedrobotics/ame2_minimal/blob/main/ame2/ame2/sensors/livox_neural_map_sensor.py) 指向未发布权重。公开的 [AME2_models.py](https://github.com/leggedrobotics/ame2_minimal/blob/main/rsl_rl/rsl_rl/models/AME2_models.py)、[奖励及课程配置](https://github.com/leggedrobotics/ame2_minimal/blob/main/ame2/ame2/tasks/ame2_env_cfg.py) 可用于理解接口，但不能据此声称 Go1 可直接运行。官方 [LICENSE](https://github.com/leggedrobotics/ame2_minimal/blob/main/LICENSE) 是 GPL-3.0；不复制代码到本项目。
- 用户给的 [Kitjesen/ame2 README](https://github.com/Kitjesen/ame2/blob/master/README.md) 自称非官方独立重实现，面向 ANYmal-D/Isaac Lab 0.46.x；[网络模块](https://github.com/Kitjesen/ame2/blob/master/ame2/networks/ame2_model.py) 和 [合成建图训练脚本](https://github.com/Kitjesen/ame2/blob/master/scripts/train_mapping.py) 是参考实现而非官方模型/Go1 权重。README 仅报早期教师训练成功率 0%，不作性能依据。README 声称 Apache-2.0，但[GitHub 仓库元数据](https://api.github.com/repos/Kitjesen/ame2) `license:null`，[完整树](https://api.github.com/repos/Kitjesen/ame2/git/trees/master?recursive=1)未见 LICENSE 文件；**不把 README 声明当已核验的代码授权**。官方 [ame2_undesired_leaping](https://github.com/leggedrobotics/ame2_minimal/blob/main/ame2/ame2/tasks/mdp/rewards.py) 也非 Go1 专用空中稳定控制。

## 当前本地接口与可检验的适配边界

- 本项目旧文档 `sim/agent.md` 记录 Stage1 actor 1073 = 前缀49 + 高度512 + 置信度512，输出12关节动作；当前源 `sim/source/go1_hier_sim/go1_hier_sim/hier_v2/model_v2.py::EncoderActorCore.encode` 把高度×置信度与置信度送入 MLP→32 维 latent，再经 GRU128 与动作头。应以远端当前版本和接口 manifest 最终复核，不凭旧文档推断 schema 已变。
- 当前部署镜像 `v7_work/runtime.py::gate_action` 在没有 `jump_policy`/`base_policy` 时直接返回走路动作；v7b 录像 `GATE leap_envs=0`。地图编码器改版与接通/训练跳跃是两项独立可验证工作；不能只加奖励就宣称能跳。
- 当前 v7b 官方 `stage1-v2` 闸门在相同 512 实例中 `combined=0/398`、`gaps_long=44/65`，而上下楼梯独立族已经分别达标。任何新地形、感知或控制方案必须保留旧 checkpoint/旧报告，用新版本独立计划和状态；仍以**同一次确定性 512 环境的 11 族各 ≥90%**为升档条件。单环境录像只能看动作。

研究结论只是候选方向：面向现有 512 高度/置信度接口的局部-全局地图注意力、可观测性退化对照、只在长沟有效的空中姿态/冲击控制，以及在未见地形组合上留出独立泛化测试。尚未指定台阶新高度、混合赛道新比例、模型 schema、训练预算或部署感知来源；这些必须与用户确认并做 Go1 物理可行性验证后才成为实施方案。
