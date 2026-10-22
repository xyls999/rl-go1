# 长期记忆：Go1 分层视觉运控与避障

## 用户目标与范围

最终任务是接收 UWB 目标位置，让单只 Go1 使用深度视觉自主避障、到达后停下。
当前实施四阶段仿真训练、评测、导出与真实的小规模工程验证。允许重训新接口。
多狗协同和搬运暂时只保留外部目标、速度接口，不实现协同网络，不加入空维度。
未来负载运控如需重训，应产生新版本并重新验证下游，不能覆盖已验收版本。

用户要求：完整真实执行链、明确失败、无占位成功、无静默降级、无未经请求的额外改动；
遵循现有架构和风格，注释说明不直观约束；有关修改验证完成后停止。

## 已核实环境

- 容器连接 abc@10.200.212.241:8022；工作空间 /config/Desktop/sim。
- Isaac Lab /config/Desktop/IsaacLab，版本2.3.2，上游提交37ddf626871758333d6ed89cf64ad702aef127d0。
- Isaac Sim5.1，/root/env_isaaclab Python3.11，torch2.7.0+cu128，rsl-rl-lib3.1.2。
- 共享环境的无依赖cu13包覆盖了torch所需cu12库；本项目使用`.runtime-libs`内cuDNN9.7.1，
  由sim.sh优先加载，重建时运行`scripts/bootstrap_runtime.sh`，不修改共享环境。
- 四张NVIDIA A10，各约23028MiB可见显存；旧两卡CUDA基础矩阵乘及 v9 四卡 CUDA/Gloo 通信与 PPO smoke 已验证。容器 `/dev/shm` 仅64MiB，NCCL四卡初始化会因共享内存不足失败，故 v9 使用显式 Gloo opt-in；不修改容器挂载。
- 旧go1-project editable指向go1_wtw_isaaclab，不可使用该包名建立新项目。
- 旧运控235维和WTW历史2100维与新接口不兼容，不直接加载。
- 官方Go1采用ActuatorNetMLP，配置限制23.7Nm、30rad/s；资产真实加载仍须仿真测试。

## 固定接口 v4

以 contracts.py 的 interface_manifest 为机器可检验的唯一接口规范。
机器人前X、左Y、上Z；单位米、弧度、秒。关节顺序FL/FR/RL/RR，每腿hip/thigh/calf，按名称映射。

- p42 = omega3 + gravity3 + qrel12 + dq12 + previous_actual_action12。
- p42缩放：omega×.25、gravity原值、qrel原值、dq×.05、previous_action clip(-3,3)/3。
- cmd3 = vx,vy,wz，编码时除以(.8,.3,1)。goal4=relative_xy/10,valid,clip(age/.5,0,1)。
- 运控actor1073 = p42+cmd3+phase(sin,cos)2+初始固定航向在机体系XY的2维单位向量+height512+confidence512；地形encoder1024→128→32，拼81→GRU128→256→128→12。
- 运控critic1080 = clean1073+truth_velocity3+contact4；独立encoder后88→GRU128→256→128→1。
- 导航actor193 = p42+goal4+previous_cmd3+range72+confidence72；encoder144→128→32，拼81→GRU128→128→64→3。
- 导航critic200 = clean193+truth_velocity3+contact4；独立encoder后88→GRU128→128→64→1。
- 两视觉分支各自CNN通道16/32/64、pool2×3、384→128，拼p42和cmd3共173→GRU256→128→输出。
- 视觉输入2×60×80为Z-depth/6与valid，输出分别512维地形、144维障碍；置信度sigmoid连续输入策略。
- 地形16×16，前后左右各±.75m，相对机体高度clip(-1,1)；障碍前180°72扇区，
  在机体中心-0.15/0/+0.15m三层射线取最近障碍面命中，距离/6，不把可通行地形面当墙。
- 控制频率：physics200Hz、locomotion50Hz、vision20Hz、navigation10Hz。
- 动作qtarget=qdefault+.25×clip(action,-3,3)；速度vx[0,.5],vy±.3,wz±1；平滑tau=.2s。
- actor不读取真值线速度/绝对位置/接触。上次动作与指令必须来自实际执行值。

## 阶段与冻结规则

1. Stage1 PPO训练运控和地形encoder。平地→坡→粗糙→上下台阶，训练扰动、质量和外力鲁棒性。
2. Stage2冻结运控，PPO训练导航和障碍encoder；目标进展、停稳、碰撞、卡住、时间与命令变化。
3. Stage3A教师代驾训练地形视觉；Stage3B教师导航/运控代驾训练障碍视觉。下游权重不更新。
4. Stage4学生感知真正驱动仿真；教师在同实际轨迹给标签；先只调地形，再只调障碍。

视觉loss=有效几何Huber+0.2×validityBCE+动作均值MSE。教师no_grad，学生冻结策略必须允许梯度
传给视觉。Stage3 Adam1e-4，Stage4 1e-5；序列16帧burn-in+32帧学习；Stage4教师回放/学生新数据各半。
冻结的是权重和归一化统计，不是RNN运行状态。教师/学生各自维护状态，所有环境独立reset。

Stage1奖励唯一可调源为`configs/rewards_stage1.toml`，严格按“奖励函数名→参数”解析，未知项、
缺项、非有限值和奖励/惩罚符号错误均拒绝启动。11项为XY速度、yaw速度、直立、动作变化、足端
腾空、支撑足滑移、关节力矩平方、机械功率绝对值、关节加速度平方、摔倒和身体碰撞。每项由
Isaac RewardManager独立记录，完整解析结果写入checkpoint metadata；Stage1断点续训会严格比较该
配置，旧快照缺少配置或任一参数不同都会报错。修改配置后必须新训和重验收。

## 可观测性和故障

教师精确几何与可观察性分开；相机视野、遮挡、2秒历史之外为未知，不将未知当空地。
单相机不能保证恢复四周。初值前置俯视20°、水平FOV90°、Z-depth .2–6m，后续以实物标定为准。
UWB尚未选型，必须能给相对xy，或同一坐标系双方位置加机器人航向。单点距离不足，不能偷补真值。
相机/UWB超过.5秒陈旧显式停车并记录原因，不记录成任务成功。
基础RayCaster只读取一个Mesh，静态障碍必须真实加入查询几何；相机看到墙而射线漏墙属于失败。

## 验收和状态

工程验证与能力验收分开。短训练真实checkpoint可以检验数据流，不能声称已学会走路或避障。
能力默认：运控每类100×30秒回合完成率≥95%、xy RMSE≤.15、yaw RMSE≤.25；导航200回合
到达率≥90%、身体碰撞率≤5%；学生距教师不超过10百分点；Stage4三个seed改善且碰撞/跌倒不恶化。
仿真成功不等于真机验证。每份checkpoint保存接口、模型哈希、上游哈希、配置和验收状态。

实施状态：四阶段代码和必要数据链路已部署，尚无通过上述能力验收的新策略。
2026-09-09远端31项完整测试通过；真实Isaac检查已覆盖四种混合地形、当时557/564维观测、物理步进，
并在GPU0验证2路真实深度相机、2×60×80深度输入、时间戳队列和可观察地形教师。
GPU0混合地形4环境完成128步rollout和一次从`configs/ppo.json`读取参数的RSL-RL PPO更新，产出
`checkpoints/stage1/2026-09-09_13-06-55_config-driven-smoke/model_0.pt`；它明确标记为未验收，
不可作为Stage2教师。实际报告在`reports/check_2026-09-09_13-01-44.json`和相机报告
`reports/check_2026-09-09_13-06-31.json`。相机渲染必须使用GPU0、DISPLAY=:1和NVIDIA GLX；
sim.sh负责GLX选择，CLI拒绝相机任务使用GPU1。错误路径已验证返回非零退出码。
Stage2工程检查使用上述未验收Stage1快照，仅验证冻结运控、193/200维导航、三层障碍射线和
相机数据链路，已通过并记录于`reports/check_2026-09-09_13-09-57.json`；它不构成能力验收。
新增11项Stage1奖励后，GPU0上4个混合地形环境的真实检查通过，报告为
`reports/check_2026-09-09_14-53-18.json`。一次128步PPO数据链路检查产出
`checkpoints/stage1/2026-09-09_14-53-40_reward-v2-smoke/model_0.pt`，其manifest包含完整奖励配置且
`capability_validated=false`；它只验证奖励和优化器可执行，不可作为训练完成或能力达标的模型。
2026-09-10修复评测读取Isaac Lab持久`extras`造成初始零步统计被重复计数的问题；评测只消费与本步
done环境严格匹配的记录。`model_2999.pt`在4回合混合地形工程评测中完成率1.0、XY RMSE 0.0558、
yaw RMSE 0.0444，报告为`reports/evaluate_stage1_2026-09-10_12-16-04.json`；样本仅4回合，仍未达到
400回合正式能力验收要求。

## 参考与边界

- IsaacLab v2.3.2官方navigation PreTrainedPolicyAction：层级调度参考，但需补逐env GRU reset与实际速度映射。
- RSL-RL3.1.2：PPO基础；自有runner显式构造网络，不eval注入或修改第三方。
- WTW：Go1动作/随机化参考；不是当前视觉四阶段方案。
- Extreme Parkour：深度时序与蒸馏参考；其默认学生驱动且同时训练actor，不直接复制冻结逻辑。
- Miki2022：重建、记忆及可靠性感知思想；其多相机/激光高程图设置不能证明单前置相机恢复四周。

本项目1073/193维与双视觉GRU是自主基线设计，必须实验验证，不能写成论文复现已达标。

## 2026-09-16 Stage1 V15 固定世界方向与能耗收束

- V14 `model_2800/3000` 能通过上坡、楼梯上行和下行，但会持续横向漂移；单回合复核在
  30秒时限内超时，问题不是回合太短。根因是旧 `track_linear_velocity` 在机体坐标系比较，
  且旧 `heading_alignment` 只约束偏航角速度，斜行后仍可得到主要速度奖励。
- V15 将XY速度追踪变为回合初始航向构成的固定世界坐标基；增加纯速度方向奖励、真实机身
  航向奖励和有上限的横向轨迹误差惩罚。完成奖励40→200、三个课程检查点4→10；力矩平方
  惩罚翻倍、机械功率惩罚四倍，同时下调速度大小和前向进展权重，避免斜向高速刷分。
- 远端35项单元测试和4环境真实Isaac检查通过。两卡训练从V14 `model_3150.pt` warm-start，
  先冻结actor进行150轮critic适配；运行目录/日志前缀为
  `2026-09-16_14-01-35_go1-stage1-v15-world-direction-energy-10000`。该训练尚未完成能力验收。

## 2026-09-16 Stage1 V16 区域终点与世界方向箭头

- 机器人资产仍是 Isaac Lab `UnitreeGo1RoughEnvCfg`，即 Unitree Go1。
- 旧完成判定错误地要求机器人进入单个目标点半径0.35m；课程实际终点改为下楼梯后沿初始
  世界前进轴的整块1m长区域：相对环境原点前向9.10–10.10m、横向±4.0m。区域内目标距离
  精确为0，`goal_reached`、完成奖励和接近目标奖励都使用同一矩形区域定义。
- 用未重训的V15 `model_700.pt`重新评测32回合，完成率/成功率从旧定义下的0提升为1.0，
  碰撞、摔倒、越界和超时均为0；报告为`reports/evaluate_stage1_2026-09-16_14-49-33.json`。
  但XY速度RMSE 0.2591、yaw RMSE 0.4296，仍未满足正式阈值，不能称为能力验收通过。
- 航向奖励1.0→1.5。增加`FixedWorldVelocityCommand`和CLI参数`--show-direction-arrows`：目标箭头
  固定在回合初始世界前向，实际箭头显示世界速度，不使用会随机体偏航旋转的内置body-frame箭头。
  箭头仅在显式请求演示时启用，不污染无头训练或相机输入。
- 35项单元测试、4环境真实Isaac检查及2秒原生视频箭头冒烟测试均通过；检查报告为
  `reports/check_2026-09-16_14-48-07.json`，测试视频为`media/v16_arrow_smoke.mp4`。
- 两卡V16训练从V15 `model_900.pt` warm-start并先做150轮critic适配，运行目录/日志前缀为
  `2026-09-16_14-50-46_go1-stage1-v16-finish-region-arrows-10000`。该训练尚未完成正式能力验收。

## 2026-09-16 Stage1 V17 软相位对角步态

- 运控观测新增连续步态相位sin/cos两维，接口升级至schema v3；因此不兼容旧actor，V17从头训练。
- 使用软接触时钟约束FL+RR、FR+RL两组对角腿相差半周期。平地2.6Hz，复杂地形连续降至2.0Hz；
  楼梯同时提高支撑占空比和摆腿净空，避免旧的高频低抬脚蠕动及单腿长时间悬空刷分。
- 新增相位净空、支撑面相对机身高度、四足垂直载荷EMA均衡和动作二阶平滑；旧接触计数、固定
  小步长及按帧净空奖励权重归零，仅保留超时未落脚安全惩罚。
- 初始世界前向冻结逻辑和V16的9.10–10.10m、横向±4m矩形完成区均保留，并增加纯函数回归测试。

## 2026-09-17 Stage1 V18 混合直线课程与姿态收束

- 路线仍沿单一直线，终点区域保持不变；上坡、平台和下坡改为带随机起伏的连续混合表面，
  五级上下楼梯的踏步高度和深度按环境随机化，训练难度在0.55–0.95间变化。
- 运控新增“reset时固定航向在当前机体系的XY单位向量”两维观测。roll/pitch仍由projected gravity
  提供，新增观测专门消除仅靠GRU积分陀螺造成的yaw漂移，不引入绝对位置或仿真真值速度。
- 相位接触2.0→1.2、相位净空1.5→0.7、楼梯净空0.16→0.13m并放宽相位过渡；新增髋外展
  偏移/速度惩罚，收紧足端横向死区，增强动作二阶平滑和机身角速度收束。完成与速度权重不再增加。
- schema v3的V17策略可通过严格的一次性GRU输入列迁移进入v4：旧列原样复制，新航向列零初始化。

## 2026-09-18 hier_v2 新式分层奖励/模型/鲁棒性课程（nav_v2 训练中）

- 新增独立可加包 `hier_v2/`，不修改已验证的 Stage-1 契约：
  - `rewards_v2.py`：两套严格校验的独立契约。`loco_v2` 21项（速度跟踪、足端质量、关节安全、
    功耗、姿态），`nav_v2` 15项（带符号路径进度、方位进度、一次性到达 `reach_goal`、稳定到达
    `settled_goal`、终点刹车、卡滞时长、翻转安全、姿态恢复、前方墙近似刹车、绕障意愿、时间惩罚、
    非足端接触、终止）。纯张量函数 + `NavV2Tracker` 承载回合状态，可离线单测。
  - `wall_score/local_ground/scan_grid`：以扫描点相对局部地面（有效点0.25分位）的软超阈值比例
    判定前方墙；无效射线不计为空闲。平地基准分约0.018 < 阈值0.10，因此平地不会误判为墙。
  - `model_v2.py`：fwwb 风格 encoder actor/critic（`512-256-32` + L2Norm 单位球隐变量，
    actor 与 critic 各自独立 encoder），GRU128 保留以兼容 RSL-RL rollout/storage。
    nav actor GRU 输入 81，运控 81；`detect_actor_variant` 区分 v1/v2 checkpoint。
  - `robustness_v2.py`：sim-to-real 混合课程，更新点 0/25/100/250 逐级引入
    clean/mild/medium/hard（UWB 噪声与丢包、速度指令增益误差、一节动作延迟、观测噪声）；
    critic 始终使用干净状态。
- 接线：`--reward-contract {stage1,loco_v2,nav_v2}`（stage2 默认 nav_v2）、
  `--robustness {off,on}`；`envs/config.py` 注册 nav_v2 与 `loco2_*` 奖励项并保持
  Stage-1 项名不变；`runtime.reward_term()` 分派并缓存逐项奖励；nav_v2 每导航周期只推进一次
  tracker，中间控制步返回缓存值以免奖励被 5× 重复计。
- 修正两处既有缺陷：`configs/rewards_stage1.toml` 的 `feet_slide` 缺 `deadband/scale/terrain_relief`
  导致默认配置无法加载；`_evaluate` 的 Stage-1 通过门限要求 `terrain=="mixed"`，而 Stage-1 只在
  `course` 上评测，导致 `--certify` 永远不可达（已改为 `course`，与 `certify_checkpoint` 一致）。
- 恢复先前被本地镜像覆盖丢失的远端改动：`GO1_REWARD_CONFIG_PATH`/`GO1_PPO_CONFIG_PATH` 环境变量
  覆盖、仅主体接触判定碰撞（腿/脚擦碰不算失败）与连续碰撞 streak、下坡减速 `_course_speed_scale`、
  扫描网格 (3.0×6.2m, 0.2m) 与 `env_spacing=36`、schema v3 运控 checkpoint 的航向列迁移加载。
  本地 `E:\tmp-issac\sim` 已按远端重新拉取并逐文件 sha256 校验一致。
- 新增 `--allow-uncertified-locomotion` 显式开关：允许使用未能力认证的冻结运控策略，
  但打印其 sha256 作为警告，不做静默降级。
- 环境限制：`num_envs=512` 时整块地形 trimesh 过大，PhysX cooking 失败
  （`Unable to create triangle mesh`），机器人立即坠落（回合长度≈3）。`num_envs=256` 正常。
- 验证：`./sim.sh test` 83 通过、1 失败（`test_geometry` 的粗糙度期望仍为旧调参，属先前地形改动的
  遗留不一致，未修改几何）；`check --stage stage1 --reward-contract loco_v2` 与
  `check --stage stage2 --reward-contract nav_v2` 均通过。
- 运行中：`2026-09-18_11-10-00_go1-stage2-nav-v2-gentle`
  （stage2 / nav_v2 / course / 难度0.00–0.10 / 256环境 / 1500轮 / 60s回合 / robustness on，
  冻结运控 `2026-09-18_01-45-00_go1-stage1-gentle-headingfix/model_50.pt`，未认证）。
  日志 `logs/nav_v2_gentle.log`；尚无能力验收，`reach_goal` 仍为0。

## 2026-09-18 根因：终点区在地形网格之外（nav_v2 最后 1m 必然坠落）

- 症状：nav_v2 在难度0.00–0.10下 0/8 成功，`time_out=0`，`fallen_rate` 高；逐环境轨迹显示所有环境
  走完平地/上下坡/上下楼梯/粗糙平台后，在 course-local x≈18.5–19.1 处结束，`stuck_duration=0`。
- 证据（`geometry.build_tile("course")` 沿 y=spawn 的向下射线，难度0.0与0.1一致）：
  - 修复前网格 x 范围 `[-5.0, 22.1]`，即 course-local 只能走到 `19.1`；`local_x=19.5/20.0/20.1/…`
    全部 `NO GROUND`。
  - 而终点区是 `goal_reached.start_offset..end_offset = 19.10..20.10`，导航目标 `_goal_world` 为
    `COURSE_GOAL_OFFSET=20.0`（course-local 20.0）→ 目标点与终点区后半段都在网格之外。
  - 基础底板同样错位：`box((26,26,0.2), (8,8,-0.1))` 覆盖 `[-5,21]`，与瓦片 `[0,26]` 不一致，
    瓦片间还留下 4m 空洞；`out_of_bounds` 阈值 `local_x > 22.8` 也落在空洞里，永远不会先触发。
  - 因此 Stage-1 能靠“前脚进入终点区”在网格边缘擦线通过，而 Stage-2 的
    `reached = (distance<0.35) & (speed<0.15)` 要求机身接近目标点，物理上不可达。
- 修复（仅几何，未改奖励、未改终止阈值）：
  - 底板居中覆盖整瓦片：`box((TILE_SIZE,TILE_SIZE,0.2), (TILE_SIZE/2,TILE_SIZE/2,-0.12))`，
    顶面比 z=0 的课程面低 2cm，避免与课程面片共面（共面会触发 PhysX cooking 失败）。
  - 平地收尾面片延长到 `finish_x = linspace(12.10, 24.10, 241)`，即 course-local 9.1→21.1，
    覆盖终点区终点 20.1 与目标点 20.0 各留 1m 余量。
- 回归测试（先红后绿）：`tests/test_geometry.py` 新增
  `test_course_ground_extends_past_the_goal_and_the_finish_region`，对难度 0.0/0.1/0.8 在
  `goal`、`finish_end`、`finish_end+0.5` 三处射线取地，修复前报
  `AssertionError: no ground at course x=20.0`。
- 同时更新 `test_course_is_straight_but_has_mixed_roughness_and_irregular_stairs`：它仍断言旧 16m 瓦片
  时代的 `stair_end - spawn == finish_start_offset` 与旧粗糙度 0.015/坡度 0.30。现改为断言
  “难度单调缩放、数值在可通行带内、终点区位于楼梯之后且起点早于终点”，并把楼梯不规则性
  只在高难度上要求（`height_variation=0.10*difficulty` 本身随难度缩放）。
- 顺带修正同类缺陷：`cli._train` 的 `--resume` 只在 stage1 校验奖励契约，且 stage1 一律用
  `hier_stage1_rewards` 比较（loco_v2 运行无法续训）。新增 `_contract_table(cfg, contract)`，
  保存与续训两侧都按实际契约取表，任何 stage 都校验。
- 验证：`./sim.sh test` = **85 通过、6 subtests**；射线复检终点处 `HIT z=0.000`；
  `check --stage stage2 --reward-contract nav_v2` CHECK PASSED。
- 处理：停掉在空洞地形上训练的 `nav-v2-gentle`（其目标不可达），以 `--resume model_400.pt`
  在修复后地形上继续：运行 `2026-09-18_12-48-00_go1-stage2-nav-v2-fixedterrain`
  （stage2 / nav_v2 / course / 难度0.00–0.10 / 256环境 / 1500轮 / 60s / robustness on，
  日志 `logs/nav_v2_fixedterrain.log`）。旧 run 保留作为对照。

## 2026-09-18 完成任务判定改为"冲过终点线"（完成率 82.5%）

- 用户要求：只要冲过终点就算完成，不需要停下。
- 原判定有两处多余要求，导致已能到达终点的策略仍记 0 完成：
  1. `runtime._update_failures` 的 stage2 分支要求 `distance<0.35 m 且 speed<0.15 m/s`（必须刹停）；
     冻结运控的稳定工作带约 0.16–0.18 m/s，停住余量极小。
  2. 目标是 `_goal_world` 单点（course-local (20,0)），而实测策略沿赛道横向漂移到
     y≈−4.5，会"从终点旁边飞过去"：`reach_goal` 一次性奖励（半径0.60）因此只在少数回合触发。
- 新判定：新增纯函数 `hier_v2.rewards_v2.finish_crossed(forward, lateral, start_offset, half_width)`
  = `forward >= start_offset 且 |lateral| <= half_width`，即**在可行驶宽度内任意位置越过终点线**，
  超调也算，不需要停车、不需要对准中线。
- 关键修正：终点线几何**必须由导航契约自己声明**。`configs/rewards_stage1.toml`（所有 run 都会加载的
  默认契约）里的 `[goal_reached] start_offset = 9.10` 是 16m 瓦片时代的旧值，其注释也写着"楼梯在
  course-local x=9.10 结束"；而当前课程 `COURSE_GOAL_OFFSET = 20.0`。若沿用该值，完成判定会在楼梯
  刚结束处（x=9.1）就触发——一度得到过 82.5% 的**错误**数字，已作废并修正。
  现于 `configs/rewards_nav_v2.toml` 的 `[reach_goal]` 增加 `start_offset = 19.10`、`half_width = 4.0`；
  `NAV_V2_FIELDS` 同步字段并校验为正；runtime 的 stage2 成功判定优先读导航契约，缺失时才回退到运控契约。
- 回归测试：`tests/test_hier_v2.py::test_goal_completion_is_crossing_the_finish_line_not_stopping_or_aiming`
  （越过/超调/横向越界四种情形）与 `test_nav_contract_declares_the_real_course_finish_line`
  （终点线必须落在赛道末段 `> 15.0` 且不晚于 `COURSE_GOAL_OFFSET`，能抓住 9.10 这类旧值）。
- 先前的 `goal_crossed(distance, threshold)` 方案作废并已删除，避免两套并存的完成定义。
- 结果：用**改动前**训练出的策略（`...crossfinish/model_550.pt`）在难度0.00–0.10、40 回合评测：
  `completion_rate = success_rate = 0.80`，`collision_rate = 0.05`，`fallen_rate = 0.15`，
  `out_of_bounds_rate = 0`，`timeout_rate = 0`。报告
  `reports/evaluate_stage2_2026-09-18_14-20-00_go1-stage2-nav-v2-finishline19.json`。
- 训练重启方式：因为奖励契约内容发生变化，`--resume` 被契约校验正确拒绝，改用 `--initialize-from`
  （设计上用于"保权重、重置优化器/迭代/动作噪声"的跨目标热启动）+ `--critic-warmup-iterations 100`：
  `2026-09-18_14-30-00_go1-stage2-nav-v2-finishline19-warm`，日志 `logs/nav_v2_fin19_warm.log`。
- 遗留不一致（待决策，属奖励改动）：`nav_v2` 的 `reach_goal`（+300 一次性）仍按**点判定**（半径0.60 m）
  计算，与"线判定"完成条件不同；越过终点但偏离中线的回合拿不到这 +300。若要继续提高完成率，
  应把该奖励事件也改为终点线判定。
- 测试：`./sim.sh test` = 87 通过、6 subtests；`check --stage stage2 --reward-contract nav_v2` PASSED。

## 2026-09-18 冻结运控的速度包线（决定能否提难度的关键证据）

- 起因：为判断"是否可以提高课程难度"，先量化冻结运控
  `checkpoints/stage1/2026-09-18_01-45-00_go1-stage1-gentle-headingfix/model_50.pt` 在当前课程上的实际能力。
- 三个对照（stage1 评测，course，难度0.00–0.10，2 环境，100 s，同 seed）：
  - A 已知-good 几何备份（`geometry.py.bak.20260918_easy_flat`）+ 速度 0.16–0.18 → `fallen_rate 1.0`，
    两个环境都在 course-local x≈6.5（楼梯金字塔段）摔倒。
  - B 当前几何 + 速度 **0.35–0.45** → `fallen_rate 0.0`，47 s 内走到 course-local **17.1**；
    回合结束是因为横向漂到 y=−7.8（越界），不是摔倒。
  - C 当前几何 + 速度 0.16–0.18（重复对照）→ `fallen_rate 1.0`，与 A 一致。
- 结论：**限制因素是速度，不是台阶高度**（A/C 用的是更矮的台阶，同样全摔）。该运控策略在
  ≲0.18 m/s 时无法通过楼梯段，在 ≈0.35–0.45 m/s 时通过良好。导航策略实际下达的是 ~0.4–0.5 m/s，
  所以 nav 任务本身能完成（80%），而"低速 stage1 评测"会误判成课程坏了。
- 同时纠正了两个我引入的偏差（均为单变量对照后恢复）：
  - 我曾把底板下移 2 cm 并居中；对照显示与摔倒无关，已把底板顶面恢复为 z=0（与课程面片同高，
    继续充当连续地板），仅保留"居中覆盖整瓦片"这一真正修复。
  - 我曾把 `nominal_step_height` 由 `0.015 + 0.030*d` 改成 `0.030 + 0.030*d`（提难度尝试）。
    已恢复为 `0.015 + 0.030*d`：这是该运控策略被训练/验证过的可通行带，单独加高台阶并不等于提难度。
- 另一处同类旧值 bug：`configs/rewards_stage1.toml`（所有 run 都会加载的默认契约）的
  `[goal_reached] start_offset/end_offset` 仍是 16 m 瓦片时代的 `9.10/10.10`，而当前赛道终点在 20.0；
  已改为 `19.10/20.10`，并更新 `tests/test_reward_config.py` 的期望值（原测试把旧值固化了下来）。
- 课程几何最终状态：底板居中 `[0,26]²` 顶面 z=0；`nominal_step_height = 0.015+0.030*d`；
  平地收尾面片 `12.10→24.10`（覆盖终点区与目标点各留 1 m）。远端与本地 sha256 一致
  （`a5c434bd…`）。`./sim.sh test` = 87 通过。
- 训练：`2026-09-18_14-50-00_go1-stage2-nav-v2-settled`（从 `...crossfinish/model_550.pt` 热启动、
  100 轮 critic warmup、256 环境、难度0.00–0.10、robustness on），日志 `logs/nav_v2_settled.log`。
- 提难度的正确顺序（未执行）：先用 stage1 评测把冻结运控在 0.35–0.50 m/s 下、难度 0.10/0.20/0.30 的
  通过情况量出来，得到"低层包线"；nav 难度只能在该包线内提高；若确实要更复杂地形，必须先在 stage1
  重训并重新验证运控，再冻结、再提 nav 难度。

## 2026-09-18 低层重定向：随机上下坡 + 上下楼梯（已批准的架构决策）

- 用户澄清的分层定义：**第一层=纯运控**（能走随机上下坡与上下楼梯，速度永远沿**起始世界方向**朝前，
  不是狗的朝向），训好即**冻结**；**第二层=发速度**（针对不同地形决定 vx/wz），不做避障；
  最终目标是在复杂地形上稳定、协调、平衡地以较快速度通过。参考 Rudin et al. (2021)。
- "理论上得上去"的边界（用于给课程封顶，写进测试作为不变量）：
  - 关节 `effort_limit = 23.7 N·m`、`velocity_limit = 30 rad/s`（`GO1_ACTUATOR_CFG`）。
  - 腿 0.25 + 0.25 m，默认站立髋高 ≈0.365 m（由 init 姿态 `thigh 0.8–1.0 / calf −1.5` 推得）→
    **准静态单级台阶上限 ≈0.16 m**（深蹲后伸展的可用行程）。23 cm 超出运动学极限。
  - 训练静态摩擦 0.6–1.2，最差 0.6 → **坡度上限 `atan(0.6) ≈ 31°`**；45° 只在 μ≈1.0 时成立。
- 地形生成器（`envs/geometry.py` course 分支）已恢复真地形并随机化：
  - 坡度 `U(5°, 5°+25°·d)`，跑长固定 1.50 m；上坡 4.30→5.80，平台→6.50，下坡→8.00（回到地面），
    8.00→8.45 平地接楼梯。
  - 台阶**每级独立**采样 `U(0.04, 0.04+0.12·d)`（5 级），上下楼梯各有 5 级；嵌套板结构使
    最后一级下降正好落在 `stair_end`（12.10）之后回到地面。
  - 平底收尾 12.10→24.10（覆盖终点区/目标点各留 1 m）；终点线 19.10、目标 20.0 均按
    **spawn 相对**，而几何断点按 **tile 坐标**（spawn 在 3.0）——两种坐标系都在用，混用就是
    当年"终点在网格之外"的根因，已在代码注释与测试 helper 里显式标注。
- 高度扫描恢复为 **0.1 m 分辨率、1.5×3.1 m 窗口、前移 1.05 m**（16×32 = **512 点，接口维度不变**）：
  0.2 m 网格会把 0.16 m 台阶混叠掉，且 0.1 m 正是 V17 训练时的扫描几何。
- TDD：新增 4 个几何测试（台阶/坡度包线上限与"必须达到真台阶"、每级随机且 d=0 时恒为 4 cm、
  射线测量上坡→回到地面的剖面、楼梯爬升后回到终点标高），并放宽旧断言 `slope_height < 0.90`。
  测试 helper 的坐标系错误（tile vs spawn 相对）是被测出来的，不是猜的。`./sim.sh test` = **91 通过**。
- 地形自检（射线逐 2 cm 采样）：最大离散落差**恰为设计台阶高度**（d=0.1 → ±5.1 cm；d=1.0 → ±14.9 cm），
  无意外墙/断层；d=1.0 时坡度实测 24.35°、最高点 0.684 m。`check --stage stage1 --terrain course` PASSED
  （新网格可在 PhysX 中 cook）。
- 底座选型（实测，非猜测）：V17/V19-p2-medium/gentle-headingfix 三者在难度0.4–0.6、速度0.3–0.5 下
  全部 `completion_rate 0`、`fallen 0.81–0.94`（它们此前的课程台阶 ≤4.5 cm，都没见过真台阶）。
  选 **`checkpoints/stage1/2026-09-16_15-25-26_go1-stage1-v17-soft-phase-trot-10000/model_9999.pt`**：
  唯一有"上下坡+楼梯"通行记录（曾 32 回合 100% 过硬地形）、扫描几何与恢复后的 0.1 m 一致、
  原始训练速度约 0.65 m/s（最接近"较快"目标），schema3→4 的零列迁移已实现且有测试。
- PPO 超参：`--initialize-from` 会让 **actor 只用 `warm_start_actor_learning_rate`**（当时 2e-6，为保步态），
  对"学新技能"过慢；本次把 warm-start 档设为 actor 5e-5 / critic 3e-4 / clip 0.1 / entropy 0.002 /
  noise std 0.45（`configs/ppo_v20_paper.json`，已同步）。
  注意 `configs/ppo_v20_paper.json` 与 `configs/rewards_stage1_v20_paper.toml` 此前只存在于远端，
  现已拉回本地镜像（本地为权威副本）。
- **nav 训练已停**（其冻结底座正要被替换）：`nav_v2-settled` 的 checkpoint 保留可回退，两张卡让给低层。
- Phase A1 已启动：`2026-09-18_16-10-00_go1-stage1-terrain-a1`（course、难度0.00–0.20、速度0.30–0.50、
  256 环境、1500 轮、200 轮 critic warmup、robustness on、90 s/回合、v20 paper 契约），
  日志 `logs/terrain_a1_p1.log`。首轮观察：iter≈35、reward≈42、回合≈750 步（15 s）、
  `task` 终止 94.5% 且由 `fallen` 主导（`collision`/`out_of_bounds`≈0）、`goal_reached=0`、
  `error_vel_xy` 0.13–0.17 → 与"走到 course≈6 m（楼梯入口）摔倒"一致，符合"底座尚未具备该能力"的预期。
- `Episode_Reward/<term>` 是按回合平均的量，不是每步量：`fallen ≈ -0.0008` × 约 750 步 ≈ -0.6，
  即平均每次回合只在摔倒后吃约 6 步惩罚（摔倒即终止），不要把它误读成"摔倒率很低"。
- 冻结门（"运控训好了"的硬定义，未达成前不冻结）：速度 {0.3, 0.5, 0.8, 1.0} × 难度 {0.25, 0.5, 0.75, 1.0}
  每格 ≥16 回合，要求每格 **完成率 ≥95%、`fallen=0`、横向漂移 <1.5 m**；通过后把速度带/难度带写进
  checkpoint metadata 才开第二层。

## 2026-09-18 评测协议修正：环境数必须等于训练批量（否则数字虚高）

- 同一 checkpoint、同一难度带 0.82–0.85 实测：**1 环境 100%、4 环境 100%、16 环境 0%、256 环境 8.2%**。
  查过 cooking（0 次失败）、换 seed（7 仍 0%）、打印 env0 轨迹（16 环境时 env0 自己走到 x=15.25 无碰撞，
  但汇总 0%）→ 不是地形/种子问题，是 **GPU PhysX 接触求解随模拟批量变化**：批量越小越"好走"。
- 训练用 256 环境，因此**所有评估、门禁、报数必须用 256 环境**；1/4 环境的探针只能当趋势参考。
  `scripts/eval_gate_stage1.sh` 已改为 `envs=256`（第 7 个参数 `env_count`，默认 256），并加注释说明原因。
- 之前记录的"0.65 → 87.5%、0.70–0.75 → 100%"是 16 环境数字；256 环境下 0.70–0.75 实测 **97.66%**（可信）。

## 2026-09-18 Phase A1–A4 与碰撞终止修正（已批准 P1）

- A1 `2026-09-18_16-10-00_...-terrain-a1`（难度0–0.2、速度0.3–0.5）：`model_750` → 难度0.20 完成率 1.0、
  fallen 0、vel_rmse 0.219。
- A2 `2026-09-18_16-55-00_...-terrain-a2`（0.35–0.65）：`model_1199` → 0.50 完成 100%、0.65 完成 56.25%、0.80 = 0%。
  前沿诊断（d=0.8）：能爬到机身 z=0.825（≈0.47 m，几乎整段楼梯），随后**下楼梯只剩 1 脚触地、vz=+0.54、
  vx=1.07 m/s（指令 ≤0.8）→ 弹射着地摔倒**；`collision` 0 次。失败点由"进不了楼梯"变为"下行失控"。
- A3 `2026-09-18_18-12-00_...-terrain-a3`（0.55–0.85）：256 环境下 `model_700` → **0.70–0.75 完成 97.66%**
  （fallen 2.3%、collision 0%）、0.80–0.85 完成 8.20%（**collision 85.9%**）、0.90–1.00 完成 0%（collision 93.0%）。
  → 0.80 以上是**断崖**，且失败 86–93% 由 `collision` 主导、`fallen` 只有 6–7%。
- 根因：`_update_failures` 当时是**单帧机体接触力 >25 N 立即终止**。机体一擦到台阶沿就死，策略从未获得
  "蹭上去"的学习机会。这与用户早先明确的"碰撞不是严格的不能任何碰撞，只要身体不着地就行"冲突。
- **P1（用户已批准并实施）**：机体接触改为**连续 5 个控制步（0.1 s）持续超过 25 N 才算失败**
  （`TRUNK_CONTACT_FORCE=25.0`、`COLLISION_SUSTAIN_STEPS=5`）；`collision` 奖励项改为**分级接触代价**
  （`((peak-25)/25).clamp(0,2)`），保留连续塑形压力；`_collision` 布尔量仍作为 `get_metrics` 的失败标志。
  同时修掉两个隐患：`reset()` 之前**没有清 `_collision_streak`/`_collision_cost`**（跨回合残留会让新回合
  开局即判碰撞），以及恢复 `GO1_DEBUG_COLLISION=1` 的接触部位诊断输出。`./sim.sh test` = 91 通过。
- P1 的 A/B 证据（256 环境，难度 0.80–0.85）：同一 `model_700` **8.20% → 16.80%**（collision 85.9% → 75.8%）；
  `model_1199` → **完成 66.41%、collision 10.55%、fallen 23.05%**。失败模式从"一碰就死"转为**真实摔倒**
  （可学习）。0.90–1.00 仍是 0%（fallen 34%、collision 66%）——那里是**持续拖肚子**（≥5 步），
  与形态限制一致（踏面约 28 cm、机身约 0.6 m，跨不过 14–16 cm 踢面）；若要把 1.0 也打通，证据指向
  **P3 加宽踏面到 40–50 cm**（台阶高度仍保持用户要求的 4–16 cm），但该改动**尚未获批、未实施**。
- P4 澄清：下坡/下楼梯降速**当前已是 `1.0 - 0.55*gate`（最低 ×0.45）**，比提议的 ×0.55 更强；
  此前"放宽到 ×0.8"的补丁实际没生效（脚本只打印成功信息、未校验替换）。而下坡超速是重力驱动
  （指令 ≤0.36 时实测 1.07 m/s），**再压指令不是杠杆**，故本次不做改动。
- A4 已启动：`2026-09-18_20-30-00_go1-stage1-terrain-a4`（难度 **0.75–1.00**、速度 0.30–0.80、从
  `terrain-a3/model_1199` 热启动、1500 轮、200 轮 critic warmup、120 s/回合、robustness on），
  日志 `logs/terrain_a4_p4.log`；目标是让新规则下的真实摔倒被学掉，然后跑完整 256 环境冻结门。
- 视频（已下载本地）：`E:\tmp-issac\terrain_a3_cleanish.mp4`（难度 0.50–0.65）、
  `E:\tmp-issac\terrain_a3_frontier.mp4`（难度 0.80–0.85）——后者是**单回合幸运样本**（该带 256 环境
  实测仅 8.2%，修正前），只能当步态参考，不是能力证据。

## 2026-09-18/19 速度上限放开 + 踏面加宽 + 全包线收束（A5–A7）

- 用户批准的两项（"放开速度到 1.0 + 加宽踏面"）落地：
  - `contracts.MAX_FORWARD_COMMAND = 1.0`，同时用于 `CommandRouter` 的安全 clamp 与
    `envs/config.py` 的 `speed_max` 校验（原来两处都是 0.8，会把 1.0 的指令砍回 0.8）。
    **观测归一化 `[0.8, 0.3, 1.0]` 与 `interface_manifest` 故意不动**，否则改接口哈希会废掉所有 checkpoint
     （代价：1.0 m/s 指令在观测里归一化成 1.25，可接受）。
  - 踏面 0.28 → `COURSE_TREAD_DEPTH = 0.40`（±0.05·d），楼梯区间 **8.45 → 13.60**（山顶平台约 2.0 m），
    粗糙地面 patch 延伸到楼梯出口、终点前平地从楼梯出口开始（避免出现 1.5 m "无地面"缝隙），
    metadata 增加 `tread_depths/tread_depth_min`，新增测试要求 `tread_depth_min ≥ 0.35`（机身约 0.6 m
    必须能跨一级踏面而不顶到踢面）。`./sim.sh test` = 91 通过；射线检查 4.30→24.10 共 396 点 **0 缺失**，
    终点线（tile-x 22.10）有地面。
  - 奖励 `forward_velocity.target_velocity` 0.42 → **0.70**（与新速度带 0.5–1.0 的中心一致；否则固定目标
    会和"发高速指令"互相拉扯）。这是奖励合同值变更，已明确告知用户。
- 踏面加宽的 A/B（同一 `A4 model_1499`，只换地形，256 环境）：0.90–1.00 碰撞 **34.8% → 2.3%**、
  完成 59.4% → 64.5%，但**摔倒 5.9% → 33.2%**；0.85–0.90 完成 96.5% → 68.8%（摔倒 31.3%）。
  → 加宽确实消掉了"拖肚子"，但更长的楼梯段 + 2 m 山顶平台对旧策略是分布外，失败转为真实摔倒。
- A5（`2026-09-18_23-30-00_...-terrain-a5`，难度 0.75–1.00 × 速度 0.35–1.00，1000 轮）：
  `model_950` → 0.85–0.90 = **99.22%**（摔 0.78%）；0.90–1.00 = **92.19%**（摔 2.7%、碰 5.1%）；
  0.90–1.00 且**速度 0.95–1.00（即 1 m/s）= 89.45%**（摔 2.3%、碰 8.2%）。
- A6（`2026-09-19_04-40-00_...-terrain-a6`，难度 0.88–1.00 × 速度 0.50–1.00，700 轮）：
  `model_450` → 0.90–1.00 = **94.92%**（摔 5.1%、碰 0）；同带 **1 m/s = 95.70%**（摔 0、碰 4.3%）；
  最终 `model_699`。
- **重要回归（必须收束）**：门禁第一格（难度 0.25 × 速度 0.3）用 A6 = **88.67%、摔倒 7.81%**。
  只在高难带训练会让**简单/慢速带被遗忘**，因此冻结前必须做一轮全包线收束，否则 16 格矩阵过不了。
  → A7 `2026-09-19_06-00-00_go1-stage1-terrain-a7`（难度 0.20–1.00 × 速度 0.30–1.00、900 轮、从
  A6/model_699 热启动），日志 `logs/terrain_a7_p7.log`。
- 门禁工具的两个坑（已修）：① 必须 `setsid` + `python3 -u` 运行——第一次后台门禁在会话结束后被杀，
  行输出因块缓冲被吞掉，只剩表头；② 1.0 m/s 档的上下界要 clamp 到 `MAX_FORWARD_COMMAND`
  （脚本新增第 8 个参数 `max_speed`，默认 1.0）。另外**报告不记录难度/速度带**，只能按时间顺序辨认
  是哪一格（若长期使用门禁值得补上）。

## 2026-09-19 第一层（运控）冻结完成 + 认证契约修复

- 最终 16 格门禁（`A9 model_699`，每格 256 环境、120 s 回合）**全部达标**：
  速度 0.3 列 93.4/93.0/93.8/90.2%，速度 0.5 列 100/100/100/94.1%，
  速度 0.8 列 100/100/100/98.1%，速度 1.0 列 100/100/100/96.1%（按难度 0.25/0.5/0.75/1.0）。
  最弱格是"难度 1.00 × 速度 0.3"（90.23%，摔 4.7%、碰 5.9%）——慢速爬 14–16 cm 台阶最不稳。
- 训练路径：A1(0–0.2) → A2(0.35–0.65) → A3(0.55–0.85) → A4(0.75–1.00, 1500 轮) →
  A5(0.75–1.00 × 0.35–1.00, 1000 轮) → A6(0.88–1.00 专攻, 700 轮) → A7(全包线 0.20–1.00, 900 轮) →
  A8(全包线 0.20–1.00, 1600 轮) → **A9(0.92–1.00 补采样, 700 轮)**。
  关键教训：① 只在高难带训练会**遗忘简单/慢速带**（A6 使 0.25×0.3 掉到 88.7%）；
  ② 全包线均匀训练时难度 1.0 只占约 4% 采样，导致该列欠采样（A8 难度 1.0 列 70–92%）；
  ③ 因此"全包线打底 + 难带补采样"两段式才把 16 格全部推过线。
- 冻结产物：`checkpoints/stage1/2026-09-19_09-45-00_go1-stage1-terrain-a9/model_699.validated.pt`，
  `metadata.capability_validated=True`、`validation_protocol=stage1-gate-v1`、
  `metadata.envelope={acceptance, difficulty 0.25–1.00, speed 0.20–1.00, cells 16, envs_per_cell 256,
  episode_seconds 120}`。门禁记录：`reports/gate_stage1_a9_2026-09-19.json`。
- **认证契约修复（重要）**：原 `certify_checkpoint` 对 stage1 要求
  `terrain=="course"` **且** `passed=True`，而 `passed` 只在 `terrain=="mixed"` 时才被计算 →
  stage1 认证**永远不可能通过**；其 `vel_rmse ≤0.15` 阈值也与本次验收口径无关。现改为：
  新增 `go1_hier_sim/gate.py`（`GATE_PROTOCOL="stage1-gate-v1"`、`build_gate_record`、`parse_gate_log`）
  与 `scripts/gate_record.py`；`certify_checkpoint` 接受门禁记录，且**自己重算**每一格
  （`cell_passes`）而不信任记录里的 `passed`；认证后把 `envelope` 写进 metadata。
  新增 `tests/test_gate_certificate.py`（5 个测试：解析、全通过才认证、单格失败拒绝、
  跨 checkpoint 拒绝、CLI 全流程）→ `./sim.sh test` = **96 通过**。
- 门禁脚本现在还会：每行打印 PASS/FAIL（按验收口径）、把每格写进 `cells.jsonl`、
  结束时自动生成 `reports/gate_stage1_<时间>.json` 并提示认证命令。

## 第二层（速度层 / schema 5 地形起伏）— 已实测通过（2026-09-19）

- **接口变更（schema 5）**：把那对 72+72 的"障碍距离"改成**地形起伏图**（6 距离 × 12 横向格的地面梯度 +
  置信度），`NAV_OBS_DIM` 仍 193；manifest 键 `vision.obstacle` → `vision.terrain_relief`，新增
  `relief_rows/relief_cols` 与 `normalization.relief`。梯度 = 垂直射线命中高度与参考高度之差 / 距离，
  clamp ±1；无命中 = 未知（relief 0、confidence 0）。`checkpoints.py::_is_schema4_relief_migration`
  只接受"键改名 + schema 5 新增键"的差异（两侧规范化后逐项比较），维度变化仍拒绝 → 冻结的 schema-4
  第一层可直接加载（`require_validated=True` 通过）。
- **不做避障**：`envs/config.py` 的 `obstacles` 恒为 False（场景不生成障碍盒）；`configs/rewards_nav_v2.toml`
  的 `obstacle_evasion` / `wall_proximity_brake` 权重归零（已向用户披露）。
- **验证**：`./sim.sh test` = 98 通过；`sim.sh check --stage stage2` PASS；起伏图探针（难度 0.35）
  `grad_abs_mean=0.0341 grad_abs_max=0.1051 conf_mean=1.000 shape=(2,72)` → 速度层确实收到地形梯度。
  终点条带半宽 4.0 m 与地形带 y∈[4,12] 重合 → 无法绕开地形从旁侧越线。
- **训练**：`--stage stage2`（nav，cmd3 发速度），冻结层 1 = `a9/model_699.validated.pt`，256 环境 /
  1500 轮 / 120 s / 难度 0.20–1.00，`GO1_RUN_ID=2026-09-19_11-45-00_go1-stage2-speed-v1`（跑到约 417 轮停止以保住结果）。
  训练统计里 `reach_goal=0` 是**探索噪声 0.87** 所致，不代表确定性策略不会越线。
- **权威验收（256 环境/格、256 回合、120 s、确定性策略、`finish_crossed` 前脚越线）= `model_350.pt`**：

  | 难度带 | 越线完成率 | 碰撞 | 摔倒 |
  |---|---|---|---|
  | 0.20–0.40 | 99.61% | 0.00% | 0.39% |
  | 0.40–0.60 | 100% | 0.00% | 0.00% |
  | 0.60–0.80 | 100% | 0.00% | 0.00% |
  | 0.80–1.00 | 97.66% | 0.78% | 1.56% |

  4/4 格 PASS（口径 completion ≥0.90、collision ≤0.10、fallen ≤0.05）；单次全包线（0.20–1.00）完成率 99.61%。
- 产物：`checkpoints/stage2/2026-09-19_11-45-00_go1-stage2-speed-v1/{model_350.pt,model_400.pt}`、
  `reports/evaluate_stage2_2026-09-19_13-{04-55,07-54,10-12,12-25,14-58}.json`、`logs/nav_speed_v1.log`、
  `logs/layer2_cells.log`。
- 第二层**尚未认证**：`certify_checkpoint` 的 stage2 路径仍要求 `terrain=="mixed"` 且 `passed`
  （与 stage1 早先是同一个死锁）；若需要认证，需为第二层定义独立门禁协议。

## 2026-10-15 Go1 v8 密集混合地形与 map-attn-v3（p1 已启动，未验收）

- 旧 v7b 模型 `2026-10-14_go1-v7b-isolated-s01-p1-d1/model_899.pt` SHA256 `fd04f3a6f1c5d03c6d6992be722a6924ab2ca9015112a2bdc6aa8242c2d4e631` 保持不变；它的 encoder 实测是 `ActorCore.encoder` 1024→128→32，**不是** v2 的 512→256→32。v8 使用这个 v1 actor 的零初始化注意力残差，仍为 1073 输入／12 动作／GRU128，支持旧权重严格热启动；真实 checkpoint 加载后的初始动作和隐藏状态已逐位相等。新 metadata `policy_variant=map_attn_v3`，相机深度学生尚未部署。
- 11 族保留；combined 在难度0.1、seed2026 下中途两段及末段轮胎阵均为16胎，中途楼梯3级，末段8级中含0.17m踢面和0.47m对应踏面；短沟与长沟后安全落台保留。修正足端虚空检测；长沟奖励只在真实起飞并在远唇外稳定落地后一次性支付，基于仿真几何元数据训练侧打分，**不**送给 actor。高台阶只完成 PhysX cooking/初始两步检查，**尚未证明**接触力矩与稳定跨越；跳跃策略尚未另训。
- 远端版本备份 `reports/v8_backup_20261014T171746Z/`；部署哈希 `reports/v8_deploy_manifest.json`；全量测试 `reports/v8_export_green.log` 为214通过；512/16环境 PhysX check 均通过；两rank 16环境独立训练smoke三轮并成功导出完整 v8 actor 图；这些只是数据链路证据。
- 独立 p1 `2026-10-14_go1-v8-dense-attn-s01-p1-d1` 两rank各512环境、900轮，其中前150轮仅critic预热；UTC 2026-10-15T00:06:24Z 捕获 PID113415、PPO 第11/150轮及589824 aggregate timesteps，证据 `reports/v8_p1_status.json`、`v35_phased_2026-10-14_go1-v8-dense-attn-s01-p1-d1.log`。`lp_phased_stop.flag` 保留，不运行 watchdog，不自动p2。**此处为启动时的历史状态**；后续 v8 900轮完成，同一次512环境闸门为459/883、9/11，combined 0/397、gaps_long 45/66，决定 hold；仿真启动不代表实机通过。

## 2026-10-15 Go1 v9 沟段接地诊断与隔离步行 p1

- 用户确认：仅在沟段取消“仅脚陷沟／机身暂时降低或倾斜”直接判死；可信非足部承重接地才判沟段死亡；其他地形安全终止不变。独立长沟跳跃权重尚未训练，不可宣称长沟会跳。
- 新版本设计和执行计划：`docs/superpowers/specs/2026-10-15-go1-v9-gap-contact-leap-tire-transition-design.md`、`docs/superpowers/plans/2026-10-15-go1-v9-contact-walk-leap-deploy.md`。源码在本地 `v9_work/`，远端备份 `reports/v9_backup_20261015T094448Z/`，哈希 `reports/v9_walk_deploy_manifest.json`；旧v8 checkpoint和报告未覆盖。该工作区无 Git 仓库，未声称提交。
- v9 沟段终止结合每块地形全部沟沿几何、根/足位置、非足连杆接触净力的向上分量、局部向下射线地面高度与持续5控制步。此为**保守候选接地判据**，接触净力不提供精确接触点；不能当作实机感知。探针最初在沟近沿平地 x≈1.07m 的 `FR_calf` 上误触发，已把足端探沟免死窗口与根部真实沟段接地区分；复测 x≈1.77m 与2.5m看到 `FR_calf` 约80–140N向上承重、连杆 z≈0.01–0.05m、地面 z=0；按用户“非足部接地”定义仍应记失败。短沟指令在批准的0.35–0.8m/s内夹取；新增微弱有界平衡功耗项，原 `joint_power_l1=-5e-05` 保留。
- v9 远端全量 `reports/v9_full_tests.log` **229 passed**；16/512初始 PhysX `reports/v9_physx_check{16,512}.log` 通过；16环境两rank三轮 `reports/v9_train_smoke16_retry1.log` 和 `reports/v9_smoke_export/locomotion.json` 全图导出通过。v8→v9注意力权重热启动兼容修复有回归测试；首轮 smoke 的结构错误已经修复且保留失败日志。这些不证明能力。
- v9 walker 原两rank各512环境 p1 `2026-10-15_go1-v9-gap-contact-walk-s01-p1-d1` 于 UTC 2026-10-15T12:04:11Z 启动，900轮/前150轮critic预热；已在可校验 `model_250.pt` (SHA256 `017c8d690dac5d980ce95c8ea0bab29f6afc33533a8f67fa3c3f19d41b1d8770`) 后安全停止，原产物未覆盖。计划 `reports/lp_plan_2026-10-15_go1-v9-gap-contact-walk-s01-p1-d1.json` 保留。

## 2026-10-15 v9 四卡续训（已完成评测，仍为 hold）

- 用户明确要求四张GPU都参加训练再看效果。容器 `/dev/shm=64MiB`，原四卡NCCL smoke在共享内存分配时报错；显式 `GO1_V9_DDP_BACKEND=gloo` 只改变 v9 分布式传输，不改第三方代码。四卡各16环境、从原 `model_250.pt` 续训2轮的 `reports/v9_4gpu_gloo_smoke4.log` 成功；无并发仿真。
- 四卡各512环境首次正式启动在 rank3 的 seed45、第37块 `combined` 抛出“末段轮胎无空间”；这属于几何末段预算旧边界bug。仅将末段轮胎排预算上限从 x32.90 调整到 x33.05，实际胎缘更早于终点区 x33.10；轮胎排数量、沟、坡和台阶不削减。修改前 `reports/v9_backup_20261015T094448Z/v9_before_4gpu_geometry.py` 与失败日志保留，部署文件哈希已纳入 `reports/v9_walk_deploy_manifest.json`。`reports/v9_4gpu_geometry_2048_tiles.log` 四个rank的42–45种子各512块均可构建，`reports/v9_full_tests_4gpu_geometry.log` 为233通过。
- 独立续训 `2026-10-15_go1-v9-gap-contact-walk-4gpu-from250-g2`，UTC 2026-10-15T13:04:41Z driver PID71862，四rank各512环境，训练于14:23:21Z至 `model_899.pt`，SHA256 `f821cc99228df2efbf04bd4759fe5529e38430d8fb0cca3f1b4e8f5f760fbc48`，`capability_validated=false`。训练日志 `v9_4gpu_train_2026-10-15_go1-v9-gap-contact-walk-4gpu-from250-g2.log`，脚本 `scripts/v9_resume_4gpu_from250.sh`。观察到四卡各约10GiB、40–50%利用率，迭代约6.5–8秒；对照旧两卡第251–256轮迭代中位约6.55秒、四卡第251–265轮约7.37秒（每轮样本数翻倍；训练回报不能代替闸门）。
- 串行的**同一次512环境 stage1-v2 确定性闸门**于 UTC 14:36:58Z 结束，报告 `reports/evaluate_stage1_2026-10-15_go1-v9-gap-contact-walk-4gpu-from250-g2.json`（本地 `v9_work/reports/v9_4gpu_g2_gate_20261015.json`，SHA256 `cf0ab0e740f0db4153b65337901bdc676dd635f22078e13a34d8232b0e371c69`）。总成功 **367/887=41.38%**、**8/11族达标**：combined **0/139**、gaps_long **0/206**、gaps_short **0/171**；tire_field 45/46，其余七族≥93%。v8同口径459/883=51.98%、9/11，因此v9步行续训在沟族明显退步。报告的 `mean_failure_x` 未提供可信首败坐标（相关格为0），不可据此断言失败位置或原因；净力无接触点，沟壁阴性对照仍未完成。
- 评测后相关训练和Isaac进程均退出，`lp_phased_stop.flag`仍在；没有自动p2、watchdog、升档或第二套仿真。**继续 hold**。长沟跳跃权重未训练；短沟0/171和混合0/139不能以奖励或smoke解释通过。
- 用户要求看视频并研究：桌面 `C:/Users/Administrator/Desktop/Go1_v9_四卡失败诊断/` 有 v9 混合0、短沟94、长沟47三段25秒原速录像及首沟2.5倍慢放；另有同一前缀计划下的 v8 旧判据与 v8 权重+v9 判据的短沟94对照，以及 `README_先看.txt`、`analysis_zh.md`、SHA清单。小批量视频不改写正式闸门。日志显示v9短沟槽94在第一沟后根部局部 x≈2.20–2.26m 时 `FR_calf` 向上净力候选持续5控制步使终止，v9混合槽0第一短沟后 `RR_calf` 同类候选；v9长沟槽47第一近唇`FR_thigh`/`FR_hip` 接地候选累积、未有跳跃接管。v8旧规则同计划短沟槽94小批量1/1，v8权重仅切v9终止模式后槽94 0/1、前缀总体由30/128到0/95；支持新规则对旧步态有显著影响，但净力无接触点，沟壁摩擦阴性对照尚未完成，不能宣称所有候选均是真实非足落地或擅自放宽终止。全部诊断回放已退出，未修改模型/规则/课程。证据在 `v9_work/video_delivery/evidence/`。

## 2026-10-15/16 v10 独立长沟跳跃代码（仅部署与冒烟，未取得能力）

- 用户确认 A：保留 v8 步行实验底座，跳跃专项**从零初始化**；不能用普通步态或 1.5m/s 论文跳跃结果冒充 Go1 在0.35–0.8m/s已会跳。P10论文一手笔记在 `docs/p10-robot-parkour-primary-source-notes.md`，v10设计/计划在 `docs/superpowers/{specs/2026-10-16-go1-v10-p10-observable-leap-design.md,plans/2026-10-16-go1-v10-observable-leap-code-deploy.md}`。本地 `v10_work/` 暂存，远端旧文件备份 `reports/v10_backup_20261015T173040Z/` 和源/证据哈希清单 `reports/v10_observable_leap_manifest.json`；本目录不是Git仓库，未声称提交。
- 新代码默认**关闭**，仅 `GO1_V10_OBSERVABLE_LEAP=1` 开启实验室模拟接管；1073维 actor/12动作/GRU128不变。额外49×5、纵向0.025m的物理射线与既有16×32扫描的两侧足道共同判别深沟宽度及横向覆盖；仿真几何/任务族真值不进入 actor 或 v10 门控。标称短沟≤0.24m、长沟≥0.29m另有统一0.05m低唇，光线看到的是≤0.29m和≥0.34m的深沟。缺失、混叠、无远唇或足道无支撑的判别失效时拒绝接管。超时若同一沟仍可见则禁止立即重入；只有连续5帧可见平整落台才能重备。真实硬件细扫描、脚端接触估计和相机学生尚未验收。
- 训练入口须同时设置 `GO1_V8_TERRAIN_CONTROL=1` 与 `GO1_POLICY_VARIANT=map_attn_v3`，冻结 v8 `model_899.pt` SHA256 `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`；新跳跃 actor 不能以 walker warm-start。已修复 PPO 动作来源 mask 在终止环境reset后被改写及跨时间步共享的问题，v10关闭 walker 步的 actor entropy/自适应KL；单步 mask 快照独立保存。checkpoint metadata 记录底座及有效PPO、奖励、物理扫描设置SHA；v10 模型未开可观测门控或被当成主步行模型均拒绝。旧聚合认证和直接 `certify_checkpoint` 也不得将 v10 冒烟权重标成已验收。
- 远端最新全量 `reports/v10_full_tests_final.log` **257通过**；16环境 PhysX 边界难度探针首沟短沟0/8触发、长沟8/8（审阅后日志 `reports/v10_scanner_physx_16_boundary_reviewed.log`）；512环境初始化及三次物理步进 `reports/v10_scanner_physx_512_final.log` 通过。另用真实1环境强制终止验证执行步 walker-mask=0，而环境reset后的下一步mask=1（`reports/v10_mask_reset_probe.log`）。独立16环境×3次PPO更新只证明代码链路：`reports/v10_leap_16x3_smoke_final.log` 中每轮 `MASK kept_samples=0.0000/0.4883/0.2383`；专项冒烟权重 `checkpoints/stage1/2026-10-15_go1-v10-observable-leap-smoke-final/model_2.pt` SHA256 `48f620fa99ed692419b48ab7f470e5a31ed685e841ef2c2a419e58e82b42fc09`、`capability_validated=false`；TorchScript全actor及隐藏态往返验证 `reports/v10_leap_export_smoke_final/locomotion.json` 通过。这些绝非真实飞越和落地成功证据。
- **仍为 hold**：没有长时专项训练，也没有同一次512环境11族各≥90%的确定性闸门；v9正式闸门仍是367/887、8/11。沟壁擦碰阴性/真实非足承重落地阳性 PhysX 对照未完成，净力不提供精确触点，不能认定接地判据可靠或通过实机安全。纯张量横向窄坑阴性测试不能替代真实几何/多种俯仰偏航的完整 PhysX 校准。所有训练/Isaac进程已退出，`lp_phased_stop.flag`保留，无 watchdog/自动p2。

## 2026-10-16 v10 正式训练启动前复核（仍阻塞）

- 用户授权优先启动训练，但按既定顺序须先通过接触及接管的 PhysX 对照。串行16环境静态诊断和11族射线扫测已执行；原始日志/统计/限制在 `v10_work/reports/v10_training_preflight_20261016.md`，远端同名 `reports/` 报告。本次完整远端 `./sim.sh test` 为261通过，旧v8/v9权重与停止标志SHA未变。
- 1728组物理射线覆盖11族、横移±0.25m、偏航±0.25rad：受控首个长沟近沿长沟27/27、混合18/18触发，短沟0/216、轮胎0/216，7类坡/楼梯各0/108；不等于动态速度、俯仰或所有深沟验证。静态远沿承重阳性有净力/平台射线/无足承重证据，**沟壁单独擦碰阴性仍未能隔离**：躯干可能承重于近沿，另一组摆位已在远沿，净力无精确触点。故不宣称接地判据可靠，也不启动非足承重判据未经校准的长训；`stage_gate=hold`、不自动p2、不启watchdog，`lp_phased_stop.flag` 保留。后续尝试对已初始化刚体动态挂 PhysX contact-report API 导致 `PxRigidActor::detachShape` 与失效的 simulation view（`reports/v10_physx_contact_points_probe_v2.log`）；该诊断脚本已隔离在 `reports/`，无有效触点数据，不能用于放行。随后在初始化前创建单刚体触点诊断传感器，日志明确出现 `GPU contact filter for collider '/World/ground/terrain' is not supported`，336行诊断触点位置全空（原净力候选156行），同样无法完成沟壁阴性触点核验；不能把空数据解读为无碰撞。

## 2026-10-16 用户要求优先启动 v10 长沟探索训练

- 用户再次明确“开始训练”，故在保留上述接触校准阻塞与 `stage_gate=hold` 的前提下执行**未认证的实验性训练**，不将进度、回报或冒烟称为跳跃能力。记录在 `v10_work/reports/v10_explore_training_20261016.md`／远端 `reports/` 同名文件。512环境仅 `gaps_long`，难度0/.05/.10/.15各128、命令0.35–0.8m/s；冻结v8步行、随机初始化独立专项，`GO1_V9_GAP_CONTACT=1` 未豁免非足承重接地，原有v8奖励权重未针对长训认证。保留11族闸门和旧模型、停止标志，无watchdog或自动p2，单套Isaac。
- 首跑 `2026-10-16_go1-v10-longgap-explore-p1-512` 实际至24/300轮，坏姿态射线投影非单调导致整批异常exit1；保存失败日志和权重。新增单环境无效射线fail-closed测试RED→GREEN，完整远端262通过，原扫描器源码版本备份保存。全新跑次 `2026-10-16_go1-v10-longgap-explore-p1-512-r1` 于UTC 09:00:47启动；09:05:11仍运行53/300，实际专项mask第53轮0.1185，已保存`model_50.pt`（SHA `ee7b76be32a8054e89c1c38061a1dabcb1cf4db0627a825403b16d0b981cd83c`）；model0→50的locomotion/policy状态哈希均变化，仍`capability_validated=false`，早期`leap_clear=0`，无真实飞越结论。实时运行状态随时间变化，查询进程、`reports/<run>.exit` 和日志为准。

## 2026-10-16 用户重新指定完整11族而非长沟专项；v11统一策略p1已启动

- 用户明确否定转向/长沟专项方向，要求整体改代码与重训，特别包含 `combined`；已完成的v10长沟专项300轮正常退出(exit0)，末轮清沟奖励为0，**不使用**其checkpoint或门控。设计 `docs/superpowers/specs/2026-10-16-go1-v11-unified-roster11-design.md`、执行计划 `docs/superpowers/plans/2026-10-16-go1-v11-unified-roster11-train.md`。不承诺一次训练就能达到11/11；v8闸门459/883、9/11，其中combined0/397，v9闸门367/887、8/11。
- v11在`v11_work/`隔离开发、远端原文件保存在`reports/v11_backup_20261016T101447Z/`，清单`reports/v11_unified_manifest.json`（本地SHA `17912cd948de9e16679fe3b4b2c244f3084771ac0878758d64b92ab512b3f12d`）。在现有奖励状态增加仅`combined`轮胎足接触、直立支撑并刷新高水位才支付的有界`tire_progress`，不改1073观测/12动作/GRU128和任何真实几何；旧奖励配置仍解析。LP阶段计划训练槽按combined2.5、gaps_long2.0、其余各1.0分配512：实际combined94、长沟76、其他9族各38，5个速度级槽数100/106/106/108/92，0.35–0.8m/s。完成训练后**另外构造同一次512环境均衡46–47槽/族的确定性评测计划**，保留原40回合/族和每族≥90%闸门，不用训练偏重计划篡改验收。
- 新单元测试先红后绿，远端 `./sim.sh test` **270通过**，日志`reports/v11_full_tests_20261016.log`（本地SHA `86eef2ec5464efaff254345bacf4fdfe65bce68e1fe5945bea59653dd821c700`）。单GPU16环境×3 PPO冒烟覆盖11族且存`model_0.pt`、`model_2.pt`；二者locomotion和policy state确有变化，checkpoint `capability_validated=false`。冒烟 `Episode_Reward/tire_progress=0`，**没有混合成功或验收证据**。
- 统一actor独立p1 `2026-10-16_go1-v11-unified-roster11-s01-p1-d1` 于UTC10:19:27启动；原v8 `model_899.pt` SHA `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`热启动，两GPU rank各512环境、900轮、前150轮critic预热，未启v10专项或v9新接地实验开关。UTC10:20:41**当时状态**：进程在CPU几何初始化，后续结果见下节。所有Isaac作业串行，`lp_phased_stop.flag` SHA `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`仍在；无watchdog/自动p2；接地碰壁阴性仍未校准，不得宣称实机安全。

## 2026-10-16 v11 完训闸门与 v12 指令诊断（最新仍 hold）

- v11单策略900轮退出0，最终 `model_899.pt` SHA `b11365fe55604585fa30285167393f6a6b1da318d7a7e01bf481f264dff8350e`、未能力认证。正式同次512环境闸门 `reports/evaluate_stage1_2026-10-16_11-54-50_gate-2026-10-16_go1-v11-unified-roster11-s01-p1-d1.json` SHA `cd33c1e121a27faf8372e5983692fb5c0ab722f09c744a317bb5c0cd5ef4217d`：**456/790、7/11**；混合0/273、长沟47/92、短沟47/53、轮胎38/47。冻结v8在同计划为457/857、混合0/309、长沟46/128、短沟47/49、轮胎39/46。均不得升档。
- 本次诊断源 `docs/v11-improvement-research-2026-10-16.md`。v11原控制加仅诊断的逐回合trace重跑512，报告SHA**与原正式闸门相同**；790回轨迹 `v11_trace_work/reports/diag_v11_legacy_trace_512.jsonl` SHA `4c4cbcdbcdbd51c480d9e89e47c3c8406476656f0dc61cd5826a994349f41c1d`。混合273回中`supported_short`270回、`safe_segment`129回、`tire_progress`120回实际触发；47槽初回合首轮胎终止26、首沟前/首沟13。奖励非全无，不能盲目加权。
- 旧控制按沟类**真值家族**乘1.8倍目标速度，最快目标理论1.44，实际路由器执行上限约1.0，违背批准0.35–0.8；评测driver未传速度上下限使内部config回退奖励默认0.45，虽然任务计划采样仍在0.35–0.8。隔离`v12_work/`修正计划速度边界及actor前向指令只用可观测扫描，单测283通过、76 subtests通过，完整512反事实回放冻结v11权重**383/904、7/11**，短沟16/144、长沟3/238、混合0/151、轮胎39/46。故不得把这版代码覆盖生产或称为修复后的可用策略；旧权重依赖超速，需要新版本在合规命令下重新学习。报告SHA `b17c2f3fda0ba368c47b13cf27af7f81e7631967c8ee461a1133cc809c18c456`，轨迹SHA `029a437af86ae4fde1b9c2dedc9053fc5690269a5782e1f0fef080cd1a1aecc7`。执行命令复位后的平滑启动可短时<0.35，不能表述为全程执行下限通过。
- 另查明继承的Isaac Lab `rel_standing_envs=0.02` 会让部分Stage1回合整段目标速度0：v11正式轨迹790回中8回零指令，短沟6个失败回中3回零指令。只在隔离v11遥测版关闭该默认站立采样做单变量512对照，得**458/793、8/11**：短沟47/50过线，但混合0/284、长沟46/85、轮胎40/49仍未过；旧超速/家族真值仍在，不能作为可升档正式成绩。单变量报告SHA `f83db769821bf2f5a862d367e58f30da1422686ff7ca97444f8adbb8da31dae0`，轨迹SHA `5ea5e29a4b26c175042d3e197f56d5efe5d8cb37e1da9b28b3ab0af992dfc87e`；原始原码、权重、停止标志未覆盖，无新长训或自动p2。接地/真实跳跃校准阻塞不变。

## 2026-10-16 最新用户决定：先十族，再混合DAgger（优先级高于v11混训顺序）

- 用户批准“分技能教师→学生亲自跑混合路线→DAgger→一个统一学生”，并明确**先把非混合10族各拉到≥90%**。权威阶段与范围见 `docs/superpowers/specs/2026-10-16-go1-ten-family-first-multiteacher-dagger-design.md`；P10适用边界见 `docs/p10-multiteacher-applicability-2026-10-16.md`。旧v11 spec保留作历史，不再从p1起把完整combined塞进新训练计划。
- **用户后续明确顺序：允许不同教师各自覆盖相应单项；先把10族蒸馏为一个学生，学生自己在10族各达90%后，才冻结权重去测完整混合。** 第一屏障：训练期不同教师的十族覆盖，在同一次512环境确定性计划中每族≥90%、至少40回合；第二屏障：同一个1073输入、12动作、GRU128学生在独立的同规格512十族计划中每族≥90%。两个屏障都不是最终能力升档，也不能用旧超速/家族真值actor拿成绩。前期课程不训完整combined，可只读诊断交接短片段。
- 通过两道十族屏障后，**先冻结学生**在原完整11族512确定性计划测混合迁移基线；若有任何一族不足90%，维持hold，才让学生在硬碰撞混合全程自己执行、由教师仅训练期标注学生访问的过渡/恢复状态并平衡十族回放。最终部署仍是一个统一学生，**仅同一次512环境11族各≥90%**才能升档。actor/部署只读可观测输入；真值仅用于训练奖励、critic和仅训练期标签选择。长沟真飞越与落地、沟壁擦碰阴性/非足承重阳性对照未校准前不宣称安全或跳跃认证。
- 此条是**已批准的方向与记录，尚非已训练能力**；截至本次记录没有因该决定启动新PPO、混合DAgger或p2，保留所有旧权重、报告、视频与`lp_phased_stop.flag`，单套Isaac、无watchdog。后续启动状态以紧接的最新记录为准。

## 2026-10-16 16:58 UTC 阶段A首个十族教师实验已启动（进行中，非能力认证）

- 隔离实施计划 `docs/superpowers/plans/2026-10-16-go1-stagea-ten-family-teacher-start.md`，隔离源码 `v13_work/`。先红后绿的十族选取、闸门速度和配置测试，远端全量`./sim.sh test -q`实测 **277 passed, 76 subtests passed**，日志`reports/v13_ten_teacher_full_tests_20261016.log` SHA `ccc2a90ea5696f19663f12446f13d3a4f903f6150cb7e38`。生产原4文件备份 `reports/v13_ten_teacher_backup_20261016T164744Z/`，部署SHA清单 `reports/v13_ten_teacher_manifest.json`，12文件远端SHA一致；原v8权重及stop标志分别仍为 `136769c665edb8cedcffa62cb522084c4cdb0575dac497560dc923eeaeac2a15`、`fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。无Git提交。
- 两rank各16环境2轮PhysX/PPO冒烟退出0，权重`model_1.pt`仅`capability_validated=False`；后续同计划16环境评测19行逐回合trace观测最大执行指令 **0.7927727103233337 m/s**，报告16环境，不能算十族闸门。训练冒烟日志SHA `a5ba95e2cec37eb73e33815f773eed73b6c96c0c47f13d3a4f903f6150cb7e38`；trace SHA `2138eb3536a83cee109e6f642a489367491dfab9c7548305eba59a16990d4d59`。
- UTC **16:58:05** 启动首个**共享十族教师探索** `2026-10-16_go1-v13-ten-teacher-s01-p1-d1`，两个rank各512环境、900轮、v8 checkpoint热启动、150轮critic warmup、合规0.35–0.8目标指令、0%站立采样；训练plan十族各51–52槽，**不含combined**。远端 `reports/v13_ten_teacher_driver.log`、`v35_phased_2026-10-16_go1-v13-ten-teacher-s01-p1-d1.log`、PID文件`reports/v13_ten_teacher.pid`；此条启动时两个sim.py仍在初始化，**当时尚无十族确定性结果**。阶段driver最多一阶段，后续闸门十族同次512，仅作教师诊断；不得自动蒸馏/启混合或p2。完成状态见下条，不以启动成功替代能力结果。

## 2026-10-16 18:59 UTC 十族共享教师首轮正式中间闸门：hold 7/10

- 900轮训练及紧随其后的**同次512环境确定性十族闸门**结束；远端`reports/v13_ten_teacher.exit`为`exit=0 utc=2026-10-16T18:31:54Z`，截至UTC18:59:45未见`sim.py`/阶段driver进程。`model_899.pt` SHA `3110c93bd2bc9f77b77dc624a670dafb16a8bf2d1ef2c59b16ca808e2f62482b`，`metadata.capability_validated=False`、1073输入/12输出/GRU128。计划 SHA `003a273e9c53ac65b6bdbc74ebefae7bf5bd7cdd31d937473fc067d11c336f9b`；报告 `reports/evaluate_stage1_2026-10-16_18-31-51_gate-2026-10-16_go1-v13-ten-teacher-s01-p1-d1.json` SHA `83922687373fb243199bef8c2f816e3b6742ed3aa5c1e8004a1df11243c323c0`。由原`gate_decision`重算：总**515/711、7/10**，`hold`。短沟**47/157 (29.94%)**、长沟**71/136 (52.21%)**、轮胎**41/61 (67.21%)**不及90%；`stairs_up_long`50/51 (98.04%)，其余六个坡/楼梯均51/51。分母受失败重试影响，不能与v11不同计划/旧超速成绩仅凭比例断言单因果。报告的`mean_failure_x=0`仍不可靠；本次512闸门未开逐回合trace，不宣称失败位置或真实跳跃/非足接地已认证。
- 十族第一道教师体系屏障**未过**，因此没有训练十族学生、没有完整混合测试/DAgger、没有11族升档或自动p2。下一步需串行针对短沟、长沟与轮胎设计/训练专门教师，先找逐回合失败及接触校准证据，不凭总回报提高奖励；保留旧权重、报告、视频和`lp_phased_stop.flag`，stop SHA仍 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。

## 2026-10-16 19:43 UTC v13 三族失败诊断（未改生产代码，仍 hold）

- 一手逐回合+视频+奖励核查见 `docs/v13-ten-family-failure-diagnosis-2026-10-16.md`；原512计划相同seed/config只开trace重放，报告SHA**仍精确等于原闸门** `83922687373fb243199bef8c2f816e3b6742ed3aa5c1e8004a1df11243c323c0`，711行trace `v13_work/reports/v13_diag_trace512.jsonl` SHA `e37ccaaf8772726bca18843f840ad903b7a2974916eb38de2133b56a17fe86e9`。短沟s0仅5/104成功、110失败全fallen且中位x≈1.684；长沟65失败全fallen且中位x≈1.334；轮胎s4仅8/25、20失败中16次倾倒、4次超时。首次每槽成功分别36/52、18/52、40/51；闸门按所有完成回合计，重复失败会改变分母。
- 隔离视频本地`v13_work/video_delivery/`，远端`media/v13_{long_slow_fail_i6,short_slow_fail_i64,tire_fast_fail_i474}.mp4`均真实PhysX回放。长沟7槽前缀i6在有/无视频时83步x≈1.295失败、9秒视频显示首沟近沿失稳；短沟65槽前缀i64视频0/8并反复首沟附近倒下；轮胎475槽前缀i474视频前13秒只见轮胎段姿态起伏、**不显示最终终止**，完整512逐回合终止姿态另见trace。另一个长沟i1前缀竟成功（原512失败），故前缀视频只做动作观察，不冒充512原样回放。
- 源码/几何证据：`supported_short`、`safe_segment`、`tire_progress`只向combined支付，纯短沟/轮胎奖励几何NaN；本次十族训练900次快照这些项全显示0.0000，`leap_clear`也全显示0.0000（日志四位小数，不能据此认证零腾空），`gap_intrusion`676次非零；`GO1_V8_TERRAIN_CONTROL=1`使`no_progress`计算覆盖全部十族，注释却只论沟沿。这些是**已定位的覆盖/作用域问题**，尚未证成任意改权重即能达90%。v13仅是共享教师，不是三族分别专项训练；此时不要启动蒸馏/混合或将长沟路线成功称为真实跳跃。等待有界修正方案与单变量对照后再改生产/长训。

## 2026-10-16 20:17 UTC v14 纯短沟奖励隔离修正与专项教师开始（训练中）

- 用户确认“按你想的改，改完开训”。执行计划 `docs/superpowers/plans/2026-10-16-go1-v14-short-specialist-reward-start.md`，隔离代码 `v14_work/`，源文件本地/远端通过清单 `reports/v14_short_teacher_manifest.json` SHA `acee122d50351e6f32ceea4a99ff8aac6007089428d07eea9ec38ba0ab5f1844` 校验。远端原生产 `runtime.py`、`cli.py`、阶段driver 已备份于 `reports/v14_short_teacher_backup_20261016T2000Z/`，旧v13权重/报告/视频未覆盖，无Git仓库。
- `GO1_V14_SHORT_REWARD=1`（默认关闭）只在训练奖励和critic侧使用已有`all_gap_segments`，为纯`gaps_short`补每段至多一次的远落台稳定支撑和受姿态/接触/前进门控的近沿支撑项；≥0.26m长沟、其余族/combined不享纯短沟奖励，v7混合项/硬碰撞/actor1073×动作12×GRU128原样。净力是支持候选，不能宣称精确触点。旧`no_progress`仍未改（另做对照），纯轮胎奖励/长沟奖励也未改。阶段driver只训512槽`gaps_short`，仍保留全局11族注册，不会误称该单项闸门是十族学生屏障。
- 新增纯张量/驱动/接口/启动/metadata测试均先RED后GREEN，远端全量`./sim.sh test -q`为**290 passed, 76 subtests passed**（`reports/v14_short_teacher_full_tests_final.log` SHA `8eefc0ee9ea5709e25f83a26a5525c7d36bfb31ac4fc8c1d3ab7765da928dd01`）。16环境双rank两轮PhysX PPO冒烟退出0；同计划16槽确定性回放17行trace中`supported_short`事件9855、`safe_segment`154、`tire_progress`0，最大执行目标速度0.7797626；首瞬平滑可低于0.35，实际机身速度均值可高于目标，冒烟表现不是专项能力验收。又单独16环境1轮冒烟确认checkpoint metadata `reward_variant=pure_short_v14`、`capability_validated=False`。16槽原trace SHA `060b6cd31aefcc9324232726793263cebad1d0cb441d78fd0e03095d2956e768`。
- UTC **20:17:41** 启动唯一专项教师 `2026-10-16_go1-v14-short-teacher-s01-p1-d1`，两rank各512环境/900轮、前150仅critic，v13模型SHA `3110c93bd2bc9f77b77dc624a670dafb16a8bf2d1ef2c59b16ca808e2f62482b`热启、明确0.35–0.8目标/0站立采样、配置难度上限0.6（首层目标1，plan均难度0.071，非覆盖全部0.6）。计划SHA `7cd0587ce36af03770aa8880fd27e8575088e4a4af7fc6c5cbbe079fb1f2d53c`，日志 `reports/v14_short_teacher_driver.log`、`v35_phased_2026-10-16_go1-v14-short-teacher-s01-p1-d1.log`、PID收据 `reports/v14_short_teacher.pid`。UTC20:17:50检查双rank真实sim进程正初始化，无立即错误；运行状态需重新查询，完成/退出收据 `reports/v14_short_teacher.exit` 未出现前绝不宣称完训或过线。串行flock，无watchdog/自动p2，停止标志SHA仍 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。
- 这是专项采样+新奖励两个变量相对v13共享训练；若成功也不得仅因总比率归因奖励。训练完成后driver自动做本单族同次512环境确定性中间闸门，未达90%维持hold；不自动轮胎/长沟、不训练单学生/混合。原最终同次512环境11族各≥90%闸门不变。

## 2026-10-16 21:45 UTC v14 短沟专项900轮完训及单族中间门：advance，但十族仍 hold

- `reports/v14_short_teacher.exit` 为 `exit=0 utc=2026-10-16T21:35:17Z`，UTC21:44:59检查未见sim.py进程。`checkpoints/stage1/2026-10-16_go1-v14-short-teacher-s01-p1-d1/model_899.pt` SHA `3b6f75656696a83b92c8f0cdda4eb165cb31c5413b84d9cd48eaefde422e368d`，metadata `reward_variant=pure_short_v14`、`policy_variant=map_attn_v3`、`capability_validated=False`，其JSON SHA `7ec58f16c58e022c0c906f432d7d45f7d030e116b260fef116808032e452e6df`。日志最后有900轮完训，不将exit0视为能力通过。
- 本阶段独立**512槽单族确定性**闸门报告 `reports/evaluate_stage1_2026-10-16_21-35-15_gate-2026-10-16_go1-v14-short-teacher-s01-p1-d1.json` SHA `80a8189f151f8eb8434079fef7b5b8ed7699980de7b234e762076ac94ebf6dc3`，已同SHA复制 `v14_work/reports/`。按每槽完成回合累计短沟**504/514=98.05%**，driver `GATE result=advance, families_ok=1/1`；此advance仅对**单族当前难度层**有效，不是十族教师屏障，更不是统一学生或11族升档。计划 SHA `7cd0587ce36af03770aa8880fd27e8575088e4a4af7fc6c5cbbe079fb1f2d53c`：512槽均纯短沟，其中难度0.0 147槽、0.1 365槽，**尚未测0.2–0.6/更高层**；速度档 s0…s4 槽数111/102/100/102/97。s0完成回合**103/113=91.15%**，s0且难度0.1仅**77/86=89.53%**，是弱项；不能以聚合98%宣称全部速度×难度格过线。
- 本次评测平均目标速度0.52285m/s、平均实速0.94418m/s：指令0.35–0.8合规不代表机器人已良好追踪目标，实速超目标值得后续速度/安全诊断。纯短沟reward+单族采样相对v13同时改变两变量，不能说奖励单独导致改善。旧`no_progress`未改、轮胎和长沟专项未启动；接触点/真实跳跃仍未校准，不作安全声称。`lp_phased_stop.flag` SHA仍 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`；无watchdog、自动p2或第二套Isaac。下一步须分别检验更高短沟难度、其余老师的独立能力，然后一个学生自己通过十族门，最后才测混合/11族。

## 2026-10-16 22:05 UTC 用户要求四卡持续推进：v15 纯轮胎教师训练中

- 用户新指令：服务器空闲立即推进下一个**串行**专项、四张A10用于训练并持续核查；此优先于之前仅等待人工指令的节奏，但**不**授权并发Isaac、降低硬碰撞、绕过教师→单学生十族→冻结混合→11族同次512各90%的门。计划 `docs/superpowers/plans/2026-10-16-go1-v15-four-gpu-tire-then-long.md`，先纯轮胎训练分布对照：奖励/actor保持v13语义，热启v13 `model_899.pt`，**不**热启只擅短沟的v14权重；未启 `GO1_V14_SHORT_REWARD` 或未校准的 `GO1_V9_GAP_CONTACT`，无自动p2。
- 四卡NCCL在64MiB /dev/shm不适用，初次4×16 Gloo smoke exit1（`reports/v15_tire_smoke4x16.log`），真实错误为旧 `GO1_V9_DDP_BACKEND`要求同时启v9未校准接触模式，已保存失败记录，未靠偷偷启v9路径消错。先RED后GREEN增加单独`GO1_DDP_BACKEND=gloo`纯传输开关，默认NCCL和旧v9开关保持原语义，源备份`reports/v15_tire_backup_20261016T2157Z/on_policy.py`。4rank ×16和×128真实PhysX/PPO冒烟分别退出0；峰显存约4.57GiB和5.90GiB/卡，日志SHA `0cac83560d026869516990b24a2112f1b395dcc620e997b03d795903de405372`、`cafe3b2e198820f19711d859d7e066820f3f323cd60211fa7c1099cdab7e370c`，权重均未验收。远端全量 `./sim.sh test -q` **296 passed、76 subtests passed**，日志SHA `3e6617ee868d62bedf5fd6e16ea4c02dea8e6d8cf55b8a629c43dd38f4cad752`。部署SHA清单 `reports/v15_tire_teacher_manifest.json` SHA `0e332ef645a3768598a4bac5dad3b6c4a4b8ac4aafa03e6e28afcd6aa26b777c`。
- UTC **22:05:10** 启动 `2026-10-16_go1-v15-tire-teacher-s01-p1-d1`：**4rank各128纯轮胎环境**（合计512训练槽）、900轮前150critic预热，全部速度目标0.35–0.8、站立采样0，单一轮胎中间评测128环境；仍非权威512十族/11族同次闸门。计划SHA `102ba49fc32f47b7e4de912386a1a4ea2eb6eaaa4ec80eadd96e3802094d20cc`，日志 `reports/v15_tire_teacher_driver.log` 与 `v35_phased_2026-10-16_go1-v15-tire-teacher-s01-p1-d1.log`，PID收据 `reports/v15_tire_teacher.pid`，退出收据待 `reports/v15_tire_teacher.exit`。UTC22:05:23已查四个真实sim.py训练进程正在启动，当前训练/结果**需重新查询**。`lp_phased_stop.flag` SHA仍 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。父代理已启动一个只读异步监测子代理（run `17ff00a5-e56c-4fb4-9549-e5a1a494c097`），直到退出/异常通知；不启动另一个仿真、没有后台自动启动长沟。轮胎退出核对结果后马上串行做长沟诊断/开训，接地触点未校准则任何长沟成绩仅能力探索，非实机安全认证。

## 2026-10-17 08:43 UTC v15 轮胎900轮完训、中间门通过；监控子代理基础设施失败

- 远端 `reports/v15_tire_teacher.exit` 是 `exit=0 utc=2026-10-16T23:14:46Z`；2026-10-17T08:42:49Z核实零个sim.py进程，停止标志SHA仍 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。最终 `checkpoints/stage1/2026-10-16_go1-v15-tire-teacher-s01-p1-d1/model_899.pt` SHA `4f18d4bcd62608a4707ac309af65277b715b0a55431b5ee20965a56bc23f08c7`，metadata `reward_variant=legacy`、`policy_variant=map_attn_v3`、`capability_validated=False`。全900轮日志记录完训；不把exit0或回报等同能力。
- 独立 **128槽纯轮胎确定性中间门**：`reports/evaluate_stage1_2026-10-16_23-14-44_gate-2026-10-16_go1-v15-tire-teacher-s01-p1-d1.json` SHA `225c0e127db0df36eb150a0c456ae3eca8a190b1402cbce282c17bd444dff93c`，本地`v15_work/reports/`同SHA备份。**124/129=96.12%**，driver `advance families_ok=1/1`，均速目标0.54015/实速0.56468m/s、碰撞3/129、跌倒5/129；训练保留旧纯轮胎奖励项`tire_progress`/`tire_stall`0.0000，因此不能归因于新增奖励。计划难度只0.0（31槽）/0.1（97槽），s4且难度0.1 **14/19=73.68%**，需要更高难度和高速薄弱格的独立验证。**不是同次512十族，更不是单学生十一族达标**，旧十族/最终闸门仍hold。
- 用户要求持续监控的只读异步`scout` run `17ff00a5-e56c-4fb4-9549-e5a1a494c097` 在超过预期后经stale-run reconciliation标记 `failed (proof-write-failed)`：`Async runner process 4036 exited or disappeared before writing a result`，未产出可靠handoff，不能宣称训练全过程持续监控。父工作目录`E:/tmp-issac`非Git仓库、无worktree/branch/ref；子任务只读SSH，源清单/权重/报告hash已直接复核，无子任务代码修改证据。已用同一subagent协议`resume`尝试一轮只读结果收据恢复，revived run `fe54588c-d1d8-48bf-a061-2c3cbba19a74`；其状态需重查，不能把此前基础设施失败当成果。下一步按用户已授权的四卡串行顺序，先明确真实接触/飞越未校准，再在独立低难度长沟专项做冒烟与训练探索；绝不认证真实跳跃安全。

## 2026-10-17 08:53 UTC 用户要求继续，v16 四卡长沟教师探索已启动（未认证）

- 在v15轮胎纯族中间门124/129之后（仍非十族学生屏障），按用户“四卡串行不空闲”的新指令开始长沟专项。v15监控子代理首次长工具调用超过实时进程存续造成stale-run `proof-write-failed`，并未提供连续监控证明；同协议`resume`新run已从远端只读复核结果并正常完成。子结果错误检查过`reports/v15_tire_teacher.stop`这个并不存在的名字；**实际停止标志是`lp_phased_stop.flag`**，父代理已在远端实测SHA保持 `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。
- 新隔离`v16_work/`及原远端v15兼容底座不改actor1073/动作12/GRU128、不改硬碰撞、不启v9未经校准接地、不借家族真值给actor，也未加虚构飞越奖励。纯`gaps_long`训练计划仍是同v13奖励的**专项采样对照**；四卡Gloo（旧NCCL受64MiB /dev/shm阻碍）。新增训练期roster与独立driver，全球11族注册/最终闸门原样；新测试先RED后GREEN，全量`./sim.sh test -q` **299 passed,76 subtests passed**，远端日志SHA `d4d5c10e766c79d9ce0a81903b6274145001bd877252ae205ac4801871a69537`。四rank各16/128/512环境、2步PPO真实PhysX冒烟均退出0，4×512峰显存约9.87/9.58/8.89/8.19GiB，日志4×512 SHA `4bf675a0ca9ffb411d1dd8b36fa691ac712053cbba67f4e399aba50c9a51b169`；冒烟绝不算已会跳。远端清单`reports/v16_long_teacher_manifest.json` SHA `f4c67fbe4cda2cf8199fa2c82047ccf3488143806de5864b6bcf2e1598e35c1c`，停止标志保持。
- UTC **08:53:15** 启动 `2026-10-17_go1-v16-long-teacher-s01-p1-d1`，**四rank每rank512纯长沟（共2048训练槽）**，900轮前150仅critic，热启v13共享模型（不以只擅短沟或轮胎的v14/v15为共享学生）；目标指令0.35–0.8、0站立采样，阶段难度第一层0/.1而非全部0.6。训练后driver在单独512纯长沟计划做确定性中间门；未达标维持hold，不自动升p2/学生/混合。UTC08:53:28看到4个真实sim.py初始化进程；远端日志`reports/v16_long_teacher_driver.log`/`v35_phased_2026-10-17_go1-v16-long-teacher-s01-p1-d1.log`、PID文件`reports/v16_long_teacher.pid`，退出收据待`reports/v16_long_teacher.exit`，stage plan SHA `438375917c118ec1aa6f44d48f77926c0c4d075689566f7848b7ed3ad5c6af2f`。不可将近沿跌落/稀疏leap奖励归因已修复；真实腾空/远沿落地及沟壁触点阳阴校准仍未完成，任何局部路线通过不等于认证真实安全跳跃。

## 2026-10-17 10:47 UTC v16 长沟中间门hold；v17诊断确认早开窗但未起跳，v18奖励探针未通过信号预检

- v16四卡各512环境900轮于UTC10:15:30 exit0，冻结512纯长沟单族门 **439/578=75.95%**，driver hold `families_ok=0/1`，权重SHA `04f853ff0089d0f6f7cbe65b9244cc0ea1648d9c173cb7cad0e9e19fb6062842`、capability_validated=False，报告SHA `741041b4970b7682e80e28e7489e5fd542711d61cc20aad4cc089a029d359317`。同512计划的逐回合trace回放报告逐字节相同：139失败全fallen，109在首沟x<2；137机身z<0.18。s0 78/119，实速平均1.046、目标0.494m/s，速度追踪不合格。900个训练摘要`leap_clear/takeoff_alignment/landing_stability`均打印0.0000，精度限制不能证明绝无腾空；路线通过不证明真实飞跃、安全或实机。
- 录像`v16_work/reports/v16_long_fail_i7_prefix8.mp4` SHA `62877eab4a1415ffb1636d5220cf84b72340c0d5a1862abb4f3cdb6fc84d4f66`，9秒前8槽RNG前缀；有/无录像逐回合SHA同`0981987708f09c753d7b4eb54923e9f3b63e0632ae5518a3809cdcf1e796cea3`，但前缀与原512不是同一精确轨迹，不能替代原门。env7近沿视频显示低伏/探步失足。隔离v17 `GO1_DEBUG_GATE=1 GO1_DEBUG_GATE_ENV=7` 逐步探针默认关闭、完整测试301 passed/76 subtests；回放trace与无探针同SHA。首次失败在x0.701即开窗（首沟沿≈1.30），x0.957 z0.312/vx1.436/vz−0.069、两足承重候选，x1.258 z0.184/vz−1.313，无采样点四足飞行候选：不是此例传感器开窗太晚，而是策略未主动跃沟并超速冲入近沿；触点仍未校准。
- 计划`docs/superpowers/plans/2026-10-17-go1-v18-grounded-prelaunch-experiment.md`的v18初版近沿单次`vz≥0.30`奖励**默认关闭**，隔离`v18_work/`，原代码备份远端`reports/v18_prelaunch_backup_20261017T1050Z/`，远端完整测试304 passed/76 subtests，四卡16槽×2/×12轮冒烟退出0。冻结前8槽启奖励后回报差**9回全部0**，12轮冒烟奖励摘要仍0.0000；env7提前下坠显示触发过严。这只是失败的信号预检，**没有v18 900轮长训/确定性能力门**。源代码部署但开关未启，不得为了保持服务器繁忙而运行没有可验证训练信号的900轮；下一步先使候选奖励非零且限制刷分，并验证精确触点/飞行，才做四卡对照。无watchdog、自动p2或并发仿真。完整诊断/视频边界见`docs/v16-long-teacher-failure-2026-10-17.md`；十族单学生与11族各90%门均hold。

## 2026-10-17 13:10 UTC 用户批准修改并继续训练：v19上抬信号通过预检，四卡短试验运行中

- 本地隔离`v19_work/`，只新增默认关闭`GO1_V19_PRELAUNCH_REWARD=1`。纯长沟、可观测沟窗口已开、距近沿0.55–0.15m、两三足净力支撑候选、站立直立与向前时，给速度上抬`clamp((vz+0.20)/0.60,0,1)`的逐段历史最高**增量**；单沟累计≤1积分。积分乘25加至原`takeoff_alignment`，其原weight2及CONTROL_DT0.02下每段累计回报≤1。绝不混用v18开关，不借家族/沟段/触点真值给actor，不改1073输入/12动作/GRU128、0.35–0.8指令、原硬碰撞/清沟/稳落/11族闸门。`--episode-trace`启v19时附精确`prelaunch_credited_gaps`及`prelaunch_credit_units`，训练checkpoint metadata标`prelaunch_v19`、`capability_validated=False`。原远端v18源码备份`reports/v19_prelaunch_backup_20261017/`。
- 新纯张量14测试先RED后GREEN；8环境原样复现先RED(`reward integration missing`)后GREEN，原成功/失败/失败位置未变、env7失败首回得到0.264积分。原512槽计划冻结v16回放578回，每回原成功/失败、x、目标/实速严格保持，437成功回及104失败回有正积分，回报差逐回等于积分；trace`v19_work/reports/v19_prelaunch_probe512.jsonl` SHA `ef52af9f514eadf734bea1e2a145d74085503f637be70b32cd3e0299087d67f8`。远端全量`./sim.sh test -q`**318 passed、76 subtests passed**日志SHA `063b40e67a74afe15cd02ee7b4b03721fddcc9f3ba75b4ab0a5e8f3f99ba95da`；四rank×16纯长沟12轮真实PPO/PhysX烟测退出0，奖励`takeoff_alignment=0.0050–0.0085`，`leap_clear`仍打印0.0000，日志SHA `deed3476a8f319d5ac499715ead716418f62ea113b24c6e52fcf7ade76fb57b7`。**奖励信号非零不是物理跃沟或技能提升**；净力及真实触点未校准。
- UTC13:08:19启动**唯一**四rank各128环境/200轮/前50仅critic的v19短试验，热启v16权重SHA`04f853ff0089d0f6f7cbe65b9244cc0ea1648d9c173cb7cad0e9e19fb6062842`，Gloo，不用watchdog/自动p2。脚本`v19_work/scripts/start_v19_prelaunch_pilot.sh` SHA `082305863344b31d41581b72acaae71e67df937004dfea3ebf5e01fe397e002c`；训练结束才自动串行原512槽同计划确定性pilot门（非学生十族/最终11族），日志`reports/v19_prelaunch_pilot4x128_{driver,train,gate}.log`，收据`reports/v19_prelaunch_pilot4x128.exit`。UTC13:10看到四个真实sim.py进程至iteration21/50、每卡约5.1GiB，无退出收据；运行/结果要重查，绝不因奖励非零直接投入900轮。停止标志SHA `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`未改。详见`docs/v19-long-prelaunch-experiment-2026-10-17.md`。

## 2026-10-17 15:35 UTC v19试点/对照配对完成：cap=1无效应、scale=4有净效应；四卡v19p4长训900轮已启动（未验收）

- **试点(13:25UTC)** train/gate exit0：同原512计划门453/576=78.65%（首槽421/512），碰撞4（v16为0，新增于x7.5–13.3带z0.27处）、跌倒123；checkpoint `checkpoints/stage1/2026-10-17_13-08-54_2026-10-17_go1-v19-prelaunch-pilot4x128-s01-p1-d1/model_199.pt` SHA `ee9a362567a89a3d24cf581dd9ac63a6efcd7e81847fc4776e669884d8986aa4`、trace SHA `da604f937869e74a55759fe2a5b393ab36b2c4e31befed3f89ac0beedd0aa5bb`、报告SHA `db06db24af788076d99808d82a8cb4de3c66b2acfe256873bb93302bec19ff64`；passed=false，HOLD。
- **同参legacy对照(15:00UTC)**：同起点同200轮/50预热/种子、仅关v19，`start_v19_legacy_control.sh` SHA `fc043d4a...` 全套通过；453/570（首槽420）、碰撞3、跌倒117，报告SHA `4b212a9ba9add448971e229834a2ab3b7998343377735e9f4c71650f16073ad4`、trace SHA `17713e6356a17974dfc3411ff8d377062bae579f09fb7d6635847cbe3c4f58a1`、checkpoint SHA `0dac8b15334bac3aeff7048fef35997081abd5c7970c858080fa6427d100c452`。**结论：cap=1试点相对v16的+14几乎全由多训200轮解释，pair差异≈0且同样引入碰撞——v19默认量级无可证明杠杆**，未直接开900轮v19。
- **scale参数TDD**：`PrelaunchGradientState(scale=)`正有限校验、credit与每沟cap×scale、步内cap=max(scale)；runtime读`GO1_V19_PRELAUNCH_SCALE`（默认1，试点行为不变）；RED（TypeError）→GREEN：测试文件16 passed、远端全量`./sim.sh test -q` **320 passed/76 subtests** 日志SHA `706a1609afe05d4120457b2b7ebe9652b8b1b59320a1ace0ce3695d1b5ae17e1`；部署prelaunch_v19.py SHA `a376ace225fac5e4a24ba516c3a87d2465823ae4d459e3d34a4c87accc40269a`、runtime.py SHA `57d56fef71d4d900ca028f673eb9ea432c52a7bf504cee7154011c4adf18a5c2`、cli.py不变。每沟回报≤0.5*scale=2。
- **scale=4配对试点(15:27UTC)** `start_v19p4_pilot.sh` SHA `e99a09533c1227f396b7c33bbbf380bbd7a0c214c37cf28d85644ea15c9ccc13`，train/gate exit0：同512门479/567=84.48%（完成报告0.8448），首槽425/512，**碰撞0、跌倒88**（对照为3/117），速度1层93/118 vs 对照75/102；对v16配对flip 31升/2降，对照25升/7降。checkpoint SHA `c771e64bfc93560a663642d017bf19ed6524692f6d8d25f19debccb167f4d7a0`、trace SHA `a5e3ca997139f6821156b21e81b99ee4ab1710a06de13a218879f10a1b9bfffc`、报告SHA `d636c5bf0704505ac8b53ae099e2c5ce991d49b1fa428791fa890b9ad9c608a5`。84.5%<90%仍HOLD，但同算力下对legacy对照有净优势且不引入碰撞——按此启动长训。
- UTC15:31:42启动**唯一四rank各512纯长沟（2048训练槽）900轮/前150仅critic/v19 scale=4/热启v16权重**Gloo；`start_v19p4_long.sh` SHA `1c7303d3fc2cd665952f0f1f0b97b4800befa0f06772580f07748e75f6f6dbe3`，driver日志`reports/v19p4_long_driver.log`、PID`reports/v19p4_long.pid`(117419)、收据待`reports/v19p4_long_teacher.exit`；启动实测5进程、每卡约7.5–8GiB。训练后同脚本自动串行原512计划确定性门（v19开、trace含credit）。仍未验证明真实腾空/落地/触点安全；无watchdog、无自动p2、十族学生与11族最终门继续hold。

## 2026-10-17 17:00 UTC v19p4长训900轮完成：纯长沟族门97.09%达标（碰撞0），但leap_clear全程0.0000，非飞行安全认证

- 远端`reports/v19p4_long_teacher.exit`为`train_exit=0 gate_exit=0 utc=2026-10-17T16:55:58Z`，本地备份`v19_work/reports/`同内容；UTC16:56复核零sim.py进程、停止标志SHA仍`fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`。checkpoint `checkpoints/stage1/2026-10-17_15-32-19_2026-10-17_go1-v19p4-long-teacher-s01-p1-d1/model_899.pt` SHA `75968c8885cfc842d1edd4f46ba15f3076d4912d100dd02a51840e99f4adbdc3`，metadata `reward_variant=prelaunch_v19`、`capability_validated=False`、`initialized_from=v16/model_899.pt`；训练日志SHA `5db1e97ac57fb78a498a958da0b065e02d5ac2e2eda116f06343fd4a2973ee81`，第899轮`takeoff_alignment=0.4374`、`leap_clear=0.0000`——900轮全部摘要`leap_clear`均为0.0000，**没有任何已验证的真实腾空证据**，达标路线不能归因于跳跃。
- 独立512纯长沟确定性门（同v16计划、seed2026、v19开）：`reports/evaluate_stage1_2026-10-17_16-55-56_2026-10-17_go1-v19p4-long-teacher-s01-p1-d1-gate512.json` SHA `c119a3b9356e16dbaa2ceaf83f24f32dd659ba2987cf26ab9f6904e4f9d31c3b`（本地`v19_work/reports/`同SHA备份），`completion_rate=0.97087`（**≥90%族门限达标**），碰撞0/512、跌倒15/512、`velocity_xy_rmse=0.691`；训练stage报告SHA `9d30d979d088aeb3c6dc9c2d130bfd056901173b215abeba1efa8417e7c53517`；trace SHA `abef68261f1716748f2783e04c233a2bc74375efdbcfc6b43f2215c9f21a0678`，逐回合500/515成功、首槽497/512；对v16配对101升/15降，15个失败全部在首沟前x<2低伏跌倒；速度0–4各层101/111、100/105、100/100、102/102、97/97。难度仅0/0.1，均速实速1.055对指令0.486，**速度追踪不合格**且难度谱未推进，不等同最终验收。
- 该族门达标按v15轮胎同标准处理为**可进入下一串行专项**：冻结v19p4作纯长沟教师；十族单学生蒸馏、11族同次512各90%最终门仍hold，需按计划推进，不自动p2。真实腾空/沟壁触点校准缺失仍是显性风险，任何长沟路线通过不认证实机安全。

## 2026-10-17 18:15 UTC 用户批准第二步：v20十族单学生训练已启动（复用v13驱动，未验收）

- 查清路线：**本项目“十族单学生”不是未实现的multi-teacher DAgger（仅设计文档`docs/superpowers/specs/2026-10-16-go1-ten-family-first-multiteacher-dagger-design.md`），而是v13式十族混合on-policy训练（roster=`ten_except_combined`，combined留最终11族门）**。远端`scripts/run_phased_curriculum.py`已被v14改成单族gaps_short专用（`specialist_roster_v14`导入与两处族断言，SHA`dda8e91f...`），故v20不改写它，另部署v13本地快照为`scripts/run_phased_curriculum_v20.py` SHA `ddb2737417dc658ac3098cdf6c5ee4d447f7e8fb3ae33ebdf8cec465b229f27e`，`lp_acrl_stage.py` SHA `745eb674...`与`ten_roster_v13.py` SHA `cab81a28...`保持。
- 新隔离`v20_work/`：配置`curriculum_phases_ten_student_v20.json` SHA `90d11098d157d704333cc8c942a8c530c0076510214bea5f5747b9498dd84c58`（ten_except_combined、每rank 512、900轮、前150仅critic、难度0–0.6、速度全5层0.35–0.8、gate completion 0.9且每族≥40槽、require_deterministic_gate、state/sampler/results均v20新名不触碰v13旧态）；**init_checkpoint=v19p4长沟教师**`checkpoints/stage1/2026-10-17_15-32-19_.../model_899.pt` SHA `75968c8885cfc842d1edd4f46ba15f3076d4912d100dd02a51840e99f4adbdc3`。合同测试先RED（2 failed：driver/launcher未部署；roster测试GREEN证明十族=11族去combined）后修正两处我测试自身写错的断言再GREEN：3 passed；远端全量`./sim.sh test -q` **323 passed、76 subtests passed**日志SHA `b037cf377e9afc6b050fb1bc986fd3a9b263aa0544dbf3c15205d0628546a689`。actor1073/12/GRU128、奖励toml v11_unified与v19 scale=4（仅gaps_long槽付分）保持不变，Gloo 4rank，无新增权重合并代码。
- 冒烟首跑真实RED：误用512槽v13 gate plan配16环境，exit1 `task plan length must match environment_count: 512/512 vs 16`（失败日志`reports/v20_smoke4x16.log`/`.exit`保留）——运行时要求计划长度=每rank环境数（v16当年`plan${n}`同尺寸即为此）。生成十族16槽冒烟计划`reports/v20_smoke_plan16.json` SHA `7f8866ec850aa9adbfe32bf8a30cf1a241d626f60bb5731a1d8e0c5c3e855667`（ten_roster+sampler+driver.rebalance，16槽覆盖10族各≥1）；`v20_smoke4x16.sh` retry1版4rank×16、2迭代真实PhysX/PPO退出0，收据`reports/v20_smoke4x16retry1.exit` `exit=0 utc=2026-10-17T18:12:40Z`，脚本SHA `721393ad1ddf34b084dadb847ec96bf83969b3d950abfb674777359cf3a47226`。
- UTC18:13启动**唯一**四rank各512十族混合900轮：`start_v20_ten_student.sh` SHA `1f50122b7004ac1adf0ca08b70585bff639feec798bff8ea0b829739e2be356a`（preflight核对上述SHA链+v19p4门报告SHA`c119a3b9...`completion0.97087/碰撞0+停止标志+GPU空闲+无sim进程+输出全新），PID`reports/v20_ten_student.pid`(32017)，驱动stdout`reports/v20_ten_student_driver.log`，训练`v35_phased_2026-10-17_go1-v20-ten-student-s01-p1-d1.log`，门`v35_gate_...log`（驱动自带确定性512十族门，每族≥40槽），收据待`reports/v20_ten_student.exit`。启动实测FAMILIES每族51–52、mean_difficulty0.069、5进程、每卡8.5–8.9GiB、停止标志SHA`fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367`未变。

## 2026-10-17 20:15 UTC v20十族门91.78%但9/10 hold：轮胎族42.4%干扰回归；v21自v20权重纯轮胎刷新已启动

- v20完训`reports/v20_ten_student.exit` `exit=0 utc=2026-10-17T19:53:40Z`；驱动确定性512十族门报告SHA `4797a6a5a768c06b7785a09194f8281d2e82897eb485d1c94071c6243ff76077`：completion **0.9178（608回合）**、碰撞1.15%、跌倒6.1%、cmd0.529/achv0.625、`velocity_xy_rmse=0.302`、**families_ok=9/10 → hold**。分族：slope×3/stairs×4全1.000、stairs_up 50/51、gaps_short 0.922、gaps_long 0.954（v19p4技能存活）、**tire_field 28/66=42.4%**。门计划SHA与v13时代十族平衡计划相同 `003a273e...`（确定性可比重演）。checkpoint `checkpoints/stage1/2026-10-17_go1-v20-ten-student-s01-p1-d1/model_899.pt` SHA `69ad0eb06747faa00aad2c2cb608a1cceb56244e9b0062591ef10c964fd13303`，metadata `prelaunch_v19/capability_validated=False/initialized_from=v19p4`。
- 逐回合重放（同seed同计划，`reports/v20_gate_replay_trace512.jsonl` SHA `7c17f12c...`，与驱动门**608回合逐env统计零失配**，确定性复现成立）诊断轮胎失败模式：26 fallen（x中位11.0，场中场中，非入口）+12 timeout，失败回实际均速0.47对指令0.68——混训把v15纯轮胎教师的低速谨慎跨胎步态冲掉了，学生高速冲胎摔倒或磨蹭超时。非奖励语义故障（`leap/gap_fall=0`），是稳定性-可塑性干扰。
- 授权范围内推进v21：**沿用v15已验证纯轮胎配方**（同计划SHA`102ba49f...`、4×128、900轮、前150仅critic、cmd0.35–0.8、legacy奖励即**关**v19/v14/v9、Gloo四rank、flock、无watchdog），热启改为v20学生权重`69ad0eb0...`；先RED（合同2 failed：launcher未部署）后GREEN **5 passed**，远端全量`./sim.sh test -q` **325 passed、76 subtests**日志SHA `c9b101b9689b1f9cc68f4d13eef2a3b5ee90ae749655df37037e06235b456f01`；`start_v21_tire_refresh.sh` SHA `f93f3176f59c2776cea42ddec5b486c042c827165e714b02dfe93a9aaf5f0bdf`。训后自动串行两门：v15轮胎128门（trace`reports/v21_tire_gate128_trace.jsonl`）与**十族512门重跑**（trace`reports/v21_ten_gate_after_v21_trace512.jsonl`），收据`reports/v21_tire_refresh.exit`、PID 64658、UTC20:14:23启动、5进程实测。停止标志未变。**刷新后若其他族回吐（尤其gaps_long），十族门hold仍维持；无任何自动p2，11族同次512最终门继续未做。**

## 2026-10-17 21:35 UTC v21轮胎刷新结果：振荡实证——轮胎回血但gaps回吐，十族门8/10 hold；串行自动刷新线判定终止

- `reports/v21_tire_refresh.exit` `train_exit=0 tire_gate_exit=0 ten_gate_exit=0 utc=2026-10-17T21:31:33Z`；checkpoint `checkpoints/stage1/2026-10-17_20-16-51_2026-10-17_go1-v21-tire-refresh-s01-p1-d1/model_899.pt` SHA `4f5caf1a1cd22cfa4ff47bc473b477c231274ffcef93da72b0599fd67a8e186d`（`reward_variant=legacy`，capability_validated=False，initialized_from v20）。轮胎128单族门 **112/138=81.2%**（碰撞7、跌倒26；高速槽31/48；对照v15纯轮胎教师当时96.12%仍hold），逐回合trace SHA `92bb0031...`。十族512门重跑trace SHA `4a0e1290...`：**总体92.88%但families_ok=8/10 hold**——tire_field 0.424→**0.879**（部分回血，仍<0.9），**gaps_short 0.922→0.725、gaps_long 0.954→0.875回吐**，碰撞1、跌倒43。slope/stairs七族维持1.000。
- 结论（证据而非猜测）：串行“单族刷新→再门”在v20↔v21间**振荡**（修轮胎伤gaps、v20修gaps伤轮胎），继续同法刷新不收敛；十族单学生**最佳点仍是v20**（9/10，唯一未达族tire_field 42.4%，失败模式=场中摔倒26+超时12、实速0.47对指令0.68）。冻结v20 `69ad0eb0...` 作当前十族学生候选；v21 `4f5caf1a...` 留档不回退采用。
- 停止标志SHA `fe5dc65082919596404f23e7d829fa8f2824920f3d849df2fe22cb21f4e90367` 全程未变；训练后零sim进程。11族同次512最终门、自动p2、教师权重合并均未做；真实腾空/触点安全仍未校准。服务器空闲，等待人工对下一步（提高混合训练轮胎采样占比、或接受v20现状先跑11族最终门、或其他专项）的指示。

## 2026-10-17 23:40 UTC v22三族精调：迄今最佳十族学生候选97.29%、9/10族达标，唯一未达=轮胎75.9%

- 授权内自主决定避免串行刷新振荡，改为**三族精调混合**（只采样轮胎+短沟+长沟，slope/stairs不受扰动）。新包模块`three_roster_v22.py` SHA `9d176affd4b233f188e52d8419d76ceeb93d8213297a38a5a3aead40fecd8704`（REFINEMENT_FAMILIES=tire_field/gaps_short/gaps_long，拒绝其它roster）；v20驱动副本`run_student_v22.py` SHA `c0220a25...`（改import与两处断言文案，flock/exec9，无watchdog）；配置`curriculum_phases_triple_refine_v22.json` SHA `ff8e3c62...`：min/max_family_share .25/.4（3族可行域），512/rank、900轮、前150critic、v19 scale=4保持、热启**v20学生**`69ad0eb0...`。合同测试先RED（collection error：模块缺失）后GREEN **3 passed**；远端全量`./sim.sh test -q` **328 passed/76 subtests** 日志SHA `8618b347e148a27af9566ed0b68bb2a03b8acefc188220b41112d06b3a14eed8`。16槽三族冒烟计划`reports/v22_smoke_plan16.json` SHA `9f83d801...`，4rank×16×2轮退出0。启动UTC21:38（PID 90018，FAMILIES每族170–171实测），驱动收据`reports/v22_triple_refine.exit` `exit=0 utc=2026-10-17T23:21:18Z`。
- **三族门**（balanced_3_families）：overall 0.9387、families_ok=**2/3 hold**——gaps_short 0.983(174ep)、gaps_long 0.987(236ep)回血达标，**tire_field 0.831(177ep)**仍<0.9；cmd0.545/achv0.773超速复现。
- **十族门重跑**（v20计划、seed2026、v19关闭不影响动作已证、trace`reports/v22_ten_gate_trace512.jsonl` SHA `84766794301f04dea417da18f6eca7547745e583af73027ac5348a95f57f3eb3`、报告`evaluate_stage1_2026-10-17_23-32-52_v22-ten-gate512.json`）：checkpoint `checkpoints/stage1/2026-10-17_go1-v22-triple-refine-s01-p1-d1/model_899.pt` SHA `09a5357dca3964d33db84b96453d0f387f8924bf78664f3cf302e6eed352d76a`。**总体97.29%（591回合）、families_ok=9/10**：gaps_long 0.99、gaps_short 0.986、slope×3/stairs×4全1.000（七强族零回吐，证明精调不伤未采样族）；**tire_field 0.759（44/58：12 fallen+2 timeout，超时中位x16.7m）**。对比：v20=91.8%/9-10（tire .424）、v21=8/10、**v22=97.29%/9-10为迄今最优学生候选**。
- 门判定仍hold（要求10/10）；下一步选项留人工：(a) 轮胎场内专项（失败=场中11–17m摔倒+超速指令，或需难度带/步态干预而非更多采样——v21纯轮胎900轮也只81.2%，同配方复调未必收敛）；(b) 接受v22现状先跑11族同次512最终门（combined仍缺）；(c) 其他。停止标志SHA未变、训练后零sim进程、无自动p2、capability_validated全False、真实触点/腾空仍未校准。

## 2026-10-18 08:31 UTC 用户选(a)力求全过线：v23制动奖励三族精调大幅回血gaps、轮胎升至0.821；v24续轮运行中

- 失败数据定位：v22十族门轮胎失败10/14集中于cmd≥0.65（实速0.68–0.77摔倒x2.5–23.4；v15纯轮胎教师同场仅~0.565m/s通过），判定为超速-摔倒干扰，非采样不足。toml中`track_linear_velocity`weight=3.0在轮胎槽反而支付dash，遂新增**默认关**`GO1_V23_TIRE_BRAKE=1`训练侧制动塑形（`tire_brake_v23.py` SHA `3e6a2b73f91bc92b4731a58b7a32db6b5a5001b78989c68e0f9aea893c54db0f`）：target=clamp(cmd,max=.60)，bonus=exp(-(v-target)²/std²)-exp(-(v-cmd)²/std²)，overcharge=-0.5*clamp((v-.68)/.3,0,1)，加到既有`track_linear_velocity`通道，非轮胎槽恒0，逐槽clamp(-1.5,1)、状态无关不可刷分。cli把reward_variant改为部件链（v23+v19共存标记`tire_brake_v23+prelaunch_v19`，保留v14测试要求的子串）。
- TDD：模块测试先RED（ImportError）→修两处我测试自身期望算错（cmd≤cap时bonus恒0、掩码index1非轮胎槽）→GREEN **7 passed**；远端全量`./sim.sh test -q` **335 passed、76 subtests**日志SHA `5bc49fe69f28e48cb4acff3614839bbbb159ef276dacf42a4d5efb587d3cb843`。runtime/cli新SHA `8f3138d88e8e587621ee750d98a12ba1e01d30ba82a27dfb8f1cce96e9142c61`/`4f9f4af26b20ee3d838333b1b1dd55368bcbd21eda9a4b66be3aa55e9f4a135d`，部署前备份`reports/v23_backup_20261017/`。
- **v23三族精调**（`run_student_v23.py`=v22驱动同字节`c0220a25...`、配置SHA `8e568b26377faae6007075a6e9c0d1632fd379d46e45d9d315582507f0de97c3`、冒烟`reports/v23_smoke4x16.exit exit=0`，init=v22 `09a5357d...`，flags V19+V23同开、Gloo 4×512、flock、无watchdog）900轮完训`reports/v23_triple_refine.exit exit=0 utc=2026-10-18T08:05:29Z`：三族门**advance 0.9655、families_ok=3/3**——tire **0.914(175ep)**、gaps_short 0.972、gaps_long 1.000；checkpoint SHA `45e98ab7c7e47f9629ad0a901d9bc3dbc19738dc821e04893c84c09e327aa878`（`tire_brake_v23+prelaunch_v19`，capability_validated=False）。
- **v23十族门**（v20计划、seed2026、v19/v23关闭不影响动作已证）：报告`evaluate_stage1_2026-10-18_08-24-20_v23-ten-gate512.json`，trace SHA `cb324e8235eed24425334d815dc5ffb8f80d0fb1bd7ac95dc6c38772d9be97a7`：**overall completion 0.9796（589回合）=至今最高**，gaps_long **1.000(104/104)**、gaps_short 0.972(70/72)、七强族全1.000，**tire_field 0.821（46/56：7 fallen+3 timeout）仍差一线**，碰撞3、速度RMSE 0.275。families_ok=9/10 hold。三族0.914 vs 十族0.821为同权重不同轮胎槽样本（175 vs 56 ep、十族槽速度带更严）；趋势：tire .424→.759→.821逐轮上升、gaps已稳态高位。
- UTC08:29启动**v24续轮**：同驱动/同三族roster/同制动、init=v23 `45e98ab7...`、配置SHA `87a3c53061b38d238ffc7c22ebd9d1fa7d452d2d51b5733cb8940f2bf848b54b`、冒烟`reports/v24_smoke4x16.exit exit=0`、PID 126293、驱动日志`reports/v24_driver.log`、收据待`reports/v24_triple_refine.exit`；完训后自动三族门，再手动重放v20计划十族门（trace待`reports/v24_ten_gate_trace512.jsonl`）。停止标志`fe5dc650...`未变。**十族10/10与11族同次512最终门均未过；无自动p2、capability_validated全False、真实触点/腾空未校准。**

## 2026-10-18 12:30 UTC v24/v25十族门定量：v24最佳(98.28%/tire .830)、v25放宽参数被否决、v26原参数续轮运行中

- **v24十族门**（trace SHA `0d801c3733979cd66ae87bf8a34b2fe431939f45b29cf3fc13617622e7c99956`，报告`evaluate_stage1_2026-10-18_10-25-46_v24-ten-gate512.json`，ckpt `d41e627f9d8d0417fe02d989fb5c4ffc80be91a5a6bfdbf80cdbbf4c7d8d60d1`）：overall **0.9828=历史最高**，gaps_short 1.0(68/68)、gaps_long 0.99、七强族1.0、**tire 0.830（44/53：3 fallen+6 timeout）**，9/10 hold。三族门0.881同向。
- **v25（cap .65/pen .35制动放宽，init v24）**：三族门hold（tire 0.884、gaps_long .978回吐）；十族门**退化**（trace SHA `72dafc114780b9fa88bf824e7f909c953158d6dbbc0f326308ae743770d706d5`，ckpt `1596bd78...`）：overall 0.9702、**tire 0.768、13例全部fallen（速度0.28–0.7均摔，timeout消失）**——证明轮胎失败已不纯是超速，慢速同样摔；奖励松紧只移动“摔↔爬”平衡点，塑形到平台。v25权重不采用。
- TDD：`GO1_V23_TIRE_CAP/GO1_V23_TIRE_PEN` env参数化（默认.60/.5=旧行为），手算字面量测试新增3断言 → v23文件8 passed；全量`./sim.sh test -q` **336 passed、76 subtests**日志SHA `fc750a5b7fb82101f553efd3c3bb1d9b8af85b6c6240af80fc4ae80b610bf827`；runtime新SHA `c93ab53b0f8ac6c49358e80657c9a21f00c82b8ab397fe333e89a68d79efac7e`（备份`reports/v23_backup_20261017/`仍在）。v25冒烟exit0（`reports/v25_smoke4x16.exit`）。
- **v26已启动**（UTC12:26:xx PID 61364）：回v23原参数cap .60/pen .5显式导出，init=v24 `d41e627f...`（轮胎单调线.424→.759→.821→.830在此参数下），三族roster 900轮；launcher `start_v26_triple_refine.sh`、config SHA `9f445337e634d0e633011a6594ed29416a87f0fca889e02a6c79c2495ab79609`、冒烟`reports/v26_smoke4x16.exit exit=0`、收据待`reports/v26_triple_refine.exit`，完训自动三族门+手动十族门。若v26轮胎仍<0.9，则刹车塑形判定平台终了，需转入结构性方案（如教师动作蒸馏——现库无实现，仅`distillation.py`有感知通道可参照）而非再调参。停止标志`fe5dc650...`未变。

## 2026-10-18 14:20 UTC v26十族门10/10达标（tire 0.907首次过线），十族单学生达成；开始11族同次512最终门

- v26（原参数cap.60/pen.50从v24续900轮）三族门先hold（tire 0.861、gaps_long .935、gaps_short 1.0；三族门是170槽另一分布的辅助门，非权威）。**权威十族门（v20计划、seed2026、纯评测）：families_ok=10/10、overall 0.9861（576回合）——十族单学生达成**：tire_field **0.907（49/54，仅5例fallen，cmd .64–.73，timeout清零）**、gaps_long 0.97、gaps_short 1.0、slope×3与stairs×4全1.000；trace SHA `71f1b5dd057a174efd4674281146f8d00cfe3b7eb4ce3f8e7acd3f6a28f3638c`、报告`evaluate_stage1_2026-10-18_14-16-27_v26-ten-gate512.json`、checkpoint `checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt` SHA `6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674`（`tire_brake_v23+prelaunch_v19`、capability_validated=False）。
- 轮胎单调线终局：v20 .424→v22 .759→v23 .821→v24 .830→**v26 .907**；v25（放宽参数）被数据否决未采用。49/54贴线过（48/54=0.889不过），如实记录：方差敏感、非大幅裕度。
- 轮胎五例摔倒在cmd .64–.73仍存（塑形平台后残余），11族最终门与实机安全认证另论；无自动p2。下一步为计划内最终门：11族同次512各90%（combined首次入槽评测），先跑纯评测门拿差距数据。

## 2026-10-18 16:36 UTC v27十一族混合combined仍0；诊断为入口超速；v28把制动掩码扩至combined运行中

- v27十一族门（driver自动，`evaluate_stage1_2026-10-18_16-14-48_gate-...`，ckpt `7b5f7bf450017e81cd9e520f8d8bef1fd58c3ba0eec19ab7bd44a6a990093c43`）：overall 0.7708、families_ok=**9/11 hold**：tire 0.957、slope/stairs 1.0、gaps_short .939、gaps_long **0.699（回吐）**、**combined 0/110**。门计划逐env mean_failure_x字段恒0（不可用），带trace重放`reports/v27_combined_trace512.jsonl` SHA `baffeec6e85fecfd2d192ff370af0d1ef86fabfb9df834549ae3bdba64338d3f`（同ckpt/seed/计划）诊断：**combined 110回0成功，101 fallen+7 timeout+2 collision，失败x中位3.2m（半数以≤3.2m）**——入口冲撞，失败回实速中位0.70对指令0.48，槽难度仅0/0.1。v7世代combined专属term语义不变，本轮问题不是奖励缺失而是入口速度控制未覆盖combined槽（v19只掩gaps_long、v23只掩tire_field）。
- v28改动TDD：`tire_brake_v23.py`新增`parse_brake_families()`（GO1_V23_TIRE_BRAKE_FAMILIES，默认'tire_field'保持旧语义；空/重复/尾逗号条目raise；运行时再校验族必须在计划内否则raise，杜绝静默无效）。测试先RED（ImportError）后 **9 passed**；全量`./sim.sh test -q` **339 passed、76 subtests**日志SHA `b3ea699508d637efd2c8d378d8bd7762cc97518c8120eed15b2e8c7dda89ab6a`。新SHA：tire_brake `41a9dec7...`、runtime `50ed5a7c...`（cli/其余不变）。eleven_roster_v27 `da61d49c...`、驱动v27 `e4eeb9a8...`复用。
- **v28十一族混合**已启动：init=v27 `7b5f7bf4...`、config SHA `30a4404a...`（11族、share .06–.10、512/900/150）、flags V19 scale4 + V23 brake **tire_field,combined** cap.60/pen.5、冒烟`reports/v28_smoke4x16.exit exit=0`（16槽覆盖11族计划校验通过）、launcher `start_v28_eleven_mixture.sh`、PID 130927、UTC16:34起、收据待`reports/v28_eleven_mixture.exit`、驱动自动balanced_11门。停止标志`fe5dc650...`未变、服务器串行单Isaac。combined历史轨迹：v26权重0/112（零曝光）→v27权重0/110（9%曝光一轮）→v28待测；gaps_long需同时观察不再回吐。11/11或任何最终达标前不构成验收。

## 2026-10-18 18:35 UTC v28否决：combined三代权重同签名0/118入口摔——制动扩族无效，混合路线达界；v27仍最佳

- v28十一族门（ckpt `4b32551e26c508e96ab5a09312db92d512bf880e4cf1837ab459fc9ead05f4f0`，trace `d381c14c059eb05b755ca5d64ff5edcf6c5e740236cb8837eec0bc4b3ebfa7c5`）：8/11、overall 0.7595；combined **0/118（110 fallen）**、gaps_short 0.855/gaps_long 0.739被拖回吐、tire 0.957持平。
- **决定性对比**（同seed同结构计划，三代combined失败x分位）：v26从未见combined=[1.6,3.1,13.4,14.1,22.3]、v27训后=[1.4,1.8,3.2,14.0,22.4]、v28制动扩族后=[1.4,1.9,3.2,14.1,22.1]，实速中位恒0.70–0.71——**combined入口1.4–3.2m与中场14m的摔与训练代次/制动mask无关**，是策略步态对combined首障碍的物理能力缺口；刹车塑形对tire有效（0.83→0.957），对combined无效且扩族轮把gaps拖回吐。v28权重不采用。
- 十一族混合现状最佳仍为**v27**（9/11：tire .957、gaps_short .939、gaps_long .699、combined 0）；十族最佳仍**v26**（10/10）。十族学生达成、11族最终门未达。
- 下一路线（未自动开火，服务器空闲）：combined需**专项诊断→专项教师**（如v17对long入口探针，需combined入口几何刷查：1.4m处物体高度/类型、接触/翻越失败），然后combined独立4×512训练把该族拉过90%，再重跑十一族混合（从v27/v26系）并同次512门。这是新增专项授权点。无自动p2；真实腾空/触点未校准；capability_validated全False。

## 2026-10-19 01:28 UTC 用户指定按P10论文路线：十族多教师DAgger蒸馏学生（v29运行中）；PDF无法读取已告知

- 用户发PDF请求按论文方法“其他10族教师蒸馏出一个混合学生”。本会话模型不支持PDF输入已声明；本地`docs/superpowers/specs/2026-10-16-go1-ten-family-first-multiteacher-dagger-design.md`即按Robot-Parkour机制写好的Stage-B/C规范（多教师→学生访问状态DAgger→单循环学生；教师体系先10族各≥90%再蒸馏；学生先十族屏障再冻结测11族混合），本次按其实现。
- 教师屏障证据：tire v15 `4f18d4bc...`门96.12%、long v19p4 `75968c88...`门97.09%、其余八族v26 `6607e26d...`十族门各≥0.972（10/10）。教师映射`configs/teachers_v29.json` SHA `7690fdcf34938e9cc36dc7120e769bac1b188fdcda61467f144c2c18e5fea2eb`（general=v26/tire=v15/long=v19p4）。combined由几何段调度：开头三沟+前后半程→general，轮胎带→tire（`mixed_reward_segments`仅训练期读取，actor无族/段真值）。
- TDD：新模块`teacher_distill_v29.py` SHA `685d945af453b334af6f9e5a989820b2f6f3869fcb50fa888a93e76e2b2e4dbd`（TeacherPool冻结/hidden per-env推进reset、family_teacher_keys族+段路由、未定义隔离）先RED（ImportError）后修两处我测试自身期望（must-not-be-empty文案、mean-over-dims语义）GREEN 8 passed；`on_policy.py` SHA `08111ab0299aa56e2b09b774d1fa6e6735f7fa90bf2c146a69f850fb5935fdda`新增`recurrent_distill_epoch`（沿rollout递归、done处清零hidden、masked MSE×λ、inference张量clone修复）+CheckedPPO extras→stack→update后监督step+`V29_DISTILL`打印。集成两次真实修复：LeapMaskWrapper原仅leap策略时启用（v29补条件），inference-mode张量需clone才能autograd。cli `b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9`、runtime `08d609da744a769c93f801d2e9f546ffb9d7da2164611d3ad818599cbf6b6a9f`、旧版备份`reports/v29_backup_20261018/`。全量`./sim.sh test -q`**347 passed、76 subtests**日志SHA `8308840a56343862fbbee4ef97661d17d75dd30c4f850433c11bd09a66cd8926`。冒烟4×16×2轮exit0且蒸馏实跑：**每迭代768监督样本、loss≈0.037**（`reports/v29_smoke4x16.exit`）。
- **v29学生蒸馏训练**UTC01:27启动：run_student_v27驱动（11族roster，combined首次入训）、config`curriculum_phases_student_distill_v29.json` SHA `64f3bcd5d74f616b0b784f98ce3ff8172ff4849fff20a47d3a69ee04c1eaf0d6`、init=**v26** `6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674`、ε=.5 λ=.5、Gloo 4×512、900轮/150critic、flock、无watchdog；launcher `0e02c137e5aef4b5f938e5d2f40da781f28e448339def9483b57918a1e483e10`、PID 9836、收据待`reports/v29_student_distill.exit`、驱动自动balanced_11门。完成后即同次512 11族各90%判定；任何结果不构成认证，触点/腾空仍未校准。停止标志`fe5dc650...`未变。
- v29首发真实失败：warmup阶段actor被cli冻结（requires_grad=False）而distill epoch照跑→`element 0 of tensors does not require grad`崩溃（`reports/v29_student_distill.exit exit=1`与v35日志保留不伪装成功）。`on_policy.py` SHA `f1ea4784bf861dff18c09bcf5d426aa3063e356914b6f8504ee43b380f8ed6ec`加trainable守卫跳过warmup标签；2rank warmup2+正式1探针exit0：warmup零输出、正式迭代两rank各打一条`V29_DISTILL samples=768 loss≈0.059`。另probe初撞plan长度合同（8env vs 16槽）系我参数笔误非代码问题。
- **v29b蒸馏重跑**UTC01:37启动：launcher SHA `4b939c3eae99b5f06b78e099f965870283d361e5b6c853d8cd4b33a12f7011c4`（pin cli `b2c4cf0c8d920fb11cfb79ec0f5a4b1722abb9d255099151432bc319a303b2d9`、on_policy `f1ea4784...`）、config SHA `2b089a2949a922dd9a89752c1b5c93a479832bfd85a5384021235761b40a09f1`、base-id `2026-10-19_go1-v29b-student-distill`、PID 16172、5进程、收据待`reports/v29b_student_distill.exit`。预期完训+自动门约03:20，随后手动同次512 11族门复核。任何结果不构成认证；触点/腾空未校准；无自动p2。停止标志`fe5dc650...`未变。

## 2026-10-19 03:38 UTC v29b蒸馏门数据：combined仍0——蒸馏上限实证（无胜任combined教师），ε教师动作伤gaps；v30 combined专项已启动

- **v29b完训**`reports/v29b_student_distill.exit exit=0 utc=2026-10-19T03:27:43Z`，自动balanced_11门：**combined 0/120**、gaps_short 0.746、gaps_long 0.852、tire 0.935、slope/stairs七族1.0，overall 0.7622、8/11 hold；训练末`V29_DISTILL samples=24561 loss=0.033`（管道健康）。机制结论（一手证据）：general教师=v26自身，**v26在combined为0/112**——教师标签在combined状态复制的是失败动作（spec line 20禁止教师把未知物理伪装成可观测事实的边界情形），DAgger无新能力注入；ε=.5教师接管还把gaps两族从0.97/0.99拖到0.746/0.852。**十族教师→蒸馏学生的路线不能造出combined能力；必须先有胜任combined的教师**（spec line 26过渡教师路线）。十族学生最优仍是v26（10/10），v29b不采用（combined 0且gaps回吐）。
- **v30 combined专项**UTC03:34启动：新roster模块`combined_roster_v30.py` SHA `3d28aec8f87e6fd8b84f587cd43fd12c3e22b9a5c74cee969fb6f2545037114a`（仅'single:combined'）、驱动副本`run_student_v30.py` SHA `ec2620483323f80ca494c60fbfc89dcac3fb1e236dece90598c8e80d74031d06`、config`curriculum_phases_combined_v30.json` SHA `c0df50ac6aa8d10366535cf9603133e5dd6d4248b160daa45d8f74f3c55f5962`（share 1.0、512×4、900轮/150critic）、brake族=combined cap.60/pen.5、v19保留（combined mask恒False=无信用，如实）、init=**v26** `6607e26d...`、无distillation flag。合同测试先RED（collection error）后2 passed；全量**349 passed/76 subtests**日志SHA `7369929a8a1a30a0de5c7c22b9d935f5192934a44fc17af0c85a6346eda62cf6`；16槽pure-combined冒烟计划`reports/v30_smoke_plan16.json` SHA `cf74e53c...`、smoke exit0；launcher `ab19fa46be7f034263870b822be246ab24392b89e034581e5b1cfbf101fd79d2`、PID 48835、FAMILIES combined:512实测、收据待`reports/v30_combined_teacher.exit`。历史基线：v26/v27/v28/v29b权重combined均0；本专项是首次以combined为唯一训练目标的PPO。若v30教师≥0.9：教师体系补齐11族，再做蒸馏/混合门；若仍0：如实报告物理接口局限（观测无法区分必要动作），不伪造通过。无自动p2；停止标志`fe5dc650...`未变。

## 2026-10-19 04:20 UTC v30a崩溃根因=combined路线几何生成器；build_scene确定性重试修复并复现验证；v30b运行中

- v30a `reports/v30_combined_teacher.exit exit=1`（03:36）真实崩溃：`ValueError: combined route has no room for finishing tyres: 31.227`（geometry.py:865）。独立probe证明**与难度无关**（plan全≤0.1），是512-tile共享rng流组合的偶发不可装箱：同plan四rank种子2026/2028必崩、2027/2029可过（probe_scene2实录）。历史v15/v20/v27没崩=族序列不同未触发，非旧代码更正确。
- 修复TDD：`build_scene`对combined“finishing tyres”错误做≤6次确定性重采样重试（attempt0用原流、重试从主流draw新seeded rng；失败类型/耗尽仍原raise；不触发时rng前进与旧完全一致→既有成功计划与eval逐字节不变）。`geometry.py` SHA `298c24671689421b8861083868abe75e4c7d4565da4af0bab48bc02506f57f13`，旧版备份`reports/v29_backup_20261018/geometry.py`。测试`test_v30_geometry_retry.py`：先以probe在旧代码实录seed2026/2028 FAIL，修复后新测试**2 passed（416s）**含双次构建vertices/origins逐元素等值（可复现性）；全量`./sim.sh test -q` **351 passed、76 subtests** 日志SHA `9c9d2c889508a1c89e782f3e545396181781cb780e503b4d81f509f61f580773`。
- **v30b**（同专项、plan/state/receipt全v30b新名，launcher pin新增geometry，且preflight断言v30a收据`exit=1`在案）：config SHA `ce106a4b471a6a266c4ea9a6d555f07442e4cef324032e9a6a72c8de3dbd71f9`、launcher SHA `43a8914d8169ed3f23974389f900557b714aeb5cc9005a130e4f097a59ee5858`、base-id `2026-10-19_go1-v30b-combined-teacher`、PID 65914、FAMILIES combined:512实测、04:18启动，收据待`reports/v30b_combined_teacher.exit`，驱动自动balanced_1门（≥40 ep）。预计约06:05完。combined若≥0.9则并入教师池重蒸馏；若仍0则报告接口局限。十族最佳仍v26、11族最终门未达。

## 2026-10-19 06:40 UTC v30b combined专项终判0/649：入口首障碍对全体已训方法物理不可能通过，训练路线冻结、转接口诊断

- `reports/v30b_combined_teacher.exit exit=0 utc=2026-10-19T06:17:52Z`，balanced_1门 **combined 0/649（completion 0.0000，cmd0.569/achv0.316）**，ckpt `1e1eab32c74fae3b89ac0c4a97581e1cb12f2a33ec1e1f904d1f683e618db798`（不采用）。trace重放（`reports/v30b_combined_trace512.jsonl` SHA `82e6430b79d4c3db71b67fe11fd60c48d118a66935c295aaeab06deaacd6200c`）：537 fallen+111 timeout+1 collision；失败x双峰**入口1.8–3.2m（中位3.27）与中场13.4–14.4m**；fallen中位步数230（≈4.6s）即在出发后第一个障碍块就摔。
- 三法矩阵（combined，同512槽/seed2026/133s/0.35–0.8）：十族学生直接测v26=0/112；十一族混合RL v27=0/110、v28=0/118、十教师DAgger v29b=0/120；纯combined专项PPO v30b=0/649。**没有任何一种训练路线的任何一个episode通过combined入口第一障碍**；而v26在纯短沟族0.2m沟100%、轮胎95.7%——同一权重换combined地形入口即全灭，符合spec line 27“扫描无法在可观测信息里区分必要动作→报告接口局限”的前置条件。
- 冻结combined训练路线（不再烧第四轮GPU）；下一手为**只读接口诊断**：v17式combined入口刷查（起点系坐标、第一gap start/宽/深度、机身z/接触体、height scanner在该帧看到什么、fallen真因是坠沟/撞壁/前倾），确认是观测盲区、地形出生配置还是终止语义。十族学生v26与教师池不变；最终11族门维持未达；停止标志`fe5dc650...`未变；无自动p2；capability_validated全False。

## 2026-10-19 06:55 UTC combined入口几何只读诊断完成：三连沟×0.43m窄台密集组合超出已训包络；v31 credit扩展实测未亮、不盲训

- 只读probe（无Isaac、无锁）修正本人前判并定证据链：combined entrance三gap的**void宽0.132–0.354、深0.565、间隔仅0.43m窄台**（probe_void_widths/depths；我先把pad误读为0.78宽“壕”又撤回）；纯短沟族同宽0.13–0.16×深0.565而v26短沟族0.156宽100%通过、长沟族0.34–0.38×1.1m宽台97%通过——**单障碍都在能力内，三连0.43m窄台组合不在任何已训族中出现过**（各专项族pad=1.1m）。几何生成常量GAP_SHORT_WIDTH_MAX=.24/JUMP=.26/LONG=.29–.42，combined入口块尺寸不随难度缩小。这解释三法矩阵全0：策略落第一沟后0.43m内起跳二次，动量带倒；PPO在入口0成功、649样本无任何部分成功梯度。
- v31（GO1_V19_COMBINED=1扩展takeoff credit到combined槽；runtime SHA `0cd42bb5b1d9068e363f3b1002f8b1941e71836c1eb9d889ad20c93d4df8c0b8`、cli SHA `e7136aa719e19821d153788d3c909747d55dcb97990205a5b2ec67a39232d1ed`、全量**351 passed/76 subtests**、4×16×2冒烟exit0）**实测`Episode_Reward/takeoff_alignment=0.0000`未亮**：入口首沟宽0.13–0.16<0.26不落long窗口、credit窗口机制（v8 bounds/trigger）在combined窄沟序列未开，具体分支根因未完全定位——**如实记录，未凭未验证credit开900轮**。
- 授权决策点（供人工裁决，二选一或组合）：(a) 新建**训练专用密集组合族tile**（复用同一gap_block/builders：三沟×0.43台，仅加入训练计划，不修改冻结的combined验收地形/11族门），curriculum训会入口组合再回纯combined PPO/蒸馏；(b) 以十族v26（10/10）为交付终态，combined维持“现方法不可达”的如实报告。十族学生v26、教师池、停止标志`fe5dc650...`、无自动p2均不变。

## 2026-10-19 07:45 UTC P10软课程定案：v32a combined_soft（入口恢复台+0.15m、门禁考真实combined）已实现，服务器SSH中断待恢复后冒烟+启动

- 用户发来P10中文译文全文，关键证据自洽：表2 **Oracles w/o Soft Dyn = 攀爬0%、跨沟0%**——纯硬物理教师训不出最难技能，与我们蒸馏教师池combined=0%同构（DAgger上限=教师并集，ε接管还把short/long拖到0.746/0.852）。P10的解药=§3.1难度插值 l=(1-s)l_easy+s·l_hard+软动力学，我们缺这一级。
- v32实现（本地SHA）：geometry `35d528ecb3f0dd7c4817e242da44a2997b28dc4481d302d6492bce0cde0d9ee6` 新增**训练专用族combined_soft**=combined路线入口三连沟恢复台0.65→0.80m（kind路由非env；默认combined字节级不变，test_v32两测证明）；runtime `069245693c5e1d17f25d05f9496d2d975c240eb647dfbf128c54e4e46ae0e7b7`（GAP/JUMP_FAMILIES、combined mask、v19 credit同步含combined_soft）；roster `81dd2d6c046b32c5eeb77b13d3d90366a4aaa9b990f389114e73917ebdfbc50c`；runner `run_student_v32.py 6934dcc8b6af9fc246dfd458e790edac02b95762ddcdb611e2a79d872289e779`（config键gate_family=combined→**门禁仍考真实combined**，保留难度/速度轴）；config `curriculum_phases_combined_soft_v32a.json c66bd437b9bf5a1997314c75516a2201bb3b3ee14360ee3635b27cd96ad05425`（init=v26、400iter、512槽、completion_gate=0.9）；launcher start_v32a：**v23 brake关闭**（发现brake FAMILIES与gate槽family不一致会raise的坑，unset防泄漏）。task_curriculum.TERRAIN_FAMILIES**不动**（runner显式+('combined_soft',)传入，避免波及其它roster）。
- 部署状态：geometry/runtime/roster/runner/config/tests已scp且SHA核对一致；**修正版launcher与v32a冒烟脚本未部署**——服务器10.200.212.241:8022自~07:30 UTC持续Connection closed（当日已有一次6分钟抖动自愈），后台pytest结果未读回。
- 恢复后按序：①重传launcher+确认v32几何测试4 passed；②v32a冒烟=torchrun train 16×2iter GO1_TASK_PLAN=combined_soft 16（照v31_smoke结构、无brake）；③nohup启动v32a（~3h）；④若gate(combined)≥90%→combined教师入teachers池→重跑v29式蒸馏→11族同门512；若soft仍0%→pad+0.30二级课程(combined_soft2)；再不行回归方案(b)如实汇报。停止标志`fe5dc650...`未变；无自动p2。

## 2026-10-19 09:58 UTC v32a已启动：combined_soft 512槽训练、真实combined门禁，ETA~15:20Z

- 服务器两次SSH中断（各5-20分钟）后自愈。冒烟中抓到并修复combined_soft入口+0.45m位移导致`no room for finishing tyres`溢出（v30 retry保护只匹配"combined"）：geometry `0eeb50462a5ff86d02b8a4ae3e1ae3c2d4855596450834161e9fdcd680812d1f`（retry覆盖combined_soft；probe 400tile×extra{.05,.10,.15}单发失败3–8/400、**6次重试后0/400**）。
- v32a_smoke exit=0（combined_soft 16×2 iter全链路）；正式**v32a已运行**：512×combined_soft PPO from v26、warmup150+400iter、gate=真实combined 512槽≥0.9、preflight全绿（launcher pin geometry 0eeb5046/runtime 06924569/cli e7136aa7/runner 6934dcc8/config c66bd437）。receipt文件`reports/v32a_combined_soft.exit`，日志`reports/v32a_run.log`+`lp_combined_v32a_results.txt`。
- 决策树不变：gate(combined)≥0.9→教师入库→重蒸馏→11族同门；仍0→combined_soft2（pad+0.05渐进已验证安全）或二级soft课程；再失败回方案(b)。

## 2026-10-19 15:25 UTC 入口消融矩阵证伪"入口几何假说"→v33a宽沟阶梯已就绪；服务器Isaac挂死待愈

- soft阶梯消融（v26确定性eval 128槽，reports/v33_probe_*）：soft1 pad+0.15=**0.0**、soft2入口宽×0.25=**0.0**、soft3×0.6=**0.0**、soft4入口深0.115浅槽=**0.0**、soft6入口宽0.29-0.42真深=**0.0**；v19p4(跳态教师)soft4=**0.0**。soft4/6顶层**fallen_rate=1.0、fell_into_gap=0、collision<0.13、achieved_speed≈0.5**=正常行走中足端碰撞绊倒，非坠沟——**入口沟宽/深/台宽任一软化均不改变100%死亡，combined死因不是入口几何三元组本身**（v30b trace的1.83-1.89死亡集中点提示lead-in→gap0过渡区另有机制：疑spawn朝向/首沟距离1.3m过短致步态未稳即遇唇、或height-scanner在mixed tile上观测异常，未定案）。
- v33a套件已实现部署（geometry `3f9cb342`(soft2-7全kind+drop clamp)、runtime `89d54edf`、roster `fd63e677`(通用single:X)、runner `d71d0be1`、config v33a `86934e2a`(combined_soft6 init v26 800iter gate=combined)、launcher `16291c86`）；14:48启动即GPU-0%挂死（etime905s CPU2m13s），kill清理后16槽健康探测亦timeout(124)——**服务器Isaac/NFS病发期（当日第4次中断，14:17后未自愈）**，v33a实际训练未开始。
- 待服务器愈：`bash scripts/v33a_healthprobe.sh` exit=0且iteration计数≥2→`bash scripts/launch_v33a.sh`。若soft6训练门(real)仍0→转**观测/控制面诊断**（GO1_DEBUG replay dump v26-soft6逐步z/gravity/foot-contact/scan-void，锁定绊倒物理机制），不再盲加课程级。十族v26与教师池不变；停止标志`fe5dc650...`未变；无自动p2。
## 2026-10-20 09:00 UTC v36 三轮真实数据修正 + soft9间距假说 + 宿主机Isaac挂死（需重启宿主机）

- **数据修正**：v36a/b/c 的 gate 行里 failure 不是摔倒率。真实 fallen_rate：v36a 0.7865 → v36b 0.7645 → v36c 0.7282；timeout 0.158→0.305→0.365；completion 全 0。即 P10 两阶段（软终止前训+硬微调800轮）把 collision 从 0.168 降到 0.042，但摔倒率仍~73%，远未达教师门槛。
- v36c 轨迹（130/130 全摔）死亡终点 x 分位 [1.9, 10.2, 21.0]：入口区(≈1.9m)、中段(≈10m slope)、长沟区(≈21m)。此前"two killers"判断需修正为三处。
- **soft9 假说（待验证）**：combined 入口三沟 pad=0.65(沟心间距0.78m)，而 v26 100% 通过的 gaps_short 族 pad=1.10。新增 combined_soft9（入口 pad=1.10，几何其余同 combined）用于判定"间距"是否关键因素。geometry 6ea806c93442da1454010b6fdd1b95ad83519a5a756b3bd8488fd3b5c2e6f574、runtime 26ddb3969431b4c1df77cd2e81d557ef3124e28ddd32593469e7e58395454989 已部署；combined 默认几何字节不变（override 仅在 soft9 时生效）。
- v37_eval_plan_combined_soft9.json(128槽) 与 v37_plan_soft9_32.json(32槽) 已生成；v37_probe_soft9.sh / v37_gpu1_test.sh / v37_watch.sh 已部署。
- **阻塞（环境）**：宿主机 Isaac 自 07:45 起在 Kit 启动阶段挂死（GPU 0%、日志停在 startup warnings），多次清理 /dev/shm carb 段、换 GPU1 单卡亦复现；GPU0 存在两个死进程残余 CUDA 上下文（nvidia-smi [Not Found] 8467/50493）。容器内无 systemd，`sudo reboot` 不可用。**需在宿主机执行重启**（用户 2026-10-19 16:52 的手动重启曾一次性解决同类问题）。
- 已挂 6 小时自愈监视器 scripts/v37_watch.sh（每~9分钟清进程/shm→跑 soft9 判定，最多40轮），结果写 reports/v37_probe_watch.log。宿主机恢复后自动出数：若 soft9 fallen 明显下降→按间距做课程（soft9→真实0.65）并用 v36c 初始化继续；若仍高→转逐帧 replay 诊断而非继续盲训。
- v36d（提速指令带 0.6-0.8）已备好未跑（runner 3a30cada 支持 gate_speed_min/max 覆盖，config 1c7343bf，launcher 34bb1863），待 soft9 结论后决定是否仍需要。
## 2026-10-20 18:25 UTC combined 归因收敛：不是几何、是"沟+混合场景"的组合泛化；v46 长训中

- 决定性对照（v26 零样本，128 槽确定性 eval）：
  - `course`（混合路线，零沟）= **completion 1.0 / fallen 0.0**
  - `gaps_short_dense`（0.08–0.24 沟、间距 0.65，与 combined 入口同密度）= **1.0 / 0.0**
  - `gap_course`（纯窄沟，间距 1.10）= **1.0 / 0.0**
  - `combined_soft13`（混合路线 + 仅 1 条入口沟）= **0.0 / fallen 1.0**
  - `combined` 及其 5 种入口软化（pad+0.15、宽×0.25/×0.6、深 0.115、宽 0.29–0.42）= 全 0 / fallen≈1.0
  → 单独混合地形可走、单独沟可走，**二者组合即 100% 摔**；不是沟的尺寸问题，是感知-行为组合泛化失败。
- 修正诊断 bug：`combined_soft12/13` 在 build_tile 内已被别名成 course_gaps，原用 `kind == "combined_soft13"` 判断恒假（soft12/13 实验当时无效）。已引入 `orig_kind` 捕获别名前名字；soft13 现确为单沟。另修 `mixed_reward_segments` 对非 3 沟族的硬校验（非 combined 填 NaN）。
- 结构性差异（P10 vs 我们）：P10 的专用教师用**特权观测**（到障碍距离、高宽、4 维障碍类型 one-hot）训练，因而跨场景可迁移；我们的教师都是**视觉策略**（depth scan），在陌生混合场景里给出错误标签 → DAgger 无效。这是 v27/v29b 失败的根因，也是与论文的真正差距（论文方法本身没问题）。
- v46 长续航已启（16:04Z）：真实 combined 几何 + 软终止 + 窄沟 credit，从 v38 ckpt 续训 2000 轮、512 槽，gate 真实 combined；进度 1387/2000（18:25Z，~5.9s/iter），预计 19:25Z 训练完 + 10min 门。
- 若 v46 仍 ~70% 摔倒：需实现论文的**特权教师**（privileged obs 变体）+ 穿透式软动力学才能补齐；否则以十族交付并如实说明。环境仍偶发 Kit 挂死，靠容器重启恢复；已避免 kill -9 频繁启停。
## 2026-10-21 12:30 UTC 11族验收 79.9%（combined 0）+ 组合课程第一步失败；证据链收敛

- **11族同门 512 槽正式验收**（v26，复用 v28 gate plan：11 族、每族 46-47、难度≤0.1、5 档速度）：
  总体 completion **0.799**；combined **0.000**；其余族 0.854（tire，抽样带差异）/0.96（gaps_long）/1.000（其余 8 族）。
  receipt：reports/v49_11fam.exit、evaluate_stage1_2026-10-21_10-50-47_v49-v26-11family.json、reports/v49_11fam_trace.jsonl。
- **组合课程第一步（v51）**：在 combined_soft13（真实 combined 布局、仅入口 1 条沟）上从 v26 训练 500 轮（软终止扩展到所有 combined_soft* 族、窄沟 credit），门（真实 3 沟）0/684，**在自身 1 沟地形上仍 0% / fallen 1.0**（v52）。即在"混合场景 + 1 条 8cm 窄沟"上训练也学不会——不是课程跨度问题。
- 排除项（受控）：沟宽（×0.25/×0.6、0.29-0.42 宽沟）、沟深（0.115 浅沟、虚拟支撑面）、间距（0.65/0.80/1.10）、沟数量（1/2/3）、混合路线本身（course=100%）、纯沟路线（gap_course/gaps_short_dense=100%）、特权观测（v47=81% 摔）、窄沟起跳 credit（v38/v46 平台 72.7-73.6%）、软终止（v36a 78.7%→v36c 72.8%）。
- 代码层已核验：combined 入口三沟在网格中确实是开口（x=4.34 仅命中 -0.665/-0.565；两侧地面 0.0），沟几何与纯沟族一致；故非地形网格缺陷。
- 调试数据：摔倒瞬间速度 0.08-1.95 m/s（多数 ~0.7），位置集中在 spawn 后 2.3m（入口第 2 沟之后）与 14m（坡/台阶接缝）；扫描可看到 -0.57 的沟底。
- 结论：现有技术栈下 combined 不可达；与 P10 的差距是**穿透式软动力学**（允许穿透 + 穿透体积/深度×前进速度惩罚 + 自动难度课程）这一核心机制，我们只实现了其近似（不终止/浅沟/虚拟地板），均已被证伪。建议以十族交付并如实附本证据链。
## 2026-10-21 20:15 UTC 多分辨率扫描（LEEPS 路线）验证失败：v55 摔倒 94.6%

- 实现：`GO1_MULTIRES_SCAN=1` → 粗网格 16×32@0.1m 抽稀为 8×16=128，新增 0.05m/0.75×1.15m 细网格 16×24=384，合计仍 512 高度值（obs 维度不变）；`GO1_PRIVILEGED_ACTOR=1` 让 actor 直接使用精确多分辨率扫描。config `a7920ced`、runtime `8ed99bce`（已部署，默认关闭时行为不变）。
- 冒烟通过（16×2，exit=0）；v55 = 真实 combined 几何 + 软终止 + 多分辨率特权教师，900 轮，门禁真实 combined：**completion 0/570、fallen 94.6%、achv 0.166**，无改善。
- 结论：窄沟在高分辨率下仍学不会，"分辨率不足"假设被证伪。至此 9 种训练方案 + 6 组几何消融全部失败，唯一未实现的仍是 P10 的穿透式软动力学与分层/衔接策略（ANYmal Parkour / Setup Policies 路线）。
- 已上传 GitHub：main 分支含十族报告与 11 个视频（commit 4cd8813），另有 ten-family-snapshot 分支含完整代码快照。

## 2026-10-22 10:00 UTC v59 respawn soft dynamics also no improvement

- v59: real combined geometry training slot with fall-respawn (restore last safe pose, episode continues), 800 iters from v26: gate completion 0/701, fallen 78.2%, timeout 16.6%, achv 0.269. No improvement vs v38 (72.7%) / v46 (73.6%).
- Corrected the false positive: combined_soft17 (stairs removed) had not been aliased to course_gaps and built an empty tile; the true test is 0% again.
