# 接口契约（schema 5）

本文件是 `contracts.py`、`models.py` 与 `checkpoints.py` 的行为说明。
**权威定义在代码中**：`contracts.interface_manifest()` 是唯一接口规范，`tests/test_contracts.py`
与 `tests/test_models.py` 逐项校验本文件的约定。

## 1. 时间基与频率

| 名称 | 值 | 说明 |
|---|---|---|
| `PHYSICS_DT` | 0.005 s | 物理仿真 200 Hz |
| `CONTROL_DT` | 0.02 s | 运控策略 50 Hz（每 4 个物理步） |
| `DEPTH_DT` | 0.05 s | 相机/地形 20 Hz（每 10 个物理步） |
| `NAV_DT` | 0.1 s | 导航策略 10 Hz（每 20 个物理步） |
| `MAX_SENSOR_AGE` | 0.5 s | 目标/观测时效上限，超时即视为失效 |

运行时会在构造时断言 `CAMERA_TICKS == 10`、`LOCOMOTION_TICKS == 4`、`NAVIGATION_TICKS == 20`，
任何非整数倍关系直接报错。

## 2. 观测布局

### 2.1 第一层（`LOCO_OBS_DIM = 1073`）

```text
偏移   0 ..   41   p42            角速度3(×0.25) | 投影重力3 | 关节相对位置12 | 关节速度12(×0.05) | 上一动作12(clip±3 /3)
偏移  42 ..   44   cmd3            [vx, vy, wz] / [0.8, 0.3, 1.0]
偏移  45 ..   46   步态相位         sin/cos(2πφ)
偏移  47 ..   48   固定航向编码     sin/cos(初始赛道方向在机体系)
偏移  49 ..  560   高度扫描        512 点，相对机身高度 clip(±1 m)
偏移 561 .. 1072   扫描置信度      512 点，0/1（射线是否命中）
```

### 2.2 第二层（`NAV_OBS_DIM = 193`）

```text
偏移   0 ..  41   p42             同第一层
偏移  42 ..  45   goal4           相对 XY / 10 | 有效性 | clip(age/0.5, 0, 1)
偏移  46 ..  48   上一指令          上一控制步 cmd3 / [0.8, 0.3, 1.0]
偏移  49 .. 120   地形起伏图        6×12 = 72 点高度，相对机身 clip(±1 m)
偏移 121 .. 192   起伏置信度        72 点，0/1（可见性/视锥）
```

### 2.3 Critic（特权）

`critic = actor_obs ‖ privileged_state`，其中 `privileged_state = [机体线速度 3, 四足接触 4]`。

| 层 | actor | critic | GRU 输入 |
|---|---:|---:|---:|
| 第一层 | 1073 | 1080 | 49 + 32 + 7 = 88 |
| 第二层 | 193 | 200 | 88 |

- 第一层常数：`LOCO_PREFIX_DIM = 49`、`TERRAIN_HEIGHT_DIM = 512`、`TERRAIN_FEATURE_DIM = 1024`。
- 第二层常数：`NAV_PREFIX_DIM = 49`、`RELIEF_ROWS = 6`、`RELIEF_COLS = 12`、
  `RELIEF_DIM = 72`、`NAV_GEOMETRY_DIM = 144`。
- actor 部署路径**不读取**任何特权量（`tests/test_models.py` 校验导出图）。

## 3. 动作映射

### 3.1 第一层 → 电机

```text
action[12] ∈ [-1, 1]            （PPO 输出，clip 到 ±3 之前先由策略自身约束）
关节目标 = 默认关节角 + 0.25 · action
再用软限位 clamp，并把“实际下发”的等效动作记录为 previous_action（供 p42 使用）
```

### 3.2 第二层 → 速度指令

```text
vx = (a0 + 1) · 0.4        ∈ [0, 0.8]
vy = a1 · 0.3              ∈ [-0.3, 0.3]
wz = a2                    ∈ [-1, 1]
```

随后：`× robustness.command_gain`（sim-to-real 课程）→ `CommandRouter.update(..., dt=NAV_DT)`：

- 一阶平滑，时间常数 0.2 s；
- 执行器限幅 `vx ∈ [0, MAX_FORWARD_COMMAND = 1.0]`、`vy ∈ ±0.3`、`wz ∈ ±1`；
- 目标失效或过期（`age > MAX_SENSOR_AGE`）时指令**强制归零**，并记录停顿原因；
- 指令按固定时延队列交给冻结的第一层（"延迟的是第一层看到的指令，而不是目标"）。

最后 `_course_speed_scale()` 依据高度扫描的**前方落差**按比例限速（落差 0.25 m 达最大削减 55 %），
非课程地形或扫描无有效点时返回 1.0。

## 4. 地形几何

| 名称 | 值 |
|---|---|
| tile 尺寸 `TILE_SIZE` | 40 m |
| 环境间距 `TILE_SPACING` / `cfg.scene.env_spacing` | 44 m |
| 出生点 `START_IN_TILE` | (3.0, 8.0, 0.0)（tile 局部） |
| 特征区 | 5.30 m → 35.00 m（`COURSE_LEAD_IN_END` / `COURSE_FEATURE_END`） |
| 终点条带（spawn 相对） | 33.10 m → 34.10 m，半宽 4.0 m |
| 目标点 `COURSE_GOAL_OFFSET` | spawn + 34.0 m |
| 高度扫描网格 | `size=(3.1, 1.5)`、`resolution=0.1`、`offset=(1.05, 0, 20)` |

> **坐标系陷阱**：赛道高度函数使用 **tile 局部坐标**（`START_IN_TILE` 为出生点），
> 而奖励与终点使用 **spawn 相对偏移**。两者混用会导致"终点落在网格之外"。

## 5. schema 迁移

`checkpoints.load_checkpoint` 比对 `data["interfaces"]` 与 `interface_manifest()`；
不一致时只接受下列**白名单迁移**：

| 迁移 | 判据 | 处理 |
|---|---|---|
| schema3 → 4（航向列） | `_is_schema3_heading_migration`：仅前缀 47 → 49，其它键一致 | `migrate_locomotion_heading_state`：旧列原位保留、新列零初始化 |
| schema4 → 5（起伏图改名） | `_is_schema4_relief_migration`：规范化（`obstacle → terrain_relief`、删除 `relief_rows/relief_cols/normalization.relief/schema_version`）后**逐项相等**，且 `schema_version == 4` | 键名搬迁，维度必须完全一致（维度变化一律拒绝） |

`_canonical_interfaces()` 是上述比较的实现；`load_actor()` 只在 **schema3** 场景运行航向迁移，
避免对 schema4/5 误用。

## 6. checkpoint 与认证

- 文件为 `*.pt`（`torch.save`，`weights_only=True` 加载）+ 同名 `*.json` 清单；
- 清单包含 `interfaces`、`model_hashes`、`metadata`（含 `reward_config`、`course_difficulty`、
  `speed_range`、`iteration`、`reward_contract`）；
- 加载时校验文件哈希、模型张量哈希、接口一致性与奖励契约一致（续训）；
- `certify_checkpoint(source, report)` 只在门禁记录全部格子达标时产出 `*.validated.pt`，
  并写入 `envelope`、`validation_protocol`、`validation_report_sha256`。
  **未认证的 checkpoint 不允许被上层冻结使用。**
