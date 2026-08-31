# R0 实现方案 — Response-Consistent Locomotion Policy

对应 `docs/project-design-rlmpc-v3-coding.md` R0 的第二份产出。
事实依据见 [rlmpc-v3-r0-facts.md](rlmpc-v3-r0-facts.md)。

**本文档在获得确认前不动代码。**

---

## 0. 已确定的范围边界（2026-08-31 决定）

| 决定    | 内容                                                                          | 影响                                                                                    |
| ------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 地形    | **v1 只做平地**，域随机化靠摩擦/质量/负载/电机强度/arm 构型撑起跨域差异 | 全局不变量 9 在 v1 期间空转；`_reward_jump` 的硬编码 0 暂不致命，但仍按预防性修复处理 |
| 兼容性  | **接受观测/命令布局不兼容，stage-1 从零重训**                           | 旧 checkpoint 只作 R9 的「原版 WTW」对照，不再续训；加载时必须显式报错                  |
| stage-2 | **完全不动**                                                            | `WBCEnv.plan()` 路径保持现状，允许它暂时与新命令布局不一致                            |
| 适应通路 | **不用 teacher-student**（R7.2 被否决）                                  | `dog_ac.py` / `ppo.py` / `rollout_storage.py` 全部不动；改用非对称 actor-critic + 历史窗口 30→50 步补偿，见 §9 |
| 时序编码 | **v1 用非线性历史（flat MLP @ 50 步）**，TCN 设计保留、延后实现 | 现在只需满足 4 条零成本的结构性约束（见 §9 R7.3），保证 TCN 日后是真正的 drop-in；temporal attention 不采用 |
| 观测边界 | **新增观测必须机上可无歧义复现**，误差源必须进 DR | `(g,l)` 只对 IMU 锚定通道（pitch/yaw_rate）启用；vx/vy/height 版本不进观测，见 §9b |

由此推出的两条工程约束：

- 所有改动集中在 **stage-1 / dog 侧**。凡是需要动 `wbc_env.py` 的，只动
  stage-1 会经过的代码路径（arm 扰动、privileged obs、reset hook），
  不碰 `plan()` / goal_reaching / trajectory 分支。
- 新增的 obs 通道必须进 `core.py` 的 `dog_obs_dim_parts`，
  让宽度推导保持单点来源；旧 checkpoint 的形状不匹配由现有的
  `_load_matching_state_dict` 机制升级为**显式报错**。

---

## 1. 代码组织

### 新包：`go1_gym/response/`

> ⚠️ **位置已修正**（2026-08-31 实现时）。原定 `go1_gym/envs/roboduet/response.py`
> **不可行**：import 该包任一子模块都会执行 `go1_gym/envs/roboduet/__init__.py`，
> 它 import `wbc_env` → IsaacGym，正好摧毁拆分这些类的全部理由。
> `go1_gym/utils/` 同样不干净（`__init__` 经 `math_utils`/`terrain` 拉入 IsaacGym）。
> `go1_gym/__init__.py` 只有 `import os`，所以新建顶层子包 `go1_gym/response/`，
> 一个类一个文件。

四个纯张量状态机，**不依赖 IsaacGym**，可在 CPU 上单元测试：

| 文件 | 类 | 职责 | 需求 | 状态 |
| --- | --- | --- | --- | --- |
| `reference.py` | `ReferenceModel` | 5 通道二阶临界阻尼 + 速率饱和参考轨迹（精确 ZOH） | R2 | ✅ **已实现，22 项测试通过** |
| `residual.py` | `PhaseResidualEstimator` | `(env, 速度桶, 相位桶, 通道)` 残差估计 + 双路去趋势 | R3 | 待实现 |
| `groups.py` | `ConsistencyGroups` | env 分组、twin 指派、同步/失效掩码 | R5 | 待实现 |
| `excitation.py` | `ExcitationSampler` | PRBS / chirp / ramp 命令生成 | R6 | 待实现 |

**为什么另起模块**：`legged_robot.py` 已 2840 行、`wbc_env.py` 2808 行；
更重要的是 R2/R3/R4 的验收条件是**数值测试**（与解析解在 1e-4 内一致、
限幅位置、人工轨迹的奖励排序），这些必须能脱离仿真跑。
把它们放进 IsaacGym 类里就没法验收。

配套测试与被测代码同目录：`go1_gym/response/test_*.py`（pytest，CPU-only）。

> ⚠️ **不要放在顶层 `tests/`**：`.gitignore:84` 的 `tests` 规则匹配**任意层级**的
> 同名目录（`tmp` 同理），这两个目录是留给临时脚本的、从未被跟踪。
> 验收测试是交付物，必须入库，所以放进包内——`test_*.py` 文件名不受该规则影响。

`pytest` 需装进 `isaacgym` 环境（`pip install pytest`，已完成）。

**测试必须做变异验证**：一次就全绿的测试套件没有价值。R2 的四个变异
（半隐式欧拉 / 去掉 clip / 命令跳变时 reset / pitch 带宽 ≥ height）
已逐个确认会被捕获。

### 新配置节：`cfg.response`

所有新超参进配置（R4 不变量），集中在一个新 section，
通过 `config/wbc.py` 的 `RESPONSE_OVERRIDES` 表注入，
`allow_new=True` 已被 `ROBODUET_PROFILE` 支持（[config/wbc.py:614-618](../go1_gym/envs/config/wbc.py#L614-L618)）。

结构草案：

```
cfg.response.channels          # 5 通道的 (name, cmd_idx, omega_n, rate_limit, sigma, weight)
cfg.response.residual          # 速度桶边界、相位桶数、EMA 时间常数、低通时间常数
cfg.response.groups            # group_size、twin 策略、失效条件
cfg.response.excitation        # 辨识环境比例、PRBS/chirp/ramp 参数
cfg.response.curriculum        # R8 四阶段的 iteration 边界与权重 ramp
cfg.response.deviation_stats   # (g,l) 的 EMA 时间常数(~5s)、启用通道、warmup 门控
```

标定产物（R8.2）写成独立文件 `configs/reference_model_<tag>.yaml`，
由 `cfg.response.reference_model_file` 指向，训练启动时加载覆盖起点值。

---

## 2. 5 个决策通道的定义（贯穿 R2/R3/R4/R5）

| ch | 名称       | `commands_dog` 索引 | 命令 u           | 实测量 y（body frame）                                                                                                         |
| -- | ---------- | --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 0  | `vx`     | 0                     | 前向速度         | `base_lin_vel[:, 0]`                                                                                                         |
| 1  | `vy`     | 1                     | 侧向速度         | `base_lin_vel[:, 1]`                                                                                                         |
| 2  | `wyaw`   | 2                     | 偏航角速度       | `base_ang_vel[:, 2]`                                                                                                         |
| 3  | `height` | **5**           | body height 增量 | `base_pos[:,2] − ref_h − cfg.rewards.base_height_target`                                                                   |
| 4  | `pitch`  | **3**           | body pitch       | `self.pitch`（`check_termination` 里已算好，[legged_robot.py:414-415](../go1_gym/envs/roboduet/legged_robot.py#L414-L415)） |

⚠️ **通道序号 ≠ 命令索引**（pitch 在命令里是 3，height 是 5）。
用一张显式索引表 `CHANNEL_CMD_IDX = [0, 1, 2, 5, 3]`，不允许在别处硬编码。

`ref_h` 统一走一个 helper `self._terrain_reference_height()`，
平地下返回 0，启用地形后返回 `measured_heights` 均值。
**顺带修掉 `_reward_jump` 和 `_reward_feet_contact_vel` 的硬编码 0**
（不变量 9 的预防性修复，平地下行为不变，可用回归测试证明）。

---

## 3. R1 — 命令空间裁剪

### 改动位置

**`config/wbc.py`** 新增 `RESPONSE_COMMAND_OVERRIDES`：

```
commands.gait_frequency_cmd_range = [2.5, 3.5]      # 半自由，窄范围
commands.limit_gait_frequency     = [2.5, 3.5]
commands.body_roll_range          = [0.0, 0.0]      # 冻结
commands.limit_body_roll          = [0.0, 0.0]
commands.footswing_height_range   = [0.06, 0.07]    # 固定 + 小幅随机
commands.stance_width_range       = [0.28, 0.32]
commands.stance_length_range      = [0.42, 0.46]
commands.gait_duration_cmd_range  = [0.49, 0.51]
# 课程网格：5 决策通道有 bin，其余为 1
commands.num_bins_body_pitch  = 5
commands.num_bins_body_height = 5
commands.num_bins_gait_frequency  = 1
commands.num_bins_footswing_height = 1
commands.num_bins_stance_width  = 1
commands.num_bins_stance_length = 1
commands.num_bins_gait_duration = 1
commands.num_bins_body_roll     = 1
```

网格从 1,964,655 降到 **21×3×21×5×1×5 = 33,075** bin，
相对 4096 环境是合理量级（[facts §3](rlmpc-v3-r0-facts.md)）。

⚠️ `enable_dyna_gait` 会用 `--dyna_gait_min_frequency` 覆盖
`gait_frequency_cmd_range[0]`（[core.py:525](../go1_gym/envs/config/core.py#L525)）。
必须让 `RESPONSE_COMMAND_OVERRIDES` 在 `enable_dyna_gait` **之后**应用，
或改用 `--dyna_gait_min_frequency 2.5` 并单独收紧上界。
方案：在 `build_roboduet_config` 的 feature-enable 段之后加一个
`apply_response_overrides(cfg)` 钩子，明确排在最后。

**`legged_robot.py::_resample_commands`**：

- gait frequency（idx 6）走**独立均匀采样**，不从 curriculum 取值
  （R1 不变量：不进课程）。
- 冻结通道（roll=4, footswing=7, stance=8/9, duration=10）在采样后
  用固定值/窄均匀覆写。
- 现有的「速度模长 < 0.1 → gait_frequency = 0」特判
  （[L1354-1356](../go1_gym/envs/roboduet/legged_robot.py#L1354-L1356)）
  必须保留还是移除？**保留**——它是站立行为，与冻结无关；
  但要记录到 R9 导出的数据里，否则残差模型会在站立段看到"频率跳到 0"。

### 实现结果（2026-08-31，已完成）

**冻结几乎全部由配置达成，`_resample_commands` 的写入逻辑一行没改。**
原因：1-bin 的课程维度在其（现已收窄的）`limit_*` 区间内均匀采样
（[base/curriculum.py:31-42](../go1_gym/envs/base/curriculum.py#L31-L42)、
[:81-84](../go1_gym/envs/base/curriculum.py#L81-L84)），
所以「窄化 limit + bins=1」既冻结了通道、又把它排除出自适应课程，
同时保留其在命令向量中的位置。

落地的两张表在 [config/wbc.py](../go1_gym/envs/config/wbc.py)：
`RESPONSE_COMMAND_OVERRIDES`（总是适用）与
`RESPONSE_GAIT_COMMAND_OVERRIDES`（仅 `use_dynamic_gait`），
由 `core.apply_response_overrides(cfg)` 在**所有 `enable_*()` 之后**应用。

实测组合结果：

| 通道 | 采样范围 | limit | bins | 分组 |
|---|---|---|---|---|
| vx / vy / yaw_rate | [-0.5,0.5] / [-0.3,0.3] / [-1,1] | [-1.5,1.5] / [-1,1] / [-2,2] | 21 / 3 / 21 | 决策 |
| body_pitch | [-0.4, 0.4] | [-0.4, 0.4] | **5**（原 1） | 决策 |
| body_height | [-0.2, 0.3] | [-0.2, 0.3] | **5**（原 1） | 决策 |
| body_roll | **[0, 0]** | **[0, 0]** | 1 | 冻结 |
| gait_frequency | **[2.5, 3.5]** | **[2.5, 3.5]** | 1 | 半自由 |
| footswing_height | [0.06, 0.061] | 同 | 1 | 冻结 |
| stance_width | **[0.28, 0.32]**（原 [0.10,0.45]） | 同 | 1 | 冻结 |
| stance_length | **[0.42, 0.46]**（原 [0.25,0.45]） | 同 | 1 | 冻结 |
| gait_duration | [0.49, 0.5] | 同 | 1 | 冻结 |

**网格 1,964,655 → 33,075**；`dog_num_commands` 仍 11、
`dog_num_observations` 仍 **90**（R1 要求"观测维度与原版一致"✅）。

### 顺带做的重构

课程的网格定义、初始激活窗口、扩张邻域从 `LeggedRobot._init_command_distribution`
（原 130 行）抽成 [base/curriculum.py](../go1_gym/envs/base/curriculum.py) 的纯函数
`command_curriculum_kwargs / _bounds / _local_range / build_command_curriculum`。

**动机不是整洁，是测试有效性**：这些函数不依赖 IsaacGym，验收测试因此可以跑
**真实的**配置和课程采样器，而不是测一份副本。测试里另有一条
`test_extracted_builders_match_the_original_inline_code`，
把 commit `9e7e995` 的原始字面代码逐字抄进测试做等价对照，防止搬运时抄错。

顺带把 `np.int` 改成 `int`（numpy 1.23.5 里是已弃用别名，行为完全相同）。

### 验收结果

[go1_gym/response/test_command_layout.py](../go1_gym/response/test_command_layout.py)
——**17 项通过**（含 dyna_gait 开/关两种构建、覆盖顺序回归、抽取等价性）。

真环境 rollout（64 env × 1200 步，跨 2 次以上重采样）：

- `body_roll` 恒等于 **0.0000**（硬冻结）
- `footswing/stance_width/stance_length/gait_duration` 全部落在各自窄带内，
  且**只在重采样点变化**、episode 内恒定
- `gait_frequency` 行进时 **[2.501, 3.497]**
- 决策通道正常变化

> ⚠️ **验收断言写法的坑**：R1 原文是"固定（**可加小幅随机化**用于鲁棒性）"。
> 最初把断言写成"整段 rollout 内数值恒定"会误报——窄带随机化是每 episode
> 重采样一次的。正确断言是「带内 + 只在重采样点变化」。

### 200 iter 回归对照

| | 基线（R1 前） | R1 后 |
|---|---|---|
| 200 iter 总耗时 | 287.5 s | 238.8 s |
| 末段 mean reward | 0.238 | 0.581 |
| 末段 mean episode length | 806 | 823 |
| 报错 / NaN | 无 | 无 |

> ⚠️ **不要读这张表里的 reward 行。** 本文档早期版本据此推测 R1 提升了奖励，
> 那是过度解读。第 3 步做了种子方差测量：**同一份代码、三个固定种子，
> 200 iter 的 mean reward 是 0.2275 / 0.7408 / 2.6352——散布 11.6 倍**。
> 这个方差完全淹没上表的差异。200 iter 的奖励水平**没有比较价值**。
>
> 可用的 200 iter 门限只有：跑通 200 iter、无报错无 NaN、
> 新指标在位、单步耗时正常（实测三个种子 233/238/231 s，±1.5%，这个是稳的）。
> 任何关于奖励的结论都要留到 R9 的多种子对照实验。

---|---|---|
| 200 iter 总耗时 | 287.5 s | **238.8 s** |
| 末段 mean reward | 0.238 | **0.581** |
| 末段 mean episode length | 806 | 823 |
| 报错 / NaN | 无 | 无 |

奖励的差距不小，且与 R1 的设计意图一致：原命令空间里存在大片**基本走不了**的角落
——gait frequency 可采到 0 Hz（无步态）、stance width 可采到 0.10 m（极窄站姿）、
body roll 可达 ±0.4 rad。裁剪掉这些之后，同样的奖励函数下策略自然拿到更高分。
**但这只是"一致"，不是因果证明**：种子不同、GPU 负载不同，正式结论要等 R9 的对照实验。

---

## 4. R2 — 参考响应模型

### `ReferenceModel` 语义

状态 `ξ (num_envs, 5)`、`ξ̇ (num_envs, 5)`。连续系统：

```
ξ̈ = ωₙ²(u − ξ) − 2ζωₙ·ξ̇          ζ = 1.0
ξ̇ ∈ [−ṙmax, +ṙmax]
```

**离散化用精确 ZOH 转移矩阵，不用欧拉。**（2026-08-31 实现时修正）
对重根 `λ = −ωₙ`，`A + ωₙI` 幂零，于是

```
s = ωₙ·dt ,  e = exp(−s)
Φ = e · [[1+s,        dt   ],
         [−ωₙ²·dt,    1−s  ]]
```

因为常值 u 下的平衡点是 `(ξ, ξ̇) = (u, 0)`，Φ 作用在**偏差态**
`z = [ξ−u, ξ̇]` 上：`z⁺ = Φ z`，然后对 ξ̇ 做 clip。
**这样命令跳变只是换平衡点，不碰状态**——不变量 1 由结构保证，而非靠约定。

> ⚠️ **本条修正了本文档早期版本的错误。** 早期版本写的是半隐式欧拉，
> 并称"离散误差远小于 1e-4"。**实测不成立**：ωₙ=5、dt=0.02 下半隐式欧拉
> 对 0.4 rad 阶跃的最大误差是 **1.27e-2**，超 R2 验收阈值 **127 倍**。
> 精确 ZOH 在 float32 下误差 **3.3e-7**，余量 299 倍。
> 见 `tests/test_response_reference.py` 与变异测试记录。

第二个理由同样重要：**MPC 会用自己的步长离散同一个连续系统**。
用精确 ZOH，训练期参考与规划期预测在任何步长下都严格一致；
用欧拉则二者相差 O(dt)，而这个偏差与真正的建模误差无法区分。

⚠️ **速率饱和作用在 ξ̇ 上，不是在 a 上**——文档公式 `ξ̇ ∈ [−ṙmax, +ṙmax]`
明确是对速度的箱式约束，这样 MPC 侧仍是"线性系统 + 箱式约束"= QP。
**由此产生的一个必须写明的后果**：饱和期间 ξ 并不严格以 ṙmax 匀速推进，
而是遵循受约束线性系统，位置增量可比 `ṙmax·dt` 高几个百分点。
这正是 MPC 会复现的行为（MPC 里状态受箱式约束的线性动力学就是这个），
所以策略必须按它训练，而不是按"理想匀速斜坡"训练。

起点值来自需求文档表格，`ζ=1.0`，
**约束 `ωₙ[pitch] < ωₙ[height]`（5.0 < 7.0）在配置校验里断言**（R2 不变量）。

### 挂载

| 事件       | 做什么                                                        | 位置                                                                                                                   |
| ---------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 构造       | 分配 ξ、ξ̇，替换`dog_vel_ref`                            | `_init_buffers`，[legged_robot.py:1914-1917](../go1_gym/envs/roboduet/legged_robot.py#L1914-L1917)                    |
| 每步积分   | 替换`_update_dog_vel_ref()` 为 `_update_response_state()` | 调用点[legged_robot.py:380](../go1_gym/envs/roboduet/legged_robot.py#L380)，**已在 `compute_reward()` 之前** ✅ |
| 命令重采样 | **什么都不做**（不变量）                                | `_resample_commands` 内不得触碰 ξ                                                                                   |
| reset      | ξ ← 当前实测 y，ξ̇ ← 0                                   | 替换[legged_robot.py:445](../go1_gym/envs/roboduet/legged_robot.py#L445) 的置零                                         |

⚠️ **reset 时序陷阱**：`reset_idx` 在 `_reset_root_states()` 之后才做记账，
此时 `base_lin_vel` / `base_pos` 等缓冲区**还是重置前的值**——它们在
`post_physics_step` 开头才刷新（[legged_robot.py:361-365](../go1_gym/envs/roboduet/legged_robot.py#L361-L365)）。
所以「对齐实测」要么用 `_reset_root_states` 写入的目标值直接推算，
要么延后到下一步的 `_update_response_state()` 里用一个 `pending_align` 掩码处理。
**采用后者**：更简单，且只差一步（0.02 s）。

### 实现结果（2026-08-31，已完成）

接入点与计划一致，但有两处实现时的修正：

**1. 调用顺序比计划更靠前。** `_update_response_state()` 放在
`check_termination()` 之后、**`_update_performance_metrics()` 和 `compute_reward()`
两者之前**。计划只要求"在 compute_reward 之前"，但指标也要读 ξ——晚一步会让
记录下来的跟踪误差整体偏移一个控制步，看起来像一个并不存在的常数滞后。

**2. 饱和必须是真正的匀速斜坡（重要）。** 最初按"精确线性映射 + 端点 clip ξ̇"
实现，真环境实测发现命令阶跃时**位置单步增量可达 `ṙmax·dt` 的 1.4 倍**。
两个问题：① 违背连续时间语义（`|ξ̇| ≤ ṙmax` 逐点成立 ⇒ 任意区间内
`|Δξ| ≤ ṙmax·dt`）；② 违背量随 dt 变化，**正好摧毁选用精确 ZOH 的理由**。
而且 R2 的设计依据原文就是"大命令自动**退化为匀速斜坡**"。

改为分段更新，两段都精确：

```
未饱和 → 精确 ZOH 映射（线性系统精确）
已饱和 → xi += clip(rate) * dt（斜坡精确）
```

只有跨边界的一两步带 O(dt) 误差（实测超出 0.2%）。
真环境复测：五个通道的位置增量精确等于 `ṙmax·dt`，峰值速率精确等于各自限幅。

### R4.1 不入奖励，先进性能指标

第 3 步的验收是"训练曲线与基线无差异"，所以参考跟踪误差**不能**作为奖励接入。
计划里写的"权重 0"行不通——`_prepare_reward_function` 会直接丢弃 scale 为 0 的项
（[legged_robot.py:2021-2023](../go1_gym/envs/roboduet/legged_robot.py#L2021-L2023)），
根本不会注册。改为走性能指标通道（`_update_performance_metrics`，不经过任何奖励
缩放），每通道记录 `perf_ref_mae_<name>_<unit>`。
这样把参考模型变成奖励的那一步有 before/after 基线可比。

### 删除### 删除

`_reward_response_consistency`、`dog_vel_ref`、`response_consistency_T`
及其 `reward_scales` 条目全部删除。
`benchmark/dog_policy/evaluation.py` 中读 `dog_vel_ref` 的三处
（L1017、L1050-1053、L1151-1152）改为读新参考状态，指标名沿用
`response_consistency_rmse` 以保持历史结果可比。

### 验收落地

`tests/test_response.py`：

1. 0.4 rad pitch 阶跃、ωₙ=5、ζ=1、无限幅，与 `0.4[1−(1+ωₙt)e^{−ωₙt}]` 比对，
   对照点 t=0.1→0.036、0.2→0.106、0.4→0.238、0.8→0.363，容差 1e-4。
2. 0.6 rad 阶跃 + ṙmax=0.8，断言峰值 |ξ̇| ≤ 0.8（未限幅解析峰值 `Aωₙ/e = 1.103`）。
3. 命令在第 N 步跳变，断言 `|ξ[N] − ξ[N−1]|` 与 `|ξ[N−1] − ξ[N−2]|` 同量级（连续无跳变）。

---

## 5. R3 — 相位条件残差估计与零延迟去趋势

### 数据结构

`delta_hat` shape `(num_envs, n_speed_bins=4, n_phase_bins=16, 5)`，float32。
4096 env → **5.2 MB**，无显存压力（[facts §13](rlmpc-v3-r0-facts.md)）。

- **相位桶**：`floor(gait_indices * 16)`，`gait_indices` 见 [facts §4](rlmpc-v3-r0-facts.md)。
- **速度桶**：按 **命令** 速度 `‖commands_dog[:, :2]‖` 分桶（边界如 `[0, 0.15, 0.35, 0.6, ∞)`）。
  用命令而非实测：命令是 MPC 已知量，且避免估计量与实测形成反馈环。
- **per-environment**，不跨 env 共享（R3 不变量）。

### 双路径（R3 不变量 4）

```
更新路径（允许滞后）：
    y_lp   ← y_lp + (y − y_lp)·dt/T_lp          T_lp ≈ 0.5 s（≫ 步态周期 0.33 s）
    resid  = y − y_lp
    delta_hat[e, s, b, :] ← (1−α)·delta_hat + α·resid    α 对应 T_est ≈ 5 s

使用路径（零延迟）：
    y_detrended = y − delta_hat[e, s, b, :]      # 相位查表，无滤波
```

### 挂载

在 `_update_response_state()` 内，**在参考模型积分之后、compute_reward 之前**，
顺序：读实测 y → 查表得 `y_detrended` → 更新 `y_lp` 和 `delta_hat` → 存到
`self.response_state` 供奖励读取。

⚠️ **相位必须是当前步的**。`_step_contact_targets()` 在
`_post_physics_step_callback` 里调用（[L1508](../go1_gym/envs/roboduet/legged_robot.py#L1508)），
早于 `_update_response_state()`（L380），顺序正确。

### 收敛门控

`delta_hat` 需要一个 per-(env, bin) 的样本计数 `delta_count`，
未达最小样本数的桶在奖励里**跳过该项**（R3 不变量：依赖此估计量的奖励项
在训练早期不得启用）。与 R8 阶段 1「估计量开始更新但不用于奖励」配合。

### 实现结果（2026-08-31，已完成）

落在 [go1_gym/response/residual.py](../go1_gym/response/residual.py)。
接口刻意分成两组，让"更新路径 / 使用路径分离"这条不变量由 API 形状保证，
而不是靠约定：`lookup / detrend / is_converged` 只读，`update` 只写。
测试 `test_use_path_never_mutates_state` 直接断言只读性。

环境侧的调用顺序（[legged_robot.py](../go1_gym/envs/roboduet/legged_robot.py)
`_update_response_state`）：**先 detrend，后 update**。
去趋势必须用本步样本**折入之前**的估计量，否则两条路径又合流了。

`active` 掩码排除两类样本，二者都会污染估计量：

- **站立**。`_resample_commands` 在速度命令 < 0.1 时把步频强制为 0，相位时钟停摆
  （实测占 17–24% 的样本）。所有样本会堆进 env 恰好冻结在的那一个相位桶。
- **刚 reset**。低通还持有上个 episode 的水平，机器人也还在稳定中。

### 验收结果

单测 14 项（合计 64 项全过）+ 三条变异测试：

| 变异 | 结果 |
|---|---|
| detrend 改用低通（毁掉零延迟） | 阶跃后 0.2 s 误差 0.1317 vs 正确实现 **0.0004** → 捕获 |
| δ̂ 跨 env 共享 | 双 env 幅值比 1.00 vs 应为 3.00 → 捕获 |
| 取消速度分桶 | 慢速桶读出 0.0483（应为 0.010）→ 捕获 |

**真环境验收**用一个已训练的行走策略（`stage1_v3_204746_seed2793`），
固定 vx=0.5 跑 60 s，每个相位桶中位 177 次访问。
δ̂ 的谐波分布物理上完全正确：

| 通道 | 主导谐波 | 为什么 |
|---|---|---|
| vx | **h2=60%, h4=26%** | trot 每周期两条对角腿各蹬一次 → 前向速度以 **2×步频**振荡 |
| height | **h2=72%**, h1=27% | 同理，每周期两次触地 |
| pitch | **h1=98%** | 前后差动每周期交替一次 |
| wyaw | h1=86% | 同上 |
| vy | h1=69%, h2=19% | 左右交替，每周期一次 |

> ⚠️ **验收判据偏离了需求文档的字面表述，理由是物理。**
> R3 写的是"1–2 次谐波为主 > 70%"，但对 vx 这不成立：它的能量合法地落在
> **偶次**谐波 h2/h4 上，1–2 次占比在 68–72% 之间抖动、会随机跳门槛。
> 判据实际想要的性质是"干净的低阶周期波形，而非噪声"，
> 因此改为 **h1–h4 占比 ≥ 85%**（实测 vx 98%、height 100%、pitch 100%、
> wyaw 97%、vy 99%），h1+h2 仍作为参考量打印。

**去趋势当前只能带来 −3% 到 −12% 的跟踪误差改善**（wyaw −11.6% 最大）。
这符合预期且不代表去趋势没用：现在这个策略是按跟踪**原始命令**训练的，
参考跟踪误差由趋势失配主导（vx 的 MAE 0.088，而步态纹波仅 0.042 峰峰值），
去趋势最多能削掉 ~20%。等 R4 把跟踪 ξ 变成目标之后趋势误差收缩，
纹波占比才会上升——那时去趋势才吃重。

### 可复现的验收脚本

三条 runtime 验收合并进 [scripts/check_response_runtime.py](../scripts/check_response_runtime.py)
（`--check {r1,r2,r3,all}`，R3 需 `--policy <ckpt>`）。
放进 `scripts/` 而不是临时目录，是因为 R1/R2/R3 的验收证据必须可复现——
留在 scratchpad 里等于没有。

---

## 6. R4 — 奖励设计

四个新 `_reward_*` 进 [rewards.py](../go1_gym/envs/rewards/rewards.py)，
走现有的自动接线机制（[facts §5](rlmpc-v3-r0-facts.md)）。

### R4.1 `_reward_ref_tracking`（正项，进 `r_pos`）

```python
err = y_detrended - xi                       # (num_envs, 5)
per_ch = torch.exp(-err**2 / sigma**2)       # sigma 每通道独立
return (per_ch * channel_weights).sum(-1) * soft_gate
```

**软目标机制（必须实现）**：`soft_gate` 是一个 `(num_envs,)` 的 0/1 掩码，
检测到大扰动或严重打滑时置 0 并保持约 0.5 s（25 步）。
可用信号（都已存在）：

- 打滑：`_reward_feet_slip` 的原始量 —— 接触足的水平速度平方和
  （[rewards.py:277-283](../go1_gym/envs/rewards/rewards.py#L277-L283)）
- 扰动：`push_robots` 触发时刻（v1 关闭，R8 阶段 4 打开）+
  base 线加速度 `‖(base_lin_vel − last)/dt‖` 超阈值

用一个倒计时缓冲 `soft_gate_timer (num_envs,)` 实现"失效 0.5 s"。

### R4.2 `_reward_phase_var`（负项，进指数因子）

```python
resid = y - y_lp                             # 实测围绕低频分量的振荡
return ((resid - delta_hat_lookup)**2 * w).sum(-1) * step_mask_after
```

`step_mask_after`：命令阶跃后 `2/ωₙ` 内为 0（每通道独立，用
`steps_since_cmd_change` 计数器）。需要一个 `(num_envs,)` 的
"距上次命令跳变的步数"缓冲，在 `_resample_commands` 里清零。

### R4.3 `_reward_steady_gain`（负项）

```python
return ((y_detrended - u)**2 * w).sum(-1) * (steps_since_cmd_change > 3/omega_n/dt)
```

### R4.4 保留并降权原姿态项

`reward_scales.orientation_control`: **−5.0 → −0.75**（15%）。
`reward_scales.jump`: 10.0 → 保留（它是 height 的瞬时项，同理降权到 ~1.5–2.0）。
**不删除**（R4.4 明确要求）。

### σ 标定（R4 的强制项）

`scripts/calibrate_reward_sigma.py`：
对每个通道，构造参考轨迹 `ξ(t)` 与「ωₙ 加倍」轨迹 `ξ₂(t)`，
在瞬态窗口 `[0, 4/ωₙ]` 上计算
`J(σ) = ∫exp(−(ξ−ξ)²/σ²) − ∫exp(−(ξ₂−ξ)²/σ²)`，
扫 σ 使 `J(σ)` 达到最大可能差值（= 窗口长度）的 30–50%。
**每通道独立**，结果写进 `configs/reward_sigma_<tag>.yaml`。
文档给的 pitch 通道参考区间 0.08–0.12 用作 sanity check。

### ⚠️ 权重量纲的坑

`scale × dt(0.02)`，负项再除以 `sigma_rew_neg = 0.02`（[facts §5](rlmpc-v3-r0-facts.md)）。
两者恰好抵消，所以**名义权重 w 在指数里的有效系数就是 w**。
这不是设计，是巧合；要在配置注释里写死，否则任何人改
`sigma_rew_neg` 都会静默地把所有一致性权重放大/缩小。

### 实现结果（2026-08-31，已完成）

奖励数学放在 [go1_gym/response/reward_terms.py](../go1_gym/response/reward_terms.py)
（纯张量函数、可在合成轨迹上单测），`rewards.py` 里只留三个薄封装。
所有函数返回**非负**量，符号由 scale 提供——这在乘性结构下比通常更要紧。

σ 标定在 [go1_gym/response/calibration.py](../go1_gym/response/calibration.py)，
脚本 [scripts/calibrate_reward_sigma.py](../scripts/calibrate_reward_sigma.py)。
纯计算、毫秒级、无需仿真。标定结果：

| 通道 | 步幅 | 窗口 | 峰值差 | **σ** |
|---|---|---|---|---|
| vx | 0.50 | 0.500 s | 0.1876 | **0.1546** |
| vy | 0.30 | 0.667 s | 0.1081 | **0.0870** |
| wyaw | 1.00 | 0.500 s | 0.3603 | **0.2883** |
| height | 0.25 | 0.571 s | 0.1090 | **0.0921** |
| pitch | 0.40 | 0.800 s | 0.1443 | **0.1152** |

**`target_discrimination` 定为 0.45 而非区间中点 0.40**，两条理由指向同一处：
0.45 是让本流程复现需求文档自己给出的 pitch σ 区间（0.08–0.12）的取值，
0.40 会标出 0.127、落在区间外；且文档的诊断表列了"σ 太大缺乏区分度"这个失效
模式，偏小的一侧更安全。

**交叉验证**：文档举的算例（0.4 rad 阶跃、ωₙ=5）被逐项复现——
参考在 t=0.1 是 **0.0361**（文档 0.036）；~4× 激进策略在 t=0.1 的误差
**0.2015**（文档 0.214）、奖励 **0.0469**、最低 **0.0065**（文档"0.01 量级"）。

### 权重是量出来的，不是拍的

乘性结构下拍权重最容易出事。因为 scale 乘 dt(0.02)、负项又除以
`sigma_rew_neg`(0.02)，**惩罚项的权重就是它在指数里的系数**，
于是 `weight = -ln(衰减) / 实测项均值`。实测（40 s、64 env、已训练策略）：

| 项 | 实测均值 | 定的权重 | 对应衰减 |
|---|---|---|---|
| `ref_tracking` | 0.515（正项） | **+2.0** | 任务项 |
| `phase_variance` | 0.00356 | **−20.0** | ~7% |
| `steady_gain` | 0.222 | **−0.5** | ~10% |

R4.4 按字面执行：`orientation_control` −5.0 → **−0.75**、
`jump`（名字是 jump，实为 body height 跟踪项）10.0 → **1.5**，均为 15%，**不删除**。

### 软目标门限：一个被低估的放大效应

R4.1 的软门最初按"打滑 > 0.5 m/s"设，结果**门被关掉 36% 的时间**——
因为每次触发要保持 25 步（0.5 s），**在 x% 的步上触发就会关掉约 25x% 的时间**。
实测行走时的打滑分位数：p50 0.057、p90 0.142、p99 0.727、p99.9 **1.665** m/s
（尾部是触地/离地瞬态，不是真打滑）；水平加速度 p99.9 只有 8.4 m/s²。

R4 说的是"**大**扰动或**严重**打滑"，所以阈值必须落在尾部：
打滑 **1.5 m/s**、加速度 **20 m/s²**，实测触发率 0.14% / 0.02%，
门关闭时间降到 **4.4%**。

门关闭时该项取 **0** 而不是 1：给满分等于**为被扰动付钱**，会制造刷分动机。
取 0 则策略在恢复期间既拿不到一致性奖励、也没有梯度要求它拿稳定性去换，
这正是"受极端扰动时优先恢复稳定"的操作化含义。

### 验收结果

单测 19 项（合计 83 项全过）。三条人工轨迹的排序**取决于窗口**，这一点值得记录：

| 窗口 | 跟随 | 迟钝 | 激进(2×) |
|---|---|---|---|
| 瞬态段 0–2/ωₙ | 1.000 | **0.659** | 0.439 |
| 全窗口 10/ωₙ | 1.000 | 0.632 | **0.815** |

> ⚠️ R4 要求的 `跟随 > 迟钝 > 激进` **只在瞬态窗口成立**，长窗口反转。
> 原因是奖励对误差是对称的：激进策略一旦收敛就与"跟随"不可区分，
> 而迟钝策略还在追。**聚合到整个 episode，超前并不比滞后受罚更多。**
> R4 没有指定窗口，本实现按 σ 标定所用的同一个瞬态窗口验收
> （文档 R4.1 的叙述也全部围绕瞬态）。
> 若这一点在训练中确实要紧，修法是把误差改成非对称的——那是超出 R4 文本的
> 设计变更，因此这里只把现状钉进测试，不擅自改。

### ⚠️ 1000 iter 验收暴露的问题：R4.2/4.3 不能从第 0 iter 满权重打开

第一次验收把三项全部按标定权重打开，结果**总奖励被压掉约 1000 倍**：

| | 原始项值 | 指数 | 总奖励乘子 |
|---|---|---|---|
| 对**已训练**策略实测（权重就是据此定的） | 0.00356 | −0.07 | 0.93 |
| **从零训练**的策略实测 | **0.339（95×）** | **−6.8 ~ −8.6** | **~1e-3** |

训练没有发散、没有 NaN（547 iter 干净），mean reward 甚至还在涨——因为衰减近似恒定。
但这正是需求文档诊断表里的 **"一致性奖励是常数负底噪、不下降"**。
文档给的两个可能原因（未按速度分桶 / 参考模型不可实现）**都不适用**，
真实原因是第三个：把 R8 的**末端权重**当成了初始权重。

**两条结论：**

1. **R4.2/4.3 的默认 scale 改为 0**，标定出的 −20.0 / −0.5 作为
   R8 阶段 3 的**末端目标**记在配置注释里。这不是保守，是实测。
2. **R3 的收敛门保护不了这件事**，这一点值得单独记住：
   它在每个相位桶约 45 次访问后就打开（几秒行走），
   那时策略还什么都不会。**它守的是"估计量未收敛"，不是"策略未训练"。**

为此把两个惩罚项的**原始量**加进了性能指标
（`perf_phase_variance_raw` / `perf_steady_gain_raw` / `perf_soft_gate_open_fraction`），
**与它们的 reward scale 无关、恒定可见**——R8 的 ramp 时机就要靠盯这两条曲线来定。

### 出厂配置的 1000 iter 验收

`--seed 42`、4096 env、1281 s：**1000 iter 跑通，0 NaN、0 报错、无发散。**

| 指标 | it1 | it200 | it600 | it1000 |
|---|---|---|---|---|
| `rew_ref_tracking` | 0.03 | 11.7 | 11.1 | 11.7 |
| `rew_tracking_lin_vel`（对照） | 0.06 | 15.2 | 12.6 | 13.5 |
| `rew_orientation_control`（R4.4 降权后） | −0.02 | −2.9 | −1.9 | −1.6 |
| `perf_phase_variance_raw` | 0.000 | 0.197 | 0.264 | 0.307 |
| `perf_steady_gain_raw` | 0.029 | 0.525 | 0.765 | 0.876 |
| `perf_soft_gate_open_fraction` | 0.43 | 0.93 | 0.87 | 0.85 |

三点读法：

- `ref_tracking` 与原版 `tracking_lin_vel` 量级相当（11.7 vs 13.5），
  scale=2.0 的定值合适，核心项没有淹没也没有被淹没。
- 两个惩罚项的原始量**随训练上升**（0.20→0.31、0.52→0.88）——策略变得更快更动态，
  振荡偏差自然变大。这再次说明它们不能早开，且 R8 的 ramp 应当在这两条曲线
  **转平之后**才起步。
- 软门开启率 0.85（关闭 15%），高于对已训练策略实测的 4.4%：
  从零训练的策略打滑更多。会随训练自行下降，不需要改阈值。

---

## 7. R5 — 跨域一致性（Nominal Twin）

**这是改动面最大、风险最高的一条。**

### 分组

`group_size = 4`，env 索引 `[g*4, g*4+1, g*4+2, g*4+3]`，
**`g*4+0` 为 nominal twin**。4096 env → 1024 组。

### twin 的标称化：需要动的采样点

`self.is_nominal_twin` 布尔掩码在 `_init_buffers` 建立，
下列每个采样处都要跳过 twin：

| 采样项                                          | 位置                                                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| friction / restitution                          | `_process_rigid_shape_props` / `_randomize_rigid_body_props`                                |
| base mass / com                                 | `_process_rigid_body_props`                                                                   |
| motor strength / offset / Kp / Kd（dog）        | `_randomize_dof_props`（[legged_robot.py:431](../go1_gym/envs/roboduet/legged_robot.py#L431)） |
| arm Kp/Kd/strength/offset/link mass/com/payload | `_randomize_arm_dof_props`（`wbc_env.py`）                                                  |
| arm mount TF                                    | **直接指派 bucket 0**（已保证标称，见 AGENTS.md）                                         |
| gravity                                         | `_push_robots` 相邻的 gravity 随机化                                                          |
| arm 扰动                                        | stage-1 arm 曲线强度置 0（arm 固定）                                                            |

⚠️ 部分随机化是 **per-env 且在 `_create_envs()` 期间一次性确定**
（arm link mass/com、mount TF、base mass/com）。这意味着 twin 掩码
**必须在建环境之前就确定**，不能在训练中重新分组。可接受。

### 命令共享与相位同步

新增 **组级时钟** `group_step_counter (num_groups,)`，
组内重采样由它驱动（`% 500 == 0`），**不再用 per-env `episode_length_buf`**。
组重采样时：

1. 为整组采样一份命令，广播到 4 个 env。
2. `gait_indices[group] ← gait_indices[twin]`（相位同步，R5 不变量）。
3. 清 `group_desync` 标记。

### 终止不对称处理（R5 不变量）

- 任一 env 在组重采样之间 reset → 置 `group_desync[g] = True`，
  该组的 R5 项失效直到下次组重采样。
- **twin 自己摔倒 → 整组失效**（对齐目标消失）。
- `reset_idx` 里对分组 env 的 `_resample_commands` 改为
  **复制该组当前命令**，而不是采新的——否则组内命令立刻错开。
  ⚠️ 这是对 `_resample_commands` 语义的实质修改，
  需要一个 `copy_from_group` 分支，风险点已标注。

### 奖励

`_reward_domain_consistency`（负项）：

```python
err = y_detrended[e] - y_detrended[twin_of(e)].detach()
return (err**2 * w).sum(-1) * group_valid_mask * (~is_twin)
```

twin 一侧不回传梯度（`.detach()`，且 twin 自身该项为 0）。
比较量全部是 body frame 速度/角速度 + 相对地形高度（不变量）。

### 验收落地

断言同组 4 个 env 的 `commands_dog` 逐维相同、`gait_indices` 相同；
人为触发一个 env 摔倒，断言 `group_valid_mask[g]` 变 0 并在下次组重采样恢复；
断言 twin 的 friction/mass/motor_strength 等于标称值。

---

## 8. R6 — 富激励命令信号

### env 划分

```
[0                     , n_consistency)   一致性组（R5，走课程）
[n_consistency         , num_envs      )   辨识组（R6，约 25%，同样按 4 分组 + twin）
```

辨识组**也分组、也有 twin**——这样 R5 的跨域一致性度量在富激励信号下同样可用，
且 R6 的 chirp 数据天然带 nominal 对照，直接服务 R9 的闭环 Bode 图。
（此为我的设计决定，需求文档未规定二者如何叠加。）

### 信号

`ExcitationSampler` 每步为辨识环境写 `commands_dog`：

| 信号  | 参数                                                | 分配          |
| ----- | --------------------------------------------------- | ------------- |
| PRBS  | 切换间隔 U(0.5, 3.0) s，幅值从各通道 limit 采       | ~50% 辨识 env |
| Chirp | 0.1 → 2.0 Hz 线性扫频，周期 20 s（= 一个 episode） | ~30%          |
| Ramp  | 斜率覆盖 0.2×–3× 的`ṙmax`                     | ~20%          |

每 env 每 episode 随机指定**一个通道**做激励，其余通道保持常值——
单通道激励让后续 SISO 辨识和 Bode 图干净。姿态通道（height、pitch）
分配更高权重（文档：原版中它们几乎没有瞬态激励）。

### 排除出课程（R6 不变量）

`_resample_commands` 中 `curriculum.update(old_bins, task_rewards, ...)`
的输入（[L1288-1298](../go1_gym/envs/roboduet/legged_robot.py#L1288-L1298)）
必须先按 `~is_identification_env` 过滤 `env_ids`。
`_update_reset_curriculum(tracking_task_rewards, ...)`（[L1300](../go1_gym/envs/roboduet/legged_robot.py#L1300)）同样过滤。

课程判据的第二条不变量（不含一致性奖励）**自动满足**：
判据 key 写死为 4 个跟踪项（[facts §3](rlmpc-v3-r0-facts.md)），
只要新奖励不重名即可。加一条断言防止将来重名。

### 验收落地

辨识 env 单 episode 内命令跳变次数 ≥ 20；
chirp env 的命令序列 FFT 覆盖 0.1–2.0 Hz；
对比开启前后课程 `command_curriculum_weight` 的推进速度。

---

## 9. R7 — 观测与适应能力

### R7.1 新增观测（dog 策略）

在 `core.py::dog_obs_dim_parts`（[core.py:311-344](../go1_gym/envs/config/core.py#L311-L344)）加：

```python
parts["reference_state"] = 5        # xi
parts["reference_rate"] = 5         # xi_dot
parts["reference_minus_cmd"] = 5    # xi - u
parts["ee_pos_in_base"] = 3
parts["response_deviation"] = 4     # (g, l) x {pitch, yaw_rate} -- IMU 通道 only
```

`dog_num_observations: 90 → 112`，history **30 → 50**（见 R7.2），展平 5600。
原始命令 11 维保留不动（R7.1 明确要求）。

#### 每一项的物理意义与不可省略的理由

**ξ（参考状态，5）——「此刻本体应该处在哪个状态」**
策略跟踪的不再是命令 u 而是 ξ(t)。0.4 rad 的 pitch 阶跃下，t=0.1 s 时
ξ=0.036 rad，就是"这一刻机身应该倾斜多少"的物理答案。
不给的话策略必须从命令历史自行重建二阶滤波器状态：需要知道阶跃发生在多久以前、
跳变前的值是多少。**硬约束**：ξ 的建立时间 `4/ωₙ`，pitch 通道 ωₙ=5 → **0.8 s**，
而原历史窗口只有 0.6 s，**装不下一次完整阶跃响应，重建在信息上不可能**。

**ξ̇（参考速率，5）**
二阶系统的状态是 `(ξ, ξ̇)` 这一对。量纲随通道变：vx 通道是要求的**加速度**
（m/s²），pitch 通道是要求的**角速度**（rad/s）。
加了速率饱和后 `(ξ−u)` 不再唯一决定 ξ̇；命令在上次瞬态结束前又跳变时状态是
真正二维的。只给 ξ 则策略分不清"ξ 正在快速爬升"与"ξ 已稳住"——
这两种情形下腿该做的事相反（主动蹬伸 vs. 维持）。
**归一化按 `ṙmax`**，饱和边界正好映射到 ±1。

**ξ − u（5）——「还差多远到位」，同时是参考模型的驱动项**
即 `ẍ = ωₙ²(u−ξ) − 2ζωₙξ̇` 的第一项，物理上是**瞬态残余量**。
它让策略知道自己在哪个 regime：`|ξ−u|` 大 = 瞬态，R4.1 主导；
`≈0` = 稳态，R4.3 接管。**它正是这两个奖励项之间的切换变量。**
对 `(u, ξ)` 严格冗余；保留是为归一化——不同幅值命令在稳态时都趋于 0，
"瞬态阶段"信号与命令绝对大小解耦。

**EE 相对 base 的位置（3）——「机械臂这团质量此刻挂在哪」**
两个动力学后果：(a) **静力矩臂** `m_arm·g·r_xy` 是腿必须抵消的重力力矩，
直接偏置 pitch/roll，因而直接改变 height/pitch 通道的 **DC gain**——
正是 R4.3 要钉到 1 的那个量；(b) **反作用力旋量的力臂**。
对已在观测里的 `arm_dof_pos`(6) 严格冗余，但它是那 6 个角度的**正解**，
一条完整运动学链，也是 arm 质心投影的一阶近似。
与响应一致性的关联：设计文档要求同一命令在**不同 arm 构型**下给出相同 base
响应，而 arm 构型改变的正是被控对象（等效惯量、质心偏移），
EE 在 base 系的位置是这个改变的首要描述量。
`arm_dof_vel` 已在观测里，覆盖反作用力旋量的速度部分；
R4 明确不做显式 reaction wrench 建模，v1 到此为止。

**(g, l)（响应偏差统计量，4）——「我这个 domain 的响应偏了多少」**
去掉 teacher-student 后，域辨识本应完全隐式。但要辨识的量其实**可以直接算出来**，
没必要让网络去学。每通道两个慢速统计量：

```
g_c = EMA_slow( y_lp,c ) / EMA_slow( u_c )          # DC gain 偏差（分母需正则化）
l_c = EMA_slow( (y_c − ξ_c) · sign(ξ̇_c) )           # 有符号滞后指示
```

`l_c` 是**跟踪误差在参考运动方向上的投影**：策略系统性落后于参考时为正、
超前为负，正是它在当前 domain 下要补偿的那个标量。
**这就是经典控制里 PID 的 I 项**——"同一命令在不同 plant 下给出相同响应"
的经典答案就是积分作用，我们只是把它显式化。

**边际成本≈0**：R3 为相位去趋势本来就要维护 `y_lp`，ξ 也已经在，两个统计量都是现成料。

⚠️ **只对 IMU 锚定的通道启用：`pitch` 与 `yaw_rate`**（共 4 维）。
`vx / vy / height` 的 (g, l) **不进观测**——理由见下方「部署契约」。

⚠️ 它确实往闭环里加了状态，但与被否决的 RNN 隐状态是两回事：
该状态 (i) 在 env 里算、(ii) **被观测**因而对所有人可见、(iii) 动力学是
完全指定的 EMA、(iv) **EMA 时间常数取 ~5 s，远长于 MPC horizon（~1 s）**，
于是规划器看到的是一个**常数参数**而不是一个状态。

⚠️ episode 开头 EMA 未收敛，需与 R3 的样本计数门控一样加 warmup 标志位。

`arm_dof_pos` / `arm_dof_vel` **已在**（[facts §8](rlmpc-v3-r0-facts.md)），不重复加。

对应写入 `get_dog_observations`（[wbc_env.py:2623+](../go1_gym/envs/roboduet/wbc_env.py#L2623)）。
EE 相对 base 的位置：**已核实可直接复用**。`self.end_effector_state`
在 [wbc_env.py:1026-1031](../go1_gym/envs/roboduet/wbc_env.py#L1026-L1031)
无条件更新（在 `global_switch.switch_open` 门控**之前**），已包含到抓取点的
`ee_local_offset` 偏移。stage-1 下即可用，转 body frame 只需
`quat_rotate_inverse(base_quat, end_effector_state[:, :3] − base_pos)`，
**不需要额外 FK**。

### R7.2 适应通路 —— **不使用 teacher-student**（2026-08-31 用户决定）

需求文档 R7.2 要求"教师用特权信息编码低维隐变量、学生用本体感知历史回归它"。
**该条被明确否决**，本方案改用隐式适应 + 自监督辅助任务。

#### 保持现状的部分

`dog.use_adaptation_module` 保持 `False`（[config/wbc.py:160](../go1_gym/envs/config/wbc.py#L160)）。
架构仍是**非对称 actor-critic**：

```
actor : obs_history(5600)                    -> action      # 只换时序编码器实现
critic: cat(obs_history, privileged_obs(106)) -> V          # 不变
```

**因此 `dog_ac.py` 的架构、`ppo.py` 的 adaptation loss、`rollout_storage.py`
全部不动**——原方案里改动面最大、唯一触及 learner 数据通路的一块直接消失。

#### 代价与补偿：历史窗口 30 → 50 步

R7.2 的原始论证是"响应一致性 ≡ 在线系统辨识"。去掉显式通路后，
域辨识只能隐式发生在 actor 对 `obs_history` 的处理里。
**必须补偿的是窗口长度**：

| 量 | 值 |
|---|---|
| pitch 通道参考轨迹建立时间 `4/ωₙ`（ωₙ=5） | **0.8 s** |
| 原历史窗口 30 × 0.02 s | 0.6 s ❌ |
| 新历史窗口 **50** × 0.02 s | **1.0 s** ✅ |

0.6 s 的窗口既装不下一次完整阶跃响应（无法重建 ξ），
也不足以从中辨识 domain。**1.0 s 是同时覆盖两者的最小值**，
也正落在 R7.2 原文给出的 "0.5–1.0 s 本体感知历史" 区间上界。

改 `cfg.dog.dog_num_observation_history: 30 → 50`。开销：

| 项 | 原 | 新 |
|---|---|---|
| `dog_num_obs_history` 展平 | 30×90 = 2700 | 50×112 = **5600** |
| actor 首层参数 | 2700×512 = 1.38 M | 5600×512 = **2.87 M** |
| rollout storage（4096 env × 24 步 × fp32） | 1.06 GB | **2.20 GB** |

基线实测显存 9.1 GB / 32 GB（[facts §13](rlmpc-v3-r0-facts.md)），**余量充足**。
用户已明确显存不是约束，此处不做任何为省显存的妥协（稀疏历史方案已否决）。

#### 不设辅助头：改为**直接观测** (g, l)

早期草案曾提议一个辅助头，让 actor 去**预测**在线拟合的
(DC gain, 时间常数)。既然这个量本来就能在线算出来，
**就没有理由让网络去学它——直接当观测喂进去即可**（见 R7.1 的
`response_deviation`）。

由此带来的简化：

- 不需要 `aux_head` 模块 → `dog_ac.py` 只改时序编码器开关
- 不需要 `aux_target` 从 env 传到 learner → **`rollout_storage.py` 零改动**
- 不需要 aux loss → **`ppo.py` 零改动**

**learner 侧因此是字面意义上的零改动。**

### R7.3 时序编码架构 —— v1 用非线性历史，TCN 设计保留、延后实现

（2026-08-31 用户决定）

#### 需求推导：编码器到底要提取什么

加了 ξ / ξ̇ / ξ−u 之后，"我处在瞬态的哪一步"这个最难的时序积分任务已从编码器
身上卸掉。剩下三件事：

| Job | 需要什么 | 现状 |
|---|---|---|
| **1. 域辨识**（核心） | 长窗口、**低时间分辨率**、**低维输出**、靠**时间平均**而非时间选择 | 历史 + 新增的 (g, l) |
| 2. 接触/步态状态 | 短窗口 | `clock_inputs` + 开环相位已覆盖大半 |
| 3. 延迟补偿（`lag_timesteps=6` = 0.12 s） | 短窗口、全分辨率 | 历史 |

Job 1 的信号本质是**两条时间序列之间的回归**（发出的关节目标 → 机身实际加速度，
比值给出等效惯量），且在 episode 内**近似恒定**。

#### 方案比较

| 方案 | 与 Job 1 的匹配 | 判定 |
|---|---|---|
| **非线性历史**（flat MLP over T×C） | 弱。首层不知道输入是时间序列，时移等变性要从数据学；有用统计量是 `Σ(a·Δv)/Σ(a²)` 这类**乘性/二阶**量，MLP 不擅长；跨时间无权重共享 | **v1 采用** |
| **TCN**（dilated causal conv） | 强。局部感受野把该配对的量放进同一个核，乘性交互就地形成；跨时间权重共享；层次化聚合。有限感受野保住"超过 1 s 影响精确为零"的可预测性保证 | **设计保留，延后实现** |
| **Temporal attention** | **方向性错误**。attention 的强项是 content-based 长程选择；我们要的是对所有时间步求平均，而均匀池化本来就免费。加上 PPO 下 transformer 难训、`jit.script` 手写 attention 比 conv 脆弱 | **不采用** |

选 v1 用 flat MLP 的理由是**排序而非优劣**：TCN 不改任何观测语义、奖励或配置布局，
随时可换；正因为随时可换，就不该在 R5（本方案的单点风险）之前引入，
否则曲线不对时分不清是一致性奖励的问题还是编码器的问题。

#### 现在必须做的事：让 TCN 日后是真正的 drop-in

四条结构性约束，全部零成本，现在就要满足：

1. **历史缓冲保持按时间步连续**（oldest→newest，每个时间步一个 C 宽的块）。
   现状已满足：[wbc_env_wrapper.py:587-589](../go1_gym/envs/roboduet/wbc_env_wrapper.py#L587-L589)
   的 `cat((history[:, C:], obs), -1)` 就是这个布局。
   **加一条单元测试断言它**，防止将来有人"优化"成交错布局。
2. **时序编码器藏在单一开关后面**：`DogAC_Args.temporal_encoder = "flat" | "tcn"`，
   切换只动 `dog_ac.py` 一处，不碰 `ppo.py` / `Runner` / 导出脚本。
3. **reshape 归编码器自己管**。编码器 `forward` 的对外签名恒为
   `(B, T*C) -> (B, hidden)`，`view(B, T, C)` 在内部完成。
   这样 `torch.jit.script(actor_body)` + 单张量输入的部署契约
   （[export_rl_sar.py:289-295](../scripts/export_rl_sar.py#L289-L295)）对两种编码器都成立。
4. **T 和 C 从 cfg 推导**，不在网络里硬编码。

#### TCN 规格（延后实现时照此落地）

```
输入   (B, T=50, C=112) -- 由 (B, 5600) reshape 而来
Conv1d(C=112 -> 128, k=3, dilation=1)  + ELU
Conv1d(128  -> 128, k=3, dilation=2)   + ELU
Conv1d(128  -> 128, k=3, dilation=4)   + ELU
Conv1d(128  -> 128, k=3, dilation=8)   + ELU
Conv1d(128  -> 128, k=3, dilation=16)  + ELU     # 感受野 1+2*(1+2+4+8+16) = 63 >= 50
取最后一个时间步 -> (B, 128) -> 接原有 actor_hidden_dims
```

参数量约 **0.24 M**，对比 flat 首层 `5600×512 = 2.87 M`，差一个数量级。
**注意：TCN 省的是参数不是显存**——rollout storage 的
`observation_histories (24, 4096, 5600)` 与架构无关。

---

### R7.4 旧 checkpoint 必须显式报错

`Runner._load_matching_state_dict`（`ppo_cse_automatic/__init__.py`）
当前对形状不匹配的非 critic 权重抛错、对 critic 跳过。
观测宽度 90 → 112、历史 30 → 50 之后 **actor 首层必然不匹配** → 已经会抛错 ✅。
再加一条显式检查：checkpoint 的 `parameters.pkl` 若存在且
`dog.dog_num_observations` / `dog.dog_num_observation_history` 与当前不符，
打印明确的「布局已变更，需从零重训」提示，而不是让人去读一条形状不匹配的堆栈。

---

## 9b. 部署契约（新增的硬不变量）

这一节回答"扩 obs 会不会把部署和下游 MPC 搞难"。

#### 关键区分：MPC 接口不受影响

MPC 与策略之间的契约是 **5 个数进去、base SE(3) 出来**。观测是策略内部的事，
**扩 obs 完全不改变这个接口**，不会让 MPC 本身更难调参。

真正会咬人的失效模式是另一个，而且隐蔽得多：

> 机上某项观测与仿真不一致 → 策略行为漂移 → **辨识出来的响应模型不再描述真实闭环**
> → MPC 是在一个已经失效的 nominal dynamics 上做优化。

它不表现为"参数难调"，而是"预测和实际总差一点，但哪儿都查不出问题"。
**弥散、无症状，是最难查的一类。**

#### 新增不变量（列入全局清单）

> **13. 任何新增观测项必须能在部署侧无歧义复现，且其机上误差源必须进域随机化。**

#### 按此判据逐项过审

| 观测项 | 机上怎么产生 | 判定 |
|---|---|---|
| ξ, ξ̇ (10) | 纯命令驱动的 ODE 积分，**不碰传感器**；且 MPC 本来就在算它（R2：参考轨迹**就是** MPC 的 nominal dynamics） | ✅ 最低风险，**且是调试资产**：机上 ξ 与 MPC 预测的 ξ 对不上即暴露参数/时钟/命令通路偏差 |
| ξ − u (5) | 两个已有量相减 | ✅ 零风险 |
| ee_pos_in_base (3) | 臂编码器 + FK | ✅ 低风险（需 URDF / `ee_local_offset` 版本对齐） |
| (g, l) @ pitch, yaw_rate (4) | IMU 直接测，**有绝对锚点**（重力方向无漂移，coding 文档原文） | ✅ 通过 |
| ~~(g, l) @ vx, vy, height~~ | 需状态估计（腿式里程计+IMU 融合），有偏、会漂、打滑时退化；**慢速 EMA 恰好放大低频偏置** | ❌ **不进观测** |

被砍掉的部分并不损失信息：**(g, l) 本来就在驱动 R4.3 的稳态增益锁定**，
只是对机上不可靠的通道停止把它反馈回输入端。

顺带一提：pitch 恰好是论文最看重的通道（R8.2 姿态 vs 速度的本质差异、
R9 姿态通道专属指标、Bode 主图）。**风险最低处正好是收益最高处。**

#### 一条贯穿全案的分界

> **奖励可以随便用真值，观测必须自己养活自己。**

奖励只在仿真里算、永不部署，所以 R4 全套（R4.1/4.2/4.3）无论用多少
measured y，部署成本都是零。这条分界让"一致性机制尽量放奖励侧、
观测侧保持吝啬"成为默认策略。

#### 两个已经存在、与本方案无关的隐患（必须记录）

1. **`roboduet/base_lin_vel` 已经是部署项**
   （[export_rl_sar.py:242](../scripts/export_rl_sar.py#L242)，`dog.observe_lin_vel = True`）。
   现在的策略**已经**依赖机上速度估计——这个依赖不是本方案引入的。
2. **`dog.add_obs_noise = False`**（[config/wbc.py:161](../go1_gym/envs/config/wbc.py#L161)）
   ——dog 策略训练时**完全不加观测噪声**，等于在"速度估计完美"的假设下训练。

第 2 条是这条链上真正的薄弱环节。**排进第一次真机之前的清单**（不是现在，
现在 R1–R9 全在仿真）：打开 `dog.add_obs_noise`，并为 `base_lin_vel`
加**慢漂移偏置**随机化（不只是白噪声——EMA 类统计量对白噪声不敏感，对偏置敏感）。

#### `export_rl_sar.py` 需要新增的 term

[export_rl_sar.py:225-252](../scripts/export_rl_sar.py#L225-L252) 的具名注册表
与 `expected_obs_width` 宽度表各加 5 项：

```
roboduet/reference_state        5
roboduet/reference_rate         5
roboduet/reference_minus_cmd    5
roboduet/ee_pos_in_base         3
roboduet/response_deviation     4
```

前三项在 rl_sar 侧就是同一个二阶积分器的输出，**一次实现三处复用**。

---

## 10. R8 — 课程与参数标定

### R8.1 四阶段

新增 `ResponseCurriculum`，读 `global_switch.count`，产出每项的权重乘子。
挂在 `Runner.learn()` 的每 iteration 开头（与 `global_switch.count += 1` 同处）。

| 阶段 | iteration（建议） | 启用                                                            | 退出判据                  |
| ---- | ----------------- | --------------------------------------------------------------- | ------------------------- |
| 1    | 0 – 3000         | 原版奖励；R3 估计量更新但不入奖励；弱随机化                     | `tracking_lin_vel` 收敛 |
| 2    | 3000 – 6000      | R4.1 权重 0 → 满，1000 iter ramp                               | R4.1 奖励 > 0.8           |
| 3    | 6000 – 12000     | R4.2 + R4.3 + R5 慢升（1000 iter ramp）；随机化拉满；arm 扰动开 | 一致性指标达标            |
| 4    | 12000 –          | `push_robots = True`、地形难度（v1 无）；权重冻结             | 鲁棒性 ≥ 阶段 1 的 90%   |

**阶段 3 结束与阶段 4 结束各存一个 checkpoint**（帕累托前沿的两个点）。
阶段边界与 ramp 长度全部进 `cfg.response.curriculum`。

⚠️ v1 无地形，阶段 4 的"地形难度"退化为只加推力扰动
（`push_robots` 当前是关的，[facts §10](rlmpc-v3-r0-facts.md)）。

### R8.2 参考模型标定

`scripts/calibrate_reference_model.py`：

1. 载入阶段 1 的纯鲁棒 checkpoint。
2. 全随机化范围下，对每个通道下阶跃命令（覆盖 limit 范围），记录实测
   加速度峰值 / 速率峰值。
3. **速度通道**：按 domain 分桶取 **20 分位数**。
4. **姿态通道**：按 `(domain, 阶跃发生时刻的相位桶)` **联合分桶**取 20 分位数
   （R8.2 强制项 2）。
5. 反推 `ωₙ`（由二阶阶跃的峰值加速度 `Aωₙ²/…` 关系）与 `ṙmax`（速率峰值）。
6. 输出 `configs/reference_model_<tag>.yaml`。

**诊断信号**：训练奖励曲线出现与步频同频的周期性波动 → 姿态通道标定
未按相位分桶，立即回头重标。把这条做成一个 wandb 检查：
对 `rew_ref_tracking` 序列做 FFT，若在 `gait_frequency` 附近有显著峰则告警。

### R8.3 域随机化分层

三档写成 `configs/domain_{in_dist,held_out,ood}.yaml`，评测直接读。
v1 的分层维度：friction、base mass、ee payload、motor strength、arm 构型。

---

## 11. R9 — 评测与数据导出

### R9.1 扩展 `benchmark/dog_policy/`

已有框架（域随机化扫描 + 指标聚合）可直接复用。新增 scenario：

- `step_family`：固定阶跃 × 三档 domain × N=20，产出响应族图
- `chirp_bode`：闭环 Bode 图（幅频/相频），跨 domain 画成带状
- `model_order`：一阶 / 二阶 / 二阶+残差 的 R² 与归一化残差能量曲线

**速度通道与姿态通道分开报告**（R9.1 强制）。
主图 = 「模型阶数 R²」与「鲁棒性指标」双轴同图。

### R9.2 数据导出

`scripts/export_identification_data.py`，输出 HDF5 或 npz，
字段：命令 u、参考状态 ξ、实测 base SE(3) 与 twist、`gait_indices`、
gait frequency、arm 状态、domain 特权参数、地形标签（v1 恒为 plane）。
附带元数据 JSON：采样率 50 Hz、body frame 约定、通道索引表。

---

## 12. 风险与待核实项

| #       | 风险/歧义                                                                                  | 影响                                                           | 处理                                                                                                                                   |
| ------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| R1      | **R5 改写 `_resample_commands` 的语义**（组级时钟 + reset 时复制组命令而非新采样） | 高——这是命令系统的核心路径，改错会静默污染课程和所有跟踪奖励 | 分两步提交：先加组级时钟但不改命令内容（可对拍原版），再切共享命令                                                                     |
| R2      | **twin 标称化要动 6+ 处随机化采样点**，其中 3 处在 `_create_envs()` 里一次性执行   | 中——遗漏一处就等于 twin 不标称，R5 静默退化                  | 写一个断言测试：枚举所有`rand_buffers`，断言 twin 行等于标称值                                                                       |
| R3      | **R7.2 改为隐式适应 + 显式 (g,l) 统计量**，且 (g,l) 只覆盖 pitch/yaw_rate | 中——vx/vy/height 的域辨识完全靠 1.0 s 历史，上限可能低于显式通路 | 已接受的取舍；R9 的跨域离散度指标会分通道暴露代价，速度通道不达标时优先换 TCN 而非重开 teacher-student |
| R13     | **R4.2/4.3 的权重对策略成熟度极其敏感**：同一项在未训练策略上比训练后大 95× | 高——按训练后标定的权重在起步阶段会把总奖励压掉 1000×，表现为"常数负底噪"而非崩溃，很容易被误读成别的问题 | 默认 scale=0，由 R8 阶段 3 慢升；`perf_phase_variance_raw` 恒定可见，用它定 ramp 时机。**R3 的收敛门不提供保护**（它守估计量收敛，不守策略成熟度） |
| R11     | **pitch/height 现在是课程维度，但还不是「会扩张的课程」** | 中——R1 的字面要求（排除非决策通道）已满足，但两处继承来的设定让新增的 bin 暂时不产生渐进解锁：① `set_to` 的初始窗口对 pitch 用 `body_pitch_range`、对 height 用 `limit_body_height`，二者当前都等于全区间 → 所有 bin 一开始就激活；② `local_range` 对 pitch/height 是 **1.0**，跨越整个网格 | **刻意留给 R8**：初始窗口与扩张邻域是课程排期问题，在第 2 步改会引入无法在本步验证的训练动力学变化。建议 R8 里定 pitch local_range ≈ 0.2、height ≈ 0.12，并让 height 的初始窗口改用 `body_height_cmd` 以消除与 pitch 的不一致 |
| R12     | **`gait_frequency` 在 `‖v_cmd‖ < 0.1` 时被强制为 0**（[legged_robot.py:1354-1356](../go1_gym/envs/roboduet/legged_robot.py#L1354-L1356)，既有行为，R1 保留） | 中——实测约 17–24% 的样本处于该状态。残差模型会在站立段看到频率从 ~3 Hz 跳到 0，相位随之停止推进 | R9.2 导出必须带「站立标志」；R3 的相位分桶要把站立段单独处理或排除，否则 `δ̂(φ)` 会被污染 |
| R10     | **`dog.add_obs_noise=False`**，策略在“速度估计完美”假设下训练（既有问题，非本方案引入） | 高（仅对真机）——任何依赖 measured y 的观测都不可靠，包括**现在就在用的** `base_lin_vel` | 排进第一次真机前的清单：打开噪声 + 为 `base_lin_vel` 加**慢漂移偏置**随机化（白噪声不够，EMA 类统计量对偏置才敏感） |
| ~~R4~~ | ~~EE 相对 base 位置在 stage-1 是否已计算~~                                                | —                                                             | **已核实解除**：`end_effector_state` 在 switch 门控前无条件更新（[wbc_env.py:1026](../go1_gym/envs/roboduet/wbc_env.py#L1026)） |
| R5      | `sigma_rew_neg = 0.02` 与 `dt = 0.02` 抵消是巧合                                       | 中——任何人改其一都会静默缩放全部一致性权重                   | 配置里加显式注释 + 一条断言                                                                                                            |
| R6      | 辨识 env 占 25%，同时它们也参与 R5 分组                                                    | 低                                                             | 已在 §8 说明设计选择                                                                                                                  |
| R7      | 阶段 3 最易崩（文档明示）                                                                  | 高                                                             | ramp ≥ 1000 iter；每 500 iter 存 checkpoint 便于回退                                                                                  |
| R8      | v1 无地形 → 不变量 9 空转、R9 指标 5 缺"崎岖地形通过率"                                   | 已知并接受                                                     | 预防性修`_reward_jump` 的硬编码 0，为后续地形留口                                                                                    |
| R9      | `commands_dog` 索引在 rewards.py 里多处硬编码（3:5、5、6、7、8、9）                      | 中——R1 冻结后这些仍会读到冻结值，行为正确但脆弱              | 引入`CHANNEL_CMD_IDX` 常量表，逐步替换硬编码                                                                                         |

---

## 13. 建议的实施顺序

每一步都可独立验证、可回退：

| #  | 内容                                                                 | 验收                                   |
| -- | -------------------------------------------------------------------- | -------------------------------------- |
| 0  | 基线复现（**已完成**，见 [facts §13](rlmpc-v3-r0-facts.md)）   | 200 iter 跑通                          |
| 1 ✅ | 新建 `go1_gym/response/` + `ReferenceModel`，不接 env（**已完成**） | R2 三条数值验收 + 4 项变异测试，22 项通过 |
| 2 ✅ | R1 命令裁剪 + 命令索引表 + 课程构造抽取（**已完成**） | 17 项通过；网格 1.96M→33,075；真环境 rollout 冻结通道带内恒定；obs 宽度仍 90 |
| 3 ✅ | 接入 `ReferenceModel`，删除一阶版本，跟踪误差进性能指标（**已完成**） | 50 项单测通过；真环境 5 通道位置增量 = ṙmax·dt、峰值速率 = 限幅、稳态增益 1.000；200 iter 无 NaN |
| 4 ✅ | R3 残差估计 + 双路去趋势（**已完成**） | 64 项单测 + 3 项变异；真环境 δ̂ 谐波分布物理正确（vx 偶次主导、pitch h1=98%）；相位查表零延迟 vs 低通滞后 330× |
| 5 ✅ | R4 四项 + σ 标定（**已完成**） | 83 项单测；σ 复现文档算例（t=0.1 参考 0.0361 / 激进误差 0.2015 / 奖励 0.0065）；1000 iter 无 NaN；R4.2/4.3 实测必须默认关闭 |
| 6  | R6 富激励（先不分组）                                                | 跳变次数 ≥ 20；课程速度不退化         |
| 7  | **R5 分组 + twin**（最高风险，分两步）                         | 组内命令/相位一致；失效逻辑            |
| 8  | R7 观测（90→112）+ 历史窗口 30→50 + R7.3 的 4 条结构性约束            | obs 宽度断言通过；旧 ckpt 显式报错；历史布局连续性测试通过；`temporal_encoder` 开关就位 |
| 9  | R8 四阶段课程 + 参考模型标定                                         | 四个 checkpoint；无步频同频波动        |
| 10 | R9 评测与导出                                                        | 主图产出                               |

第 1–5 步是方法的骨架，第 7 步是最大的单点风险。
建议在第 5 步结束时先跑一次完整训练看曲线，再决定是否继续第 7 步。
