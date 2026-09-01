# R0 事实清单 — Response-Consistent Locomotion Policy

对应 `docs/project-design-rlmpc-v3-coding.md` R0 的第一份产出。
逐条给出文件路径与行号作为证据。所有数值均为 **实际组合后的运行时配置**
（`build_roboduet_config(robot=go2_x5, dyna_gait=True)`），不是某个 profile 的源码默认值。

勘察日期：2026-08-31，分支 `refactor`，HEAD `9e7e995`。

---

## 0. 时间基准

| 量 | 值 | 证据 |
|---|---|---|
| 仿真步长 | 0.005 s | [config/legged_robot.py:410](../go1_gym/envs/config/legged_robot.py#L410) |
| decimation | 4 | [config/legged_robot.py:221](../go1_gym/envs/config/legged_robot.py#L221) |
| **策略步长 `dt`** | **0.02 s（50 Hz）** | 上两项之积 |
| episode 长度 | 20 s = 1000 步 | [config/legged_robot.py:21](../go1_gym/envs/config/legged_robot.py#L21)，[legged_robot.py:2624-2626](../go1_gym/envs/roboduet/legged_robot.py#L2624-L2626) |
| 命令重采样周期 | 10 s = 500 步 | `commands.resampling_time`，[legged_robot.py:1505-1507](../go1_gym/envs/roboduet/legged_robot.py#L1505-L1507) |

> **对 R6 的直接含义**：一个 episode 内命令只在第 0 步和第 500 步跳变两次，
> 其中第 0 步紧跟 reset（状态本来就是重置的）。**每个 episode 只有 1 次真正的
> 命令阶跃瞬态**。R6 关于"瞬态样本严重不足"的判断在本代码库里被证实，
> 而且比文档估计的更严重。

---

## 1. 命令向量的完整定义

`commands_dog` 缓冲区在 [legged_robot.py:1894-1896](../go1_gym/envs/roboduet/legged_robot.py#L1894-L1896)
分配，宽度 `cfg.dog.dog_num_commands`。

- 不带 `--dyna_gait`：宽度 **6**（[config/wbc.py:159](../go1_gym/envs/config/wbc.py#L159)）
- 带 `--dyna_gait`：宽度 **11**（`enable_dyna_gait` 加 5 维，[config/core.py:521-533](../go1_gym/envs/config/core.py#L521-L533)，
  `FEATURE_LAYOUT["dynamic_gait_command_dims"]=5`，[config/wbc.py:598-602](../go1_gym/envs/config/wbc.py#L598-L602)）

### 11 维布局（`--dyna_gait` 下）

| idx | 含义 | 单位 | 采样范围 | limit 范围 | 课程 bin 数 | obs 缩放 |
|---|---|---|---|---|---|---|
| 0 | 前向速度 vx | m/s | [-0.5, 0.5] | [-1.5, 1.5] | 21 | 2.0 |
| 1 | 侧向速度 vy | m/s | [-0.3, 0.3] | [-1.0, 1.0] | 3 | 2.0 |
| 2 | 偏航角速度 ωyaw | rad/s | [-1.0, 1.0] | [-2.0, 2.0] | 21 | 0.25 |
| 3 | body pitch | rad | [-0.4, 0.4] | [-0.4, 0.4] | **1** | 1.0 |
| 4 | body roll | rad | [-0.4, 0.4] | [-0.4, 0.4] | **1** | 1.0 |
| 5 | body height（相对 `base_height_target=0.30` 的增量） | m | [-0.2, 0.3] | [-0.2, 0.3] | **1** | 1.0 |
| 6 | gait frequency | Hz | **[0.0, 6.0]** | [0.0, 6.0] | 11 | 1.0 |
| 7 | footswing height | m | [0.06, 0.061] | 同 | 5 | 0.15 |
| 8 | stance width | m | [0.10, 0.45] | 同 | 3 | 1.0 |
| 9 | stance length | m | [0.25, 0.45] | 同 | 3 | 1.0 |
| 10 | gait duration | – | [0.49, 0.5] | 同 | 3 | 1.0 |

索引顺序的权威来源：`commands_scale_dog` 的构造，
[legged_robot.py:1897-1913](../go1_gym/envs/roboduet/legged_robot.py#L1897-L1913)；
消费端 [legged_robot.py:2715-2717](../go1_gym/envs/roboduet/legged_robot.py#L2715-L2717)（gait 6/10）、
[rewards.py:298](../go1_gym/envs/rewards/rewards.py#L298)（footswing 7）、
[rewards.py:337-338](../go1_gym/envs/rewards/rewards.py#L337-L338)（stance 8/9）、
[rewards.py:316](../go1_gym/envs/rewards/rewards.py#L316)（roll/pitch 3:5）、
[rewards.py:237](../go1_gym/envs/rewards/rewards.py#L237)（height 5）。

`obs_scales` 全表见 `cfg.obs_scales`；上表只列命令相关项。

### 与 R1 的对照

R1 的 5 个决策通道 → idx **0, 1, 2, 5, 3**；半自由通道 → idx **6**；
冻结通道 → idx **4, 7, 8, 9, 10**。**布局天然对齐，不需要重排索引**，
R1「保留冻结通道位置」的不变量零成本满足。

需要注意两处偏差：

1. **gait frequency 现在从 0 Hz 起采样**。`enable_dyna_gait` 把下界设为
   `--dyna_gait_min_frequency`（默认 0.0），[core.py:525](../go1_gym/envs/config/core.py#L525)。
   0 Hz 意味着无步态。R1 要求 2.5–3.5 Hz。
2. **body pitch / roll / height 的课程 bin 数都是 1**，见上表。
   `Curriculum.__init__` 对 1 个 bin 只生成一个中心点、覆盖整个区间
   （[base/curriculum.py:31-42](../go1_gym/envs/base/curriculum.py#L31-L42)），
   `sample_uniform_from_cell` 在该区间内均匀采样
   （[base/curriculum.py:81-84](../go1_gym/envs/base/curriculum.py#L81-L84)）。
   **结论：这三个通道目前完全不受自适应课程控制，是全程均匀采样。**

---

## 2. 命令采样与重采样机制

### 触发点（两处）

1. **周期触发**：`_post_physics_step_callback` 中
   `episode_length_buf % 500 == 0` 的环境，
   [legged_robot.py:1505-1507](../go1_gym/envs/roboduet/legged_robot.py#L1505-L1507)。
2. **reset 触发**：`reset_idx` 的第一件事就是 `_resample_commands(env_ids)`，
   [legged_robot.py:429](../go1_gym/envs/roboduet/legged_robot.py#L429)。

因为 `episode_length_buf[env_ids] = 0` 在 reset 尾部
（[legged_robot.py:441](../go1_gym/envs/roboduet/legged_robot.py#L441)），
两条路径不会在同一步重复触发。

### `_resample_commands` 主体

[legged_robot.py:1264-1358](../go1_gym/envs/roboduet/legged_robot.py#L1264-L1358)。顺序：

1. `_arm_resample_commands_train_hook(env_ids)`；stage-2 goal-reaching 下返回 True，
   此时 dog 命令由 arm policy 的 `plan()` 接管，本函数不再写 `commands_dog[:, :3]`
   （[wbc_env.py:1393-1397](../go1_gym/envs/roboduet/wbc_env.py#L1393-L1397)）。
2. 用 `command_sums` / `ep_len` 计算 4 个跟踪类奖励的 episode 均值，
   与 `curriculum_thresholds[key] * pretrained_reward_scales[key]` 比较，
   调用 `curriculum.update(...)`（[L1272-1298](../go1_gym/envs/roboduet/legged_robot.py#L1272-L1298)）。
3. `curriculum.sample(batch_size)` 得到新命令（[L1303](../go1_gym/envs/roboduet/legged_robot.py#L1303)）。
4. 写入：`[:, 0:3]` 速度（含 10% 置零 + 死区裁剪，[L1315-1325](../go1_gym/envs/roboduet/legged_robot.py#L1315-L1325)）、
   `[:, 3:6]` 姿态（仅 `not global_switch.switch_open` 时，[L1341-1344](../go1_gym/envs/roboduet/legged_robot.py#L1341-L1344)）、
   `[:, 6:11]` 步态（同样条件，[L1346-1353](../go1_gym/envs/roboduet/legged_robot.py#L1346-L1353)）。
5. 速度指令模长 < 0.1 时把 gait frequency 强制为 0（站立），[L1354-1356](../go1_gym/envs/roboduet/legged_robot.py#L1354-L1356)。
6. 清零 `command_sums[key][env_ids]`。

### 周期控制项

`resampling_time = 10.0`（[config/wtw.py:16](../go1_gym/envs/config/wtw.py#L16)）。
`subsample_gait` / `gait_interval_s` / `vel_interval_s` / `jump_interval_s`
（[config/legged_robot.py:119-123](../go1_gym/envs/config/legged_robot.py#L119-L123)）
在 RoboDuet 路径上**没有任何消费者**，是死配置。

---

## 3. 自适应命令课程

实现：`RewardThresholdCurriculum`，[base/curriculum.py](../go1_gym/envs/base/curriculum.py)。
只有一个 category `"trot"`（[legged_robot.py:1363](../go1_gym/envs/roboduet/legged_robot.py#L1363)）。

### 网格定义

`_init_command_distribution`，[legged_robot.py:1361-1490](../go1_gym/envs/roboduet/legged_robot.py#L1361-L1490)。
维度顺序：`x_vel, y_vel, yaw_vel, body_pitch, body_roll, body_height`，
`--dyna_gait` 下追加 `gait_frequency, footswing_height, stance_width, stance_length, gait_duration`。

**网格总大小 = 21×3×21×1×1×1×11×5×3×3×3 = 1,964,655 个 bin。**
以 4096 环境、每 500 步重采样计算，一次 iteration（24 步）平均只有约 200 个
环境重采样，覆盖整个网格需要约 1 万次 iteration。**网格相对样本量严重过大，
主要被 footswing(5)/stance(3×3)/duration(3) 这些 R1 要冻结的通道撑起来的。**

初始激活区间由 `set_to(low, high)` 给出，
[legged_robot.py:1441-1490](../go1_gym/envs/roboduet/legged_robot.py#L1441-L1490)：
速度用 `lin_vel_x/y`、`ang_vel_yaw`（窄），姿态用 `body_pitch_range`/`body_roll_range`/
`limit_body_height`，步态用各自的 `limit_*`。

### 进度判据

[legged_robot.py:1275-1298](../go1_gym/envs/roboduet/legged_robot.py#L1275-L1298)。
使用的 key 恰好是四个：

```
tracking_lin_vel, tracking_ang_vel,
tracking_contacts_shaped_force, tracking_contacts_shaped_vel
```

阈值 = `curriculum_thresholds[key] * pretrained_reward_scales[key]`，阈值系数分别为
0.8 / 0.7 / 0.90 / 0.90（[config/wtw.py:58-61](../go1_gym/envs/config/wtw.py#L58-L61)）。
`local_range` 固定 `[0.55,0.55,0.55,1.0,1.0,1.0]`（+ dyna_gait 时 5 个 0.55），
[L1291-1293](../go1_gym/envs/roboduet/legged_robot.py#L1291-L1293)。

> **对 R6 不变量的含义**：进度判据当前只读上述 4 个 key，天然不含一致性类奖励。
> 只要新奖励不叫这四个名字，**R6 的第二条不变量自动满足**。但辨识环境
> 目前无法被排除——`curriculum.update` 接收的是 `env_ids` 的全部，没有过滤机制。

---

## 4. 步态相位变量

| 项 | 事实 | 证据 |
|---|---|---|
| 表示 | `self.gait_indices`，shape `(num_envs,)`，∈ [0,1) 归一化全局相位 | [legged_robot.py:1979](../go1_gym/envs/roboduet/legged_robot.py#L1979) |
| 积分位置 | `_step_contact_targets()`，由 `_post_physics_step_callback` 在 `_resample_commands` **之后**调用 | [legged_robot.py:1508](../go1_gym/envs/roboduet/legged_robot.py#L1508) |
| 积分公式 | `gait_indices = (gait_indices + dt * frequencies) mod 1.0` | [legged_robot.py:2722](../go1_gym/envs/roboduet/legged_robot.py#L2722) |
| 频率来源 | `commands_dog[:, 6]`（dyna_gait）否则常数 **3.0** | [legged_robot.py:2715-2720](../go1_gym/envs/roboduet/legged_robot.py#L2715-L2720) |
| 步态类型 | 硬编码 `gaits["trotting"] = [0.5, 0, 0]` | [legged_robot.py:2712-2713](../go1_gym/envs/roboduet/legged_robot.py#L2712-L2713) |
| 派生量 | `foot_indices (4)`、`clock_inputs (4)` = `sin(2π·foot_idx)`、`desired_contact_states (4)` | [legged_robot.py:2739-2801](../go1_gym/envs/roboduet/legged_robot.py#L2739-L2801) |
| reset 行为 | `gait_indices[env_ids] = 0`，在 `reset_idx` 的最末尾 | [legged_robot.py:569](../go1_gym/envs/roboduet/legged_robot.py#L569) |
| 站立特判 | `‖commands_dog[:, :3]‖ < 0.1` 时 foot_indices 被钉在 0.25 | [legged_robot.py:2742](../go1_gym/envs/roboduet/legged_robot.py#L2742) |

> **R3 的核心前提成立**：相位完全由 gait frequency 命令开环积分得到，
> 不依赖接触估计，是精确已知的零延迟时钟。
> 注意站立特判只改 `foot_indices`，`gait_indices` 本身仍在积分。

---

## 5. 奖励组织方式与总奖励合成

### 注册机制

`_prepare_reward_function`，[legged_robot.py:1988-2038](../go1_gym/envs/roboduet/legged_robot.py#L1988-L2038)。
`_reward_<name>` 方法由 `cfg.reward_scales.<name>` / `cfg.wbc.reward_scales.<name>` 自动接线。
**scale 为 0 的项直接不注册**（[L2021-2023](../go1_gym/envs/roboduet/legged_robot.py#L2021-L2023)）。
**所有 scale 在注册时乘以 `dt`（0.02）**（[L2004](../go1_gym/envs/roboduet/legged_robot.py#L2004)、[L2007](../go1_gym/envs/roboduet/legged_robot.py#L2007)）。

两张表：`pretrained_reward_scales`（stage-1）与 `wbc_reward_scales`（stage-2），
由 `global_switch.get_reward_scales()` 按 iteration 做 sigmoid 插值
（[utils/global_switch.py:32-45](../go1_gym/utils/global_switch.py#L32-L45)）。

### 合成结构（关键）

`compute_reward`，[legged_robot.py:775-848](../go1_gym/envs/roboduet/legged_robot.py#L775-L848)。
正负项分别累加到 `rew_buf_pos_dog` / `rew_buf_neg_dog`，然后：

```python
# only_positive_rewards_ji22_style = True, sigma_rew_neg = 0.02
rew_buf_dog = rew_buf_pos_dog * exp(rew_buf_neg_dog / 0.02)
```

[legged_robot.py:825-828](../go1_gym/envs/roboduet/legged_robot.py#L825-L828)；
开关来自 [config/wtw.py:152-154](../go1_gym/envs/config/wtw.py#L152-L154)。

> **R4 不变量 10 天然满足**：总奖励已经是 `r_task · exp(k·Σr_aux)` 的乘性结构。
> R4.1 写成正奖励即进入 `r_pos`，R4.2/4.3/4.4 写成负奖励即进入指数因子。
>
> **但 σ_rew_neg = 0.02 极小**，且 scale 已乘过 dt=0.02。一个"名义权重 −1.0"的
> 惩罚项，单步实际贡献是 `−1.0 × 0.02 × raw / 0.02 = −raw`（指数内），
> 即 **名义权重 w 在指数里的有效系数恰好等于 w**（dt 与 σ 抵消）。
> 这个巧合让权重直觉还算可用，但必须在文档里写清，否则调参会失控。

### 当前活跃的 dog 侧奖励（stage-1）

来自 `config/legged_robot.py` reward_scales 默认 + `wtw.py` + `wbc.py` 覆盖：

| 名称 | scale | 类型 |
|---|---|---|
| `tracking_lin_vel` | 1.0 | 正，任务 |
| `tracking_ang_vel` | 0.5 | 正，任务 |
| `jump` | 10.0（wbc 5.0） | 负（返回 `−(h−h*)²`），**body height 跟踪** |
| `orientation_control` | −5.0 | 负，**姿态跟踪** |
| `tracking_contacts_shaped_force` | 4.0 | 负 |
| `tracking_contacts_shaped_vel` | 4.0 | 负 |
| `raibert_heuristic` | −10.0 | 负 |
| `feet_clearance_cmd_linear` | −30.0 | 负 |
| `response_consistency` | **−0.05** | 负，**已有的一阶版本** |
| `loco_energy` | −4e-5 | 负 |
| 其余 | `feet_slip −0.04`、`action_rate −0.001`、`action_smoothness_1/2 −0.1`、`dof_vel −1e-4`、`lin_vel_z −0.02`、`ang_vel_xy −0.001`、`collision −10.0` | 负 |

---

## 6. 姿态相关奖励的现有计算方式

### body pitch / roll — `_reward_orientation_control`

[rewards.py:313-326](../go1_gym/envs/rewards/rewards.py#L313-L326)。
由 `commands_dog[:, 3:5]` 构造期望四元数，转成期望 `projected_gravity`，
惩罚 `‖projected_gravity[:, :2] − desired[:, :2]‖²`。
**是瞬时误差，无任何去趋势或滤波。** 这正是 R4.4 要保留并降权到 10–20% 的项。

### body height — `_reward_jump`

[rewards.py:234-239](../go1_gym/envs/rewards/rewards.py#L234-L239)：

```python
reference_heights = 0                       # <-- 硬编码
body_height = self.env.base_pos[:, 2] - reference_heights
jump_height_target = self.env.commands_dog[:, 5] + cfg.rewards.base_height_target
return -torch.square(body_height - jump_height_target)
```

`base_height_target = 0.30`（[config/wtw.py:149](../go1_gym/envs/config/wtw.py#L149)）。

> **两点必须记录**：
> 1. 名字叫 `jump`，实际是 body height 跟踪项，权重 10.0（stage-1）。
> 2. `reference_heights = 0` 是硬编码，未扣除地形高度。同一文件的
>    `_reward_feet_contact_vel`（[rewards.py:283](../go1_gym/envs/rewards/rewards.py#L283)）
>    有同样问题。平地下无害；一旦启用 trimesh 地形，
>    **直接违反全局不变量 9**。

---

## 7. body height 是否已扣除地形高度

**分情况：**

- **奖励路径：否。** `_reward_jump` 硬编码 0（见上）。
- **性能指标路径：是（有条件）。** `_update_performance_metrics`
  [legged_robot.py:1095-1105](../go1_gym/envs/roboduet/legged_robot.py#L1095-L1105)
  用 `measured_heights` 求均值作为 `reference_height`，写法正确。
- **终止判据路径：是（有条件）。** `check_termination`
  [legged_robot.py:407-411](../go1_gym/envs/roboduet/legged_robot.py#L407-L411)
  用 `root_states[:,2] − measured_heights`。

但当前运行时 **`terrain.measure_heights = False`**
（[config/go1.py:45](../go1_gym/envs/config/go1.py#L45)），
因此 `self.measured_heights = 0`（int，[legged_robot.py:1798](../go1_gym/envs/roboduet/legged_robot.py#L1798)），
上面两条"正确路径"实际都退化成减 0。

且 `terrain.mesh_type = "plane"`（[config/wtw.py:124](../go1_gym/envs/config/wtw.py#L124)），
`_get_heights` 对 plane 直接返回全零（[legged_robot.py:2687-2688](../go1_gym/envs/roboduet/legged_robot.py#L2687-L2688)）。
**平地下三条路径等价，无害。**

---

## 8. 观测向量

三套观测，宽度由 `core.py` 的 `*_obs_dim_parts` 单点推导，
`recompute_observation_dims` 汇总（[core.py:419-424](../go1_gym/envs/config/core.py#L419-L424)）。
构造处有断言校验宽度（[roboduet/utils.py:158-162](../go1_gym/envs/roboduet/utils.py#L158-L162)）。

### dog 策略观测（本项目的主战场）

`get_dog_observations`，[wbc_env.py:2623+](../go1_gym/envs/roboduet/wbc_env.py#L2623)；
维度表 `dog_obs_dim_parts`，[core.py:311-344](../go1_gym/envs/config/core.py#L311-L344)。

| 组成 | 维度 |
|---|---|
| projected_gravity | 3 |
| dog dof pos / vel / actions | 12 × 3 = 36 |
| **dog_commands** | 11 |
| arm_commands | 6 |
| **arm_dof_pos / arm_dof_vel** | 6 + 6 = 12 |
| clock_inputs | 4 |
| base_ang_vel / base_lin_vel | 3 + 3 = 6 |
| tracking（height/pitch/roll 实测 + pose 误差 + vel 误差） | 9 |
| **合计 `dog_num_observations`** | **90** |

history 长度 30（[config/wbc.py:158](../go1_gym/envs/config/wbc.py#L158)）→ 展平 2700。
**30 × 0.02 s = 0.6 s，正落在 R7.2 要求的 0.5–1.0 s 区间。**

> **R7.1 现状**：`arm_dof_pos` / `arm_dof_vel` **已经在 dog 观测里**
> （[core.py:318-319](../go1_gym/envs/config/core.py#L318-L319)）。
> 缺的只有：参考状态 ξ、ξ−u、EE 相对 base 的位置。

其它两套：`env.num_observations = 74`（历史 30）、`arm_num_observations = 40`（历史 60）。

### dog 特权观测

`dog_num_privileged_obs = 106`，`privileged_obs_dim_parts(policy="dog")`
[core.py:346-411](../go1_gym/envs/config/core.py#L346-L411)。包含摩擦、恢复系数、
base 质量、电机强度/偏置、Kp/Kd、重力、base 速度、接触状态、
arm 动力学（Kp/Kd/strength/offset/link mass/link com）、arm mount TF、arm dof pos/vel。

> **R7.2 的教师特权信息在这里已经齐了**（摩擦、质量、负载、电机强度、arm 惯量相关项）。

---

## 9. adaptation / teacher-student 模块

**存在但当前关闭。**

- 实现：`DogActorCritic.adaptation_module`，
  [dog_ac.py:37-59](../go1_gym_learn/ppo_cse_automatic/dog_ac.py#L37-L59)。
  MLP `[256, 128]`，输入 `num_obs_history`(2700)，输出 `num_privileged_obs`(106)。
- 开关：`dog.use_adaptation_module = False`、`arm.use_adaptation_module = False`
  （[config/wbc.py:160](../go1_gym/envs/config/wbc.py#L160)、[:217](../go1_gym/envs/config/wbc.py#L217)）。
- 训练损失：`PPO.update` 中 `MSE(adaptation_module(obs_history), privileged_obs)`，
  [ppo.py:224-246](../go1_gym_learn/ppo_cse_automatic/ppo.py#L224-L246)。
- actor 输入：`use_adaptation_module` 为真时是 `cat(obs_history, latent)`，
  否则只有 `obs_history`（[dog_ac.py:62-65](../go1_gym_learn/ppo_cse_automatic/dog_ac.py#L62-L65)、
  [:126-131](../go1_gym_learn/ppo_cse_automatic/dog_ac.py#L126-L131)）。
- critic 始终用真实特权观测：`critic_body(cat(obs_history, privileged_obs))`，
  [dog_ac.py:161-163](../go1_gym_learn/ppo_cse_automatic/dog_ac.py#L161-L163)。

> **与 R7.2 的差距**：这是纯 RMA——学生回归 **完整 106 维特权向量**，
> 不是低维隐变量，也没有辅助监督头。R7.2 要求的
> 「教师编码为低维隐变量 + 学生回归 + 辅助头预测响应参数偏移」需要改
> `dog_ac.py` 架构和 `ppo.py` 的损失。

---

## 10. 域随机化全部项与范围

组合后运行时值（RoboDuet profile）。基类默认见
[config/legged_robot.py:247-280](../go1_gym/envs/config/legged_robot.py#L247-L280)，
WTW 覆盖见 [config/wtw.py:93-122](../go1_gym/envs/config/wtw.py#L93-L122)，
RoboDuet 覆盖见 [config/wbc.py:226-241](../go1_gym/envs/config/wbc.py#L226-L241)。

### 本体 / 地面

| 项 | 开 | 范围 |
|---|---|---|
| friction | ✅ | [0.1, 3.0] |
| restitution | ✅ | [0.0, 0.4] |
| base mass | ✅ | [-2.0, 2.0] kg |
| **com displacement** | ❌ | [-0.15, 0.15]（关） |
| gravity | ✅ | [-1.0, 1.0]，间隔 8 s，持续 0.99 s |
| ground friction | ✅ | [0.0, 0.0]（**范围为零，等于没开**） |
| motor strength | ✅ | [0.9, 1.1] |
| motor offset | ✅ | [-0.02, 0.02] |
| Kp / Kd factor | ✅ | [0.9, 1.1] |
| lag timesteps | ❌ | 6（RoboDuet 关掉了随机化，[wbc.py:228](../go1_gym/envs/config/wbc.py#L228)） |
| **push_robots** | ❌ | max_vel 1.0 / max_ang 0.6（**关**，[wtw.py:115](../go1_gym/envs/config/wtw.py#L115)） |
| rand_interval_s | – | 4.0 |
| randomize_rigids_after_start | ❌ | [wtw.py:95](../go1_gym/envs/config/wtw.py#L95) |
| tile_height_range | – | [0, 0]（关） |

### arm mount（per-env，创建时确定）

`randomize_mount_position` ✅ `[[-0.05,0.05],[-0.02,0.02],[-0.05,0.05]]` m；
`randomize_mount_rotation` ✅ ±3°/±3°/±5°；16 个 bucket，bucket 0 为标称。
[config/wbc.py:231-240](../go1_gym/envs/config/wbc.py#L231-L240)。

> **R5 的 nominal twin 可以直接复用 bucket 0**（AGENTS.md 明确说 bucket 0 保持标称 mount）。

### stage-1 arm（每 episode）

`Kp_factor [0.5,1.5]`、`Kd_factor [0.2,2.0]`、`motor_strength [0.7,1.3]`、
`motor_offset ±0.05`、`link_mass [0.1,2.0]`、`link_com ±0.1`、
**`ee_payload_mass [0.0, 1.5] kg`**（[config/wbc.py:261-275](../go1_gym/envs/config/wbc.py#L261-L275)）。

arm 扰动本身是**加速度重采样的随机运动**，不是回放轨迹：
`stage1_arm_max_accel 10.0`、`stage1_arm_max_vel 5.0`、
`accel_resample_time_s 0.01`、`zero_accel_probability 0.3`
（[config/wbc.py:250-258](../go1_gym/envs/config/wbc.py#L250-L258)），
强度由 `stage1_arm_ramp_iterations = 20000` 的课程斜升。

> **与全局不变量 12 的差距**：不变量要求「arm 扰动使用回放式轨迹，
> 负载随机化含质心偏置」。当前是随机加速度而非回放轨迹；
> 负载 `ee_payload_mass` 有质量但**无质心偏置**，
> 且 base 的 `randomize_com_displacement` 是**关**的。

### 地形

`mesh_type = "plane"`、`measure_heights = False`。地形随机化实质为零。

---

## 11. 环境 reset 时被重置的状态

`reset_idx`，[legged_robot.py:421-569](../go1_gym/envs/roboduet/legged_robot.py#L421-L569)。顺序：

```
_resample_commands(env_ids)              # L429
_arm_reset_hook(env_ids)                 # L430
_randomize_dof_props(env_ids, cfg)       # L431
_arm_post_dof_randomization_hook(env_ids)# L432
[randomize_rigids_after_start 分支 —— 当前关]
_reset_dofs(env_ids, cfg)                # L436
_reset_root_states(env_ids, cfg)         # L437
_arm_post_reset_refresh_hook(env_ids)    # L438
# --- 纯记账 ---
last_actions / last_last_actions = 0     # L441-442
last_dof_vel = 0                         # L443
dog_vel_ref = 0                          # L445   <-- 现有一阶参考模型
feet_air_time = 0                        # L446
episode_length_buf = 0                   # L447
reset_buf = 1                            # L448
[日志 / episode_sums 清零 / performance_metric_sums 清零]
gait_indices[env_ids] = 0                # L569
```

`_reset_root_states` 之后禁止写 sim 状态（AGENTS.md 明确约束）。

> **注意 `dog_vel_ref[env_ids] = 0`**：R2 不变量要求参考状态在 reset 时
> **对齐到实测值**，不是置零。当前实现置零；因为 reset 后 base 速度确实接近 0，
> 平地下差别很小，但姿态通道（pitch / height）reset 后不是 0，
> **一旦扩展到 5 通道，置零就是明确的 bug**。

---

## 12. 已有的一阶响应一致性实现（必须替换）

| 位置 | 内容 |
|---|---|
| [legged_robot.py:1914-1917](../go1_gym/envs/roboduet/legged_robot.py#L1914-L1917) | `dog_vel_ref` 缓冲，shape `(num_envs, 2)`，**只有 vx/vy** |
| [legged_robot.py:767-773](../go1_gym/envs/roboduet/legged_robot.py#L767-L773) | `_update_dog_vel_ref()`：`v_ref += (v_cmd − v_ref) · dt/T`，一阶 |
| [config/legged_robot.py:306-308](../go1_gym/envs/config/legged_robot.py#L306-L308) | `response_consistency_T = 0.4` s |
| [legged_robot.py:379-380](../go1_gym/envs/roboduet/legged_robot.py#L379-L380) | 调用点：`_update_performance_metrics()` 之后、**`compute_reward()` 之前** |
| [rewards.py:82-90](../go1_gym/envs/rewards/rewards.py#L82-L90) | `_reward_response_consistency`：`Σ(v_meas − v_ref)²` |
| [config/wbc.py:223](../go1_gym/envs/config/wbc.py#L223) | scale `−0.05` |
| [legged_robot.py:445](../go1_gym/envs/roboduet/legged_robot.py#L445) | reset 置零 |
| `benchmark/dog_policy/evaluation.py:1017,1050-1053,1151-1152` | 评测指标 `response_consistency_rmse` |

**与 R2 的差距**：一阶（R2 要二阶）、只 2 通道（R2 要 5）、无速率饱和、
reset 置零而非对齐实测、无标定流程。

**唯一已经正确的部分**：调用点在 `compute_reward()` 之前，
满足 R2「参考状态的更新必须发生在奖励计算之前」。

---

## 13. 训练入口、并行环境数、耗时与显存

### 入口

`scripts/auto_train.py`，[main() at L104](../scripts/auto_train.py#L104)。
默认 `--num_envs 4096`、`--robot go2_x5`、`--train_stage two_stage`、
`--num_learning_iterations 100000`。
stage 调度由 `StageSchedule.configure(global_switch)`
（[roboduet/utils.py:105-131](../go1_gym/envs/roboduet/utils.py#L105-L131)）。

配置快照：`parameters.pkl` 保存 `cfg_to_dict(cfg)` 的**完整 20 个 section**
（[core.py:65-73](../go1_gym/envs/config/core.py#L65-L73)，实测已验证），
以及 `RunnerArgs / ArmAC_Args / DogAC_Args / PPO_Args`。
另外 `runs/<name>/scripts/` 会拷贝 `roboduet/*.py` 和整个 `config/` 目录。

### R0 基线实测（本次勘察，2026-08-31）

```
python scripts/auto_train.py --train_stage stage1 --dyna_gait --headless \
       --no_wandb --num_envs 4096 --num_learning_iterations 200 \
       --run_name r0_baseline
```

硬件：NVIDIA GeForce RTX 5090（32607 MiB）。

| 指标 | 值 |
|---|---|
| 吞吐 | ~85,000 steps/s（collection 0.885 s + learning 0.270 s） |
| 单 iteration 耗时 | 起始 ~1.15 s，收敛后 ~1.65 s（episode 变长、reset 变少） |
| **200 iter 总耗时** | **287.5 s（4 分 48 秒）**，共 19,660,800 transitions |
| 显存占用 | **~9.1 GB**（进程），全卡 11.3 GB（含 2.1 GB 基线占用） |
| 每 iteration 采样 | 24 步 × 4096 env = 98,304 transitions |
| 训练健康度 | 200 iter 无报错、无 NaN；mean episode length 149 → 806，mean reward 0.033 → 0.238 |

> ⚠️ **200 iter 的 mean reward 不能用于版本间比较。** 实测：同一份代码、
> 三个固定种子（42/43/44），200 iter 末段 mean reward 分别是
> **0.2275 / 0.7408 / 2.6352**，散布 11.6 倍；episode length 584/917/882。
> 而总耗时是稳的（233/238/231 s，±1.5%）。
> 因此 200 iter 只能做「跑得通 / 无 NaN / 耗时正常」的冒烟门限，
> 任何奖励层面的结论都需要 R9 的多种子对照。

> 显存有充裕余量。R3 的 `(num_envs, 4速度桶, 16相位桶, 5通道)` float32
> 估计量只占 **5.2 MB**，R5/R6 的分组掩码可忽略，
> **新增状态不构成显存约束**。

---


## 13b. ⚠️ 两条训练入口，用的是不同的 learner（2026-09-01 发现）

| 入口 | learner 包 | actor 架构 |
| --- | --- | --- |
| **`scripts/auto_train.py`** | `go1_gym_learn/ppo_cse_automatic` | dog / arm **分离**，`DogActorCritic`，actor 吃 `obs_history`（非对称 actor-critic） |
| `scripts/unified_train.py` | `go1_gym_learn/ppo_cse_unified` | **统一双头** `Unified2AC`，actor 吃 `(privileged_estimate, obs)`，历史只喂 adaptation module 与 critic |

**`tmp/run_cluster.sh` 跑的是 `auto_train.py`**，因此
R7 的观测/编码器改动落在 `ppo_cse_automatic` 是正确的。

⚠️ 但我在第 1–7 步的冒烟训练全部用的是 `unified_train.py`——
**env 侧改动两条路都会经过（奖励、指标、命令、分组都在 env 里），
但 R7.3 的 `TemporalEncoder` 与 R7.4 的 checkpoint 检查只在 `auto_train.py` 路径上**。
第 8 步起冒烟测试改用 `auto_train.py`。

> 教训：`--train_stage stage1` 这个参数两个脚本都接受，跑起来都不报错、
> 曲线也都正常，**没有任何症状提示你选错了 learner**。

## 14. 与 R1–R9 相关的其它既有资产

| 资产 | 位置 | 对哪条需求有用 |
|---|---|---|
| dog policy 评测框架（含域随机化扫描、`response_consistency_rmse`） | [benchmark/dog_policy/](../benchmark/dog_policy/) | R9.1 |
| 性能指标（vx MAE/RMSE、roll/pitch RMS、height RMSE、滑移、功率、早停率） | [legged_robot.py:1063-1155](../go1_gym/envs/roboduet/legged_robot.py#L1063-L1155) | R9.1 指标 5 |
| reset 课程（按跟踪成功率放开 reset 随机化） | `_init/_update/_get_reset_curriculum_range` | R8.1 的模板 |
| Rerun 可视化 | [go1_gym/utils/viz.py](../go1_gym/utils/viz.py) | R3/R4 调试 |
| 4 个历史 stage-1 checkpoint | [benchmark/candidates/](../benchmark/candidates/) | R9 的「原版 WTW」对照基线 |

---

## 15. 事实汇总：与全局不变量清单的对照

| # | 不变量 | 当前状态 |
|---|---|---|
| 1 | 参考状态只在 reset 时对齐实测，重采样时连续 | ❌ reset 置零（L445）；重采样时不动 ✅ |
| 2 | 跟踪误差测量量需零延迟去趋势 | ❌ 完全没有 |
| 3 | 相位残差按速度分桶、per-env | ❌ 不存在 |
| 4 | 估计量更新路径与使用路径分离 | ❌ 不存在 |
| 5 | 跨域一致性用 nominal twin | ❌ 不存在，无 env 分组机制 |
| 6 | 组内相位同步；终止不对称时失效 | ❌ 不存在 |
| 7 | 参考模型参数取 20 分位数，姿态按相位联合分桶 | ❌ 无标定流程，`T=0.4` 是拍的 |
| 8 | 课程判据不含一致性奖励；辨识环境排除 | ✅ 已实现（R6）。前半从「自动满足」升级为启动断言（`CURRICULUM_PROGRESS_REWARDS` ∩ `CONSISTENCY_REWARDS` = ∅）；后半在 `_resample_commands` 内部按 `~is_identification_env` 过滤，运行期 spy 验证 145/161 精确排除 |
| 9 | 崎岖地形用相对地形高度 | ⚠️ 平地下无害；`_reward_jump` 硬编码 0 是定时炸弹 |
| 10 | 一致性奖励在乘性结构下作为 aux 因子 | ✅ 结构已就位（ji22 style） |
| 11 | σ 由标定得出，每通道独立 | ❌ 无标定 |
| 12 | arm 扰动用回放轨迹；负载含质心偏置 | ❌ 随机加速度；负载无质心偏置，base com 随机化关闭 |
| **13** | **（本方案新增）新增观测项必须机上可无歧义复现，误差源必须进 DR** | ❌ `dog.add_obs_noise=False`（[config/wbc.py:161](../go1_gym/envs/config/wbc.py#L161)），策略在“速度估计完美”假设下训练，而 `roboduet/base_lin_vel` 已是部署项（[export_rl_sar.py:242](../scripts/export_rl_sar.py#L242)） |
