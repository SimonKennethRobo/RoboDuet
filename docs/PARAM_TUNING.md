# 参数说明 / 调参指南（wbc.py）

本文件是 `go1_gym/envs/config/wbc.py` 中 `ROBODUET_OVERRIDES` 及相关常量的**唯一注释来源**。
`wbc.py` 里只保留裸参数、不写任何注释；所有含义、取值理由、调参方向集中在这里。

> 约定：`wbc.py` 用完整 `Cfg` 路径作为键，仅覆盖从 LeggedRobot → Go1 → WTW 继承链中
> **新增或修改**的字段；派生的观测维度和运行时特征开关在 `core.py` 里由这些源值计算，不在此表。

目录：

1. [机器人资产与臂接线（ROBOT_ASSET_FILES / ROBOT_ARM_SPEC）](#1-机器人资产与臂接线)
2. [臂 PD 增益（arm.control.stiffness_arm / damping_arm）](#2-臂-pd-增益)
3. [底层控制与资产（control.* / asset.*）](#3-底层控制与资产)
4. [环境与观测开关（env.*）](#4-环境与观测开关)
5. [stage-1 臂扰动课程（env.stage1_arm_*）](#5-stage-1-臂扰动课程)
6. [狗命令与奖励（commands.* / reward_scales.*）](#6-狗命令与奖励)
7. [狗 policy 布局与 critic 特权观测（dog.*）](#7-狗-policy-布局与-critic-特权观测)
8. [臂 policy 布局与目标采样（arm.num_actions_arm\* / num_privileged_links / arm.target.\*）](#8-臂-policy-布局与目标采样)
9. [stage-2 DLS-IK 控制器（arm.ik.\*）](#9-stage-2-dls-ik-控制器)
10. [WBC 奖励与终止（wbc.*）](#10-wbc-奖励与终止)
11. [域随机化（domain_rand.*）](#11-域随机化)

---

## 1. 机器人资产与臂接线

`ROBOT_ARM_SPEC` 把每个机器人的臂接线参数化，在 `core.configure_robot_asset` 中按 `--robot` 应用。

| 字段                 | 含义                                                                             |
| -------------------- | -------------------------------------------------------------------------------- |
| `ee_body_name`     | 被跟踪的末端连杆（go1/go2 =`zarx_body6`；go2_x5 = `x5_link6`）               |
| `ee_local_pos`     | 把跟踪点从该连杆原点平移到抓取点的偏移（仅 x5 臂的 gripper_center 需要，见 §9） |
| `mount_joint_name` | mount 随机化作用的固定关节（其 transform 被抖动）                                |

`ROBOT_ASSET_FILES` 为各机器人的 URDF 路径。三个机器人（go1 / go2 / go2_x5）共用一套 union 配置。

---

## 2. 臂 PD 增益

`arm.control.stiffness_arm` / `damping_arm` 按 **精确 DOF 名**查表（见 `_process_dof_props` 与
`_init_buffers` 的增益推导）。它是**所有已注册臂 DOF 名的并集**——未命中的键永远不会被用到，
所以同一张表同时服务 arx/zarx 臂（go1、go2）和 x5 臂（go2_x5）。给新臂加 DOF 时往表里补键即可，
不会影响其它机器人。

---

3. 底层控制与资产

| 参数                                  | 含义 / 备注                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `control.control_type`              | `"M"` = mixed：腿用力矩控制，臂 slice 用位置目标                                                                                                                 |
| `control.update_obs_freq`           | **仅当 `use_vision=True` 时生效**，视觉观测更新频率                                                                                                        |
| `asset.penalize_contacts_on`        | 这些连杆接触受碰撞惩罚                                                                                                                                             |
| `asset.terminate_after_contacts_on` | `[""]` = 不因接触终止 episode                                                                                                                                    |
| `asset.self_collisions`             | **1 = 关闭自碰，0 = 开启自碰**（IsaacGym 语义）                                                                                                              |
| `asset.render_sphere`               | 是否在 viewer 里画出**EE 目标可视化球**（`legged_robot.py:151`）。**纯渲染开关，不影响物理/训练**；headless 下无效。play 置 True，benchmark 置 False |

> 注：腿/臂的真实 PD 增益不在 `control.*`，而在 `dog.control.stiffness_leg`/`damping_leg`（§7）与
> `arm.control.stiffness_arm`/`damping_arm`（§2）。原先的 `control.stiffness`/`control.damping` 已删除
> （前者只被当分组 key 用、值失效，后者零读取）。

---

## 4. 环境与观测开关

| 参数                                        | 含义 / 备注                                                                                                                                                                                                                                                |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env.keep_arm_fixed`                      | True 时臂**不由臂 policy 驱动**，而是被 `_keep_arm_fixed()` 保持在每个 env 复位时随机化的固定位姿上（`wbc_env.py:517`）。stage-1 训练狗时臂当作静态负载/扰动源；开关/课程会进一步调制（见 §5）。臂 policy 接管（stage-2/switch_open）后此项让位 |
| `env.num_actions`                         | 总动作维（12 腿 + 6 臂）                                                                                                                                                                                                                                   |
| `env.priv_observe_*`                      | 是否把对应量加入**critic 特权观测**（base_mass、com、Kp/Kd、dof_damping、vel 等）                                                                                                                                                                    |
| `env.priv_observe_high_freq_goal`         | 是否向特权观测加入**未降采样**的目标相对 EE 位姿                                                                                                                                                                                                     |
| `env.observe_two_prev_actions`            | 是否观测上上一步动作                                                                                                                                                                                                                                       |
| `env.record_video` / `recording_*`      | 录像开关与分辨率/帧步长/叠加文字轨迹                                                                                                                                                                                                                       |
| `env.debug_viz`                           | 调试可视化                                                                                                                                                                                                                                                 |
| `env.arm_policy_enabled`                  | 臂 policy 路径是否可用。**仅被双 policy runner 在纯 stage-1 训练时置 False**；two_stage / stage-2 / unified / play 都保持 True                                                                                                                       |
| `env.arm_observe_dog_state`               | 跨 policy 通道：让臂 policy 看到狗的 gait phase、足端接触状态、`v_actual − v_cmd` 跟踪残差                                                                                                                                                              |
| `env.priv_observe_stage1_ee_payload_mass` | 把 stage-1 EE payload 质量加入特权观测（配合 §11 的 payload 随机化）                                                                                                                                                                                      |

---

## 5. stage-1 臂扰动课程

stage-1 训练腿部 policy 时，臂不是简单固定，而是按课程逐步加入激进扰动（模拟臂运动对 base 的反作用力）。
强度 `intensity` 随迭代从 0 线性 ramp 到 1（见 `_get_stage1_arm_curriculum_intensity`）。

| 参数                                      | 含义                                | 当前值    |
| ----------------------------------------- | ----------------------------------- | --------- |
| `env.stage1_arm_ramp_iterations`        | 扰动强度从 0 → 满值的迭代数        | `20000` |
| `env.stage1_arm_fixed_fraction`         | 保持臂完全固定的 env 比例（对照组） | `0.1`   |
| `env.stage1_arm_saturation_fraction`    | 达到满强度的迭代占 ramp 的比例      | `0.8`   |
| `env.stage1_arm_accel_resample_time_s`  | 臂目标加速度重采样周期（s）         | `0.01`  |
| `env.stage1_arm_zero_accel_probability` | 每次重采样置零加速度的概率          | `0.3`   |
| `env.stage1_arm_zero_vel_probability`   | 每步置零速度的概率                  | `0.005` |
| `env.stage1_arm_max_accel`              | 臂目标最大加速度（满强度时）        | `10.0`  |
| `env.stage1_arm_max_vel`                | 臂目标最大速度（满强度时）          | `5.0`   |
| `env.stage1_arm_init_dof_pos_noise`     | 复位时臂初始关节角随机噪声幅度      | `1.0`   |

**调参方向**：狗训不稳/爱摔 → 减小 `max_accel` / `max_vel` 或拉长 `ramp_iterations`（更慢加压）；
想让狗对臂扰动更鲁棒 → 反之增大。`init_dof_pos_noise` 越大，臂初始位形越发散、top-heavy 风险越高
（详见 memory: arm-ik-motor-strength-floor 里对 go2_x5 top-heavy 的分析）。

---

## 6. 狗命令与奖励

| 参数                                               | 含义 / 备注                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `commands.body_roll_range` / `limit_body_roll` | body roll 命令范围与硬限                                                                                                                                                                                                                                                                                                          |
| `commands.T_force_range`                         | 外力持续时间范围（s），**仅当 `randomize_end_effector_force=True` 生效**                                                                                                                                                                                                                                                  |
| `commands.add_force_thres`                       | 施加外力的触发阈值                                                                                                                                                                                                                                                                                                                |
| `rewards.terminal_body_height`                   | 低于此高度判摔倒终止                                                                                                                                                                                                                                                                                                              |
| `reward_scales.loco_energy`                      | 腿部能耗惩罚系数                                                                                                                                                                                                                                                                                                                  |
| `reward_scales.response_consistency`             | **关键**：惩罚腿部 base 速度响应偏离 `commands_dog` 的**一阶参考模型**（见 `_reward_response_consistency` 与 `LeggedRobot._update_dog_vel_ref`）。作用：让腿在任意 payload/posture 扰动下都表现为**固定时间常数的可预测线性 plant**，上层 `v_ff` 前馈才能依赖它。这是 project-design-v3.md §2.3 的落地 |

---

## 7. 狗 policy 布局与 critic 特权观测

| 参数                                                                        | 含义                                                                                                                                                                         |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dog.num_actions_loco`                                                    | **环境侧**的腿部执行 DOF 数（12）= dof/torque/action 张量里 **腿 slice `[:12]` 与臂 slice `[12:]` 的分界索引**，rewards、切片到处用它。是机器人 DOF 布局属性 |
| `dog.dog_actions`                                                         | **狗 policy 的动作输出维**（12）= 狗 actor 网络输出宽度（`load_policy.py:35`）与 `dog_actions` 观测项宽度（`wbc_env.py:1374`）                                   |
| `dog.dog_num_observation_history`                                         | 狗观测历史长度                                                                                                                                                               |
| `dog.dog_num_commands`                                                    | 狗命令维                                                                                                                                                                     |
| `dog.use_adaptation_module`                                               | 是否用 adaptation module（当前关）                                                                                                                                           |
| `dog.observe_lin_vel` / `observe_pose_actual` / `observe_track_error` | 狗观测内容开关                                                                                                                                                               |

**狗 critic 独有的特权动力学观测**（`dog.priv_observe_*`）：腿的共享因子紧凑表示，臂因子按关节采样故逐关节保留。

| 参数                                                                                                         | 值                             | 说明                                                                                       |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------ |
| `dog.priv_observe_motor_strength` / `motor_offset` / `gravity` / `contact_states` / `arm_dynamics` | True                           | 加入狗 critic 特权观测                                                                     |
| `dog.priv_observe_com_displacement` / `joint_friction` / `dof_damping`                                 | False                          | 本 profile 中这些是**固定值**，故从狗特权观测中省略（臂 critic 仍可见共享 env 字段） |
| `dog.control.stiffness_leg` / `damping_leg`                                                              | `{joint:35}` / `{joint:1}` | 腿 PD 增益                                                                                 |

---

## 8. 臂 policy 布局与目标采样

### 8.0 臂布局参数

| 参数                                | 含义                                                                                                                                                                                                                                                                                                                                                                                                 | 当前值    |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `arm.num_actions_arm`             | 臂**执行**的关节残差维数 = env 在 arm slice 上实际施加的 Δq 维度（6 轴 → 6）                                                                                                                                                                                                                                                                                                                 | `6`     |
| `arm.num_actions_arm_cd`          | 臂**policy actor 的输出宽度**（"cd" = 含协调/plan 通道版本）。actor 输出前 `num_actions_arm` 维当关节残差、其余当 plan 通道（历史上 `v_ff`/posture，取末 2 维经 `env.plan(a[...,-2:])`）。**当前 DLS-IK 架构无 plan 通道**，故 `num_actions_arm_cd == num_actions_arm == 6`。消费：`Unified2AC_Args.num_actions_arm`、actor/loader、privileged arm slice（`wbc_env.py:675`） | `6`     |
| `arm.num_privileged_links`        | 进入**臂 critic 特权观测**的臂连杆数（其随机化 link-mass-scale / com-offset 打包进 priv obs）。运行时会与实际臂刚体数断言相等（`wbc_env.py:866`），改臂/改 URDF 连杆数时需同步                                                                                                                                                                                                               | `8`     |
| `arm.arm_num_observation_history` | 臂观测历史长度                                                                                                                                                                                                                                                                                                                                                                                       | `60`    |
| `arm.arm_num_commands`            | 暴露给狗 policy 观测的臂命令槽位数                                                                                                                                                                                                                                                                                                                                                                   | `6`     |
| `arm.use_adaptation_module`       | 是否用 adaptation module（当前关）                                                                                                                                                                                                                                                                                                                                                                   | `False` |

> `num_actions_arm` vs `num_actions_arm_cd`：前者是**环境真正施加**的臂关节残差数，后者是**策略网络输出**的总维数。二者相等仅因当前没有额外协调通道；一旦加回 `v_ff`/posture 输出，`_cd` 会大于 `num_actions_arm`。

### 8.1 目标采样（绝对 box SE(3)）

每个 episode（以及每 `resample_time_s` 秒）在**基座坐标系**里独立采一个静态 SE(3) 目标。
**采用绝对 box，不做任何可达性筛选 / nominal 中心化**：目标可能落在 6 自由度机械臂
工作空间之外，此时 IK 会在关节限位/奇异处饱和，由 IK + 策略残差学着尽量逼近。

消费位置：`WBCEnv._resample_arm_target`（wbc_env.py）。

| 参数                           | 含义                                                          | 当前值                                 | 备注                                                                                     |
| ------------------------------ | ------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `arm.target.pos_range`       | 位置 box`[[x_lo,x_hi],[y_lo,y_hi],[z_lo,z_hi]]`，m，基座系  | `[[0.0,0.55],[-0.4,0.4],[0.25,0.9]]` | 均匀采样。参考：默认位姿 nominal EE ≈ (0.26,0,0.71)；臂挂载点 (0.1,0,0.1)，臂展约 0.6 m |
| `arm.target.roll_ee`         | 姿态 roll 范围（rad），绕基座系 identity 的绝对 XYZ-Euler box | `±60°`                             | 绝对姿态，非 delta                                                                       |
| `arm.target.pitch_ee`        | 姿态 pitch 范围（rad）                                        | `±75°`                             | pitch=0 → 夹爪指向基座 +x（水平向前）                                                   |
| `arm.target.yaw_ee`          | 姿态 yaw 范围（rad）                                          | `±90°`                             |                                                                                          |
| `arm.target.resample_time_s` | episode 内目标重采样周期`[lo,hi]` 秒                        | `[2.0, 3.0]`                         | 每次在`[lo/dt, hi/dt]` 内随机                                                          |

**采样公式**（`_resample_arm_target`）：

```
pos  = pos_range[:,0] + (pos_range[:,1]-pos_range[:,0]) * U(0,1)^3
quat = Rz(yaw) · Ry(pitch) · Rx(roll)          # 绝对姿态，基座系
```

**调参方向**

- 想缩小任务难度 / 提高收敛率：收窄 `pos_range`、减小三个 `*_ee` 半角，让 box 更贴合可达工作空间。
- 想让策略见到更广的 SE(3)：放大 box——但要接受相当比例目标不可达、稳态误差偏大。
- 姿态 box 中心当前隐含为 identity；若想把中心挪到「向前向下」自然位姿（nominal pitch≈-37°），把
  `pitch_ee` 改成非对称区间即可，无需改代码。

---

## 9. stage-2 DLS-IK 控制器

每个控制步做**一次**阻尼最小二乘（DLS）一阶修正（不是收敛求解器；FK 只在真实 `simulate()` 后更新，
靠多步在仿真时间里收敛，等价于真实机器人的伺服环）。消费位置：`_solve_arm_dls_ik_step` /
`_apply_stage2_arm_ik_action`。

```
err  = [pos_err(3); axis_angle_rot_err(3)]
W^.5 = diag([√pos_weight]*3, [√rot_weight]*3)   # 任务空间加权
J   ← W^.5 · J,   err ← W^.5 · err               # 同时缩放 J 行与 err
dq  = Jᵀ (J Jᵀ + λ²I)⁻¹ · err                    # 加权阻尼最小二乘
dq *= step_gain
dq  = clamp_by_norm(dq, max_step_rad)             # 按范数裁剪，保方向
q_target = dof_pos + dq + tanh(policy_raw) * residual_scale
```

其中 Jacobian 取 IsaacGym 世界系 Jacobian 在 `ee_body_name` 行、机械臂 6 列，并把线速度行
从连杆原点**点转移**到抓取点（`ee_local_pos` 偏移），使线性 Jacobian 与实际跟踪的抓取点一致。

| 参数                      | 含义                                             | 当前值                   | 备注                                                                                                                         |
| ------------------------- | ------------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `arm.ik.damping`        | DLS 阻尼 λ                                      | `0.1`                  | 越大越稳、越慢；奇异附近抗爆                                                                                                 |
| `arm.ik.step_gain`      | 每步修正增益                                     | `1.0`                  | full resolved-rate；位置驱动 + 足够刚度下可直接用 1.0                                                                        |
| `arm.ik.max_step_rad`   | 单步 Δq 范数上限（rad）                         | `0.5`                  | 仅作奇异位形安全帽，常态不触发                                                                                               |
| `arm.ik.residual_scale` | 策略 Δq 残差幅度（rad），`tanh(action)*scale` | `0.07`（≈4°）        | IK 给基线跟踪，策略只做小幅微调                                                                                              |
| `arm.ik.pos_weight`     | 位置误差任务权重                                 | `1.0`                  | 加权 DLS 里 pos 块的权重；相对 `rot_weight` 越大，6-DoF 臂越优先消位置误差、牺牲姿态                                        |
| `arm.ik.rot_weight`     | 姿态误差任务权重                                 | `3.0`                  | 同上，越大越优先姿态。两者只看**相对比例**（等值 = 未加权，与原行为完全一致）。当前 `3.0` = 姿态优先于位置                    |
| `arm.ik.ee_local_pos`   | 抓取点在`ee_body_name` 系下的固定偏移（m）     | `[0.1424,0,0.0001057]` | URDF`gripper_center`（fixed joint 被 collapse），EE 状态每步按此平移。与 `ROBOT_ARM_SPEC` 保持一致，由 core 按机器人覆盖 |

**架构说明**：DLS-IK 架构下臂 policy **只输出 Δq 残差**，没有 plan-action 通道（`v_ff`/posture 输出
延后，见 project-design-v3.md §5），故 `arm.num_actions_arm_cd == arm.num_actions_arm == 6`。

**已知精度**（`scripts/debug_ik_reach.py`，纯 IK、策略残差置零）

- 干净条件（贴合可达 box、无复位噪声、无臂 DR）：位置 median ~2 cm。
- 绝对 box（当前设置、无筛选）：位置 median 偏大、收敛率偏低——源于 box 含大量不可达目标，
  非 IK 数学问题（同一 DLS 在可达目标上收敛到 1–2 cm）。

**注意**

- 臂在 `control_type "M"` 下是**位置驱动**，PD 用 DOF 属性刚度（见 §2），`Kp_factor` 域随机化对臂无效。
- 位置指令**不再**乘 `motor_strengths`（那会给关节角注入 IK 补不掉的误差，历史 bug）；
  执行器强度域随机化若要对臂生效，应加在 DOF 驱动刚度上。

---

## 10. WBC 奖励与终止

stage-2 MVP（project-design-v3.md 的 DLS-IK-only 切片）：base 不动，臂 policy 在静态 SE(3) 目标上
叠加 Δq 残差。`v_ff/ρ`、`γ(s)/s_ref(t)`、multi-critic、安全滤波器本轮**刻意不在范围内**。

| 参数                                                        | 含义 / 备注                                                                   |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `wbc.use_vision`                                          | 视觉开关（本轮 False）                                                        |
| `wbc.rewards.use_terminal_body_height`                    | 是否因 body height 超限终止（True）                                           |
| `wbc.rewards.use_terminal_roll` / `use_terminal_pitch`  | 是否因 roll/pitch 超限终止（当前 False）                                      |
| `wbc.rewards.terminal_body_{height,roll,pitch}`           | 对应终止阈值                                                                  |
| `rewards.ee_pos_tracking_sigma`                           | `exp(-err²/σ)` 位置跟踪 σ（m²）；`0.02` → 约 14 cm 误差时 reward=0.5 |
| `rewards.ee_rot_tracking_sigma`                           | 姿态跟踪 σ（rad²）；`0.25` → 约 35° 误差时 reward=0.5                   |
| `wbc.reward_scales.ee_pos_tracking` / `ee_rot_tracking` | 位置/姿态跟踪奖励权重（`4.0` / `1.0`）                                    |
| `wbc.reward_scales.jump`                                  | 跳跃奖励                                                                      |
| `wbc.reward_scales.hip_action_l2`                         | hip 动作 L2 惩罚                                                              |
| `wbc.reward_scales.raibert_heuristic`                     | Raibert 落足启发（当前 0）                                                    |
| `wbc.reward_scales.arm_control_limits`                    | 臂残差饱和惩罚（用 pre-combine 的`arm_residual_raw`）                       |
| `wbc.reward_scales.ee_smoothness`                         | EE 平滑惩罚                                                                   |
| `wbc.reward_scales.arm_contact`                           | 臂接触惩罚                                                                    |

`WBC_REWARD_FACTORS`（非 Cfg 常量，用于派生 WBC 奖励）：`tracking_lin_vel`/`tracking_ang_vel` 及臂
energy/dof_vel/dof_acc/action_rate/smoothness 的基础因子。

---

## 11. 域随机化

**base / mount**

| 参数                                                                | 含义 / 备注                                        |
| ------------------------------------------------------------------- | -------------------------------------------------- |
| `domain_rand.dog_obs_frame_drop_prob`                             | 狗观测丢帧概率                                     |
| `domain_rand.added_mass_range`                                    | base 附加质量范围（kg）                            |
| `domain_rand.randomize_end_effector_force`                        | 是否施加 EE 外力（配合`commands.T_force_range`） |
| `domain_rand.max_force` / `max_force_offset`                    | 外力大小与作用点偏移                               |
| `domain_rand.randomize_mount_position` / `mount_position_range` | 臂挂载点位置随机化`[[x],[y],[z]]`（m）           |
| `domain_rand.randomize_mount_rotation` / `mount_rpy_range`      | 臂挂载点姿态随机化（rad，约 ±3°/±3°/±5°）    |
| `domain_rand.mount_tf_buckets` / `mount_tf_bucket_seed`         | mount transform 分桶数与种子（离散化以复现）       |

**stage1_arm.\***（臂动力学随机化，stage-1 用；范围较宽）：`Kp_factor` `[0.5,1.5]`、`Kd_factor` `[0.2,2.0]`、
`motor_strength` `[0.7,1.3]`、`motor_offset` `0.05`、`link_mass` `[0.1,2.0]`、`link_com` `0.1`。

**EE payload**（`stage1_arm.randomize_ee_payload`）：每 episode 随机质量 `ee_payload_mass_range=[0.0,1.5] kg`，
以**持续的、重力对齐的力**施加在 EE 刚体上（不是刚体质量编辑——IsaacGym 只允许在 actor 创建时改质量）。
按与其余臂扰动相同的 stage1 课程强度缩放（见 `_get_stage1_arm_curriculum_intensity`）。

**stage2_arm.\***（臂动力学随机化，stage-2 用；范围较窄，因为 stage-2 要 cm 级精度，DR 过猛会向 EE 误差
注入不可消除的噪声——见 project-design-v3.md §1.2）：`Kp/Kd_factor` `[0.9,1.1]`、`motor_strength`
`[0.85,1.15]`、`motor_offset` `0.025`、`link_mass`/`link_com` 随机化**关闭**。
