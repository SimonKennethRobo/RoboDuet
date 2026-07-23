# 参数说明 / 调参指南（wbc.py）

本文件解释 `wbc.py` 中与**机械臂 EE 目标采样**和 **stage-2 DLS-IK 控制器**相关的参数。
`wbc.py` 里只保留裸参数、不写注释；所有解释集中在这里。

参数名 → 代码消费位置：
- `arm.target.*` → `WBCEnv._resample_arm_target`（wbc_env.py）
- `arm.ik.*` → `WBCEnv._solve_arm_dls_ik_step` / `_apply_stage2_arm_ik_action`

---

## arm.target.\*  —— EE 目标采样（绝对 box SE(3)）

每个 episode（以及每 `resample_time_s` 秒）在**基座坐标系**里独立采一个静态 SE(3) 目标。
**采用绝对 box，不做任何可达性筛选 / nominal 中心化**：目标可能落在 6 自由度机械臂
工作空间之外，此时 IK 会在关节限位/奇异处饱和，由 IK + 策略残差学着尽量逼近。

| 参数 | 含义 | 当前值 | 备注 |
|---|---|---|---|
| `arm.target.pos_range` | 位置 box，`[[x_lo,x_hi],[y_lo,y_hi],[z_lo,z_hi]]`，单位 m，基座系 | `[[0.0,0.55],[-0.4,0.4],[0.25,0.9]]` | 均匀采样。参考：默认位姿 nominal EE ≈ (0.26, 0, 0.71)；臂挂载点在 (0.1,0,0.1)，臂展约 0.6 m |
| `arm.target.roll_ee` | 姿态 roll 范围（rad），绕基座系 identity 的绝对 XYZ-Euler box | `±60°` | 绝对姿态，非 delta |
| `arm.target.pitch_ee` | 姿态 pitch 范围（rad） | `±75°` | pitch=0 → 夹爪指向基座 +x（水平向前） |
| `arm.target.yaw_ee` | 姿态 yaw 范围（rad） | `±90°` | |
| `arm.target.resample_time_s` | episode 内目标重采样周期 `[lo,hi]` 秒 | `[2.0, 3.0]` | 每次在 `[lo/dt, hi/dt]` 内随机 |

**采样公式**（`_resample_arm_target`）：
```
pos  = pos_range[:,0] + (pos_range[:,1]-pos_range[:,0]) * U(0,1)^3
quat = Rz(yaw) · Ry(pitch) · Rx(roll)          # 绝对姿态，基座系
```

**调参方向**
- 想缩小任务难度 / 提高收敛率：收窄 `pos_range`、减小三个 `*_ee` 半角，让 box 更贴合
  可达工作空间（当前绝对 box 收敛率 ~18%，clean 条件；越贴合越高）。
- 想让策略见到更广的 SE(3)：放大 box——但要接受相当比例目标不可达、稳态误差偏大。
- box 的中心当前隐含在 `pos_range` / identity 姿态里；若想把姿态 box 中心挪到「向前
  向下」的自然位姿（nominal pitch≈-37°），把 `pitch_ee` 改成非对称区间即可，无需改代码。

---

## arm.ik.\*  —— stage-2 DLS-IK 控制器

每个控制步做**一次**阻尼最小二乘（DLS）一阶修正（不是收敛求解器；FK 只在真实
`simulate()` 后更新，靠多步在仿真时间里收敛，等价于真实机器人的伺服环）。

```
dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · err          # err = [pos_err(3); axis_angle_rot_err(3)]
dq *= step_gain
dq  = clamp_by_norm(dq, max_step_rad)   # 按范数裁剪，保方向
q_target = dof_pos + dq + tanh(policy_raw) * residual_scale
```
其中 Jacobian 取 IsaacGym 世界系 Jacobian 在 `x5_link6` 行、机械臂 6 列，并把线速度行
从 link6 原点**点转移**到抓取点（`ee_local_pos` 偏移），使线性 Jacobian 与实际跟踪的
抓取点一致。

| 参数 | 含义 | 当前值 | 备注 |
|---|---|---|---|
| `arm.ik.damping` | DLS 阻尼 λ | `0.05` | 越大越稳、越慢；奇异附近抗爆 |
| `arm.ik.step_gain` | 每步修正增益 | `1.0` | full resolved-rate；位置驱动 + 足够刚度下可直接用 1.0 |
| `arm.ik.max_step_rad` | 单步 Δq 范数上限（rad） | `0.5` | 仅作奇异位形安全帽，常态不触发 |
| `arm.ik.residual_scale` | 策略 Δq 残差幅度（rad），`tanh(action)*scale` | `0.07`（≈4°） | IK 给基线跟踪，策略只做小幅微调 |
| `arm.ik.ee_local_pos` | 抓取点在 `x5_link6` 系下的固定偏移（m） | `[0.1424, 0.0, 0.0001057]` | URDF 的 `gripper_center`（fixed joint 被 collapse 掉），EE 状态每步按此平移 |

**已知精度**（`scripts/debug_ik_reach.py`，纯 IK、策略残差置零）
- 干净条件（贴合可达 box、无复位噪声、无臂 DR）：位置 median ~2 cm。
- 绝对 box（当前设置、无筛选）：位置 median ~10 cm、收敛率 ~18%——升高源于 box 含
  大量不可达目标，非 IK 数学问题（同一 DLS 在可达目标上收敛到 1–2 cm）。

**注意**
- 机械臂在 `control_type "M"` 下是**位置驱动**，PD 用 DOF 属性刚度（见
  `arm.control.stiffness_arm/damping_arm`），`Kp_factor` 域随机化对臂无效。
- 位置指令**不再**乘 `motor_strengths`（那会给关节角注入 IK 补不掉的误差，历史 bug）；
  执行器强度域随机化若要对臂生效，应加在 DOF 驱动刚度上。
