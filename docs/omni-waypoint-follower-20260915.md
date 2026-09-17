# EE 地面投影与全向 waypoint PID

2026-09-15。取代 Raw 原来的近/远距离切换与 Visual 原来的前进/转向 follower。
入口仍是 `run_roboduet_raw_library.sh` 和 `run_visual_wholebody_library.sh`，
默认 `--base-mode follow`；`stand` 关闭底盘辅助。

## 几何约定

参考 EE 在世界系地面的投影为 `p = [ee_x, ee_y]`，`t_hat` 为参考轨迹的
单位 XY 切向。采用“沿切向后退，使 EE 投影留在狗前方”的平移约定：

```text
base_waypoint = p - (footprint_half_length + clearance) * t_hat
yaw_target   = atan2(t_hat_y, t_hat_x)
```

当前使用固定的 Go2 footprint 矩形：长宽各 0.60 m，前沿额外留 0.06 m。
因此底盘中心 waypoint 位于 EE 投影后方 0.36 m；在目标 yaw 下，EE 投影
比目标 footprint 的前沿再远 0.06 m。footprint 是包含站立足端的保守包络，
不是随每帧摆腿变化的接触多边形。

这个 waypoint 在世界系由轨迹定义，**不随实测底盘位置重新向外推**，
避免静止目标被每步推远。这里约束的是目标 footprint 的几何关系，实际
机器狗仍可能存在位置和 yaw 跟踪误差。

## 速度合成与 PID

全部向量先在世界 XY 系计算。“切向向量”按参考轨迹的水平速度定标：

```text
v_tangent = |v_reference_xy| * t_hat
v_sum     = v_base_measured_xy + v_tangent

e_xy      = base_waypoint - base_measured_xy
de_xy     = waypoint_velocity - v_base_measured_xy
v_pid     = Kp_xy * e_xy + Ki_xy * integral(e_xy) + Kd_xy * de_xy
v_world   = v_sum + v_pid

e_yaw     = yaw_target_unwrapped - yaw_measured_unwrapped
w_command = w_tangent + Kp_yaw * e_yaw + Ki_yaw * integral(e_yaw)
            + Kd_yaw * (w_tangent - w_measured)
```

对 `v_world` 做速度/加速度限制，转换到当前底盘坐标系，输出
`[vx_body, vy_body, yaw_rate]`。横移、后退可以直接输出；没有“先对准目标点
才能移动”的门槛。yaw 的目标始终是轨迹切向，实际转向速率受限。

| 参数 | Raw | DWBC | Visual |
| --- | ---: | ---: | ---: |
| XY Kp / Ki / Kd | 1.5 / 0.10 / 0.80 | 相同 | 相同 |
| yaw Kp / Ki / Kd | 2.0 / 0.10 / 0.25 | 相同 | 相同 |
| 总平移速度上限 | 0.50 m/s | 0.35 m/s | 0.30 m/s |
| 横移速度上限 | 0.45 m/s | 0.35 m/s | 0.02 m/s |
| 后退速度上限 | 0.35 m/s | 0.35 m/s | 0.03 m/s |
| 平移加速度上限 | 0.70 m/s² | 0.70 m/s² | 0.40 m/s² |
| yaw 角速度上限 | 1.00 rad/s | 0.60 rad/s | 0.40 rad/s |
| yaw 角加速度上限 | 2.00 rad/s² | 1.50 rad/s² | 0.75 rad/s² |

Visual 原生训练中 `vy` 采样为零，横移能力未经训练，因此执行侧采用更小
的横移和后退范围。PID 与几何逻辑相同；这不意味着原始策略已具备理想
全向底盘的跟踪能力。姿态变换后再次应用策略轴限幅，优先保证轴限幅；
该硬限幅可能优先于常规加速度限制。

积分有幅值限制和饱和时的条件积分。每条任务 reset 都清空积分、前一拍
命令和切向记忆。参考速度为零或到达末端时，速度前馈归零，PID 继续完成
位置修正；进入位置 0.025 m、yaw 0.03 rad 的容差后相应命令收敛到零。
末端保持最后有效切向；纯静止或纯竖直轨迹没有 XY 切向时使用初始 yaw。
这些容差只控制停车，不改变 benchmark 的成功判定。

yaw 目标和实测 yaw 都连续展开。这样当实际朝向落后超过 π 时，控制器仍沿
轨迹切向的连续旋转方向追赶，不会因为 `wrap_to_pi` 在 ±π 处突然反向。

## 接入方法

| 方法 | 接入方式 |
| --- | --- |
| RoboDuetRaw，`plan_vel=False` | PID 提供 vx/vy/yaw-rate；原 arm 策略继续输出臂动作和 pitch/roll，原 dog 策略执行腿动作 |
| Visual low-level | PID 提供三个速度命令；保留原生 arm IK 和工作空间投影；步态运动判断考虑 XY 速度 |
| DWBC | 将 PID 命令填入原来固定为零的三个速度观测槽，保留原始全身 actor |
| UMI | 原生 actor 直接输出全身关节动作，没有速度命令输入，不强行接入 |
| OCS2 / qm_control / WB-LocoMan | 使用已有底盘规划路径 |

DWBC 的直接 MuJoCo CLI 可用 `--dwbc-base-mode stand` 关闭辅助，默认 follow。
同一套 PID 没有修改任何 checkpoint、物理初态、FrozenReference 或评分逻辑。

## 窗口和运行记录

原有橙色 EE 参考、绿色实际轨迹、黄色当前 EE 目标保持不变。新增蓝色内容：

- 圆点：底盘 waypoint；
- 矩形：waypoint 处按目标 yaw 摆放的 footprint；
- 从圆点出发的线：轨迹切向。

`receipt.json` 的 `base_follower` 保存算法名、完整配置和源码 hash。
`trace.npz` 新增 `follower_*` 字段，包含地面投影、底盘 waypoint、切向、
yaw 目标、参考速度、实测速度、向量和、PID 修正以及实际下发命令。
`base_feedforward_command` 继续记录最终下发的三个底盘速度命令。

结果根目录：`benchmark/results/omni_follower_20260915/`。
此前两种 follower 的源码副本及旧运行产物均保留；Visual 调限速前的
`visual_v1/v2` 包含跌倒记录，不作为通过案例。

实现：[omni_waypoint_follower.py](../benchmark/wbc/omni_waypoint_follower.py)。

## 本次验证

40 项相关测试通过，包含几何偏移、世界/底盘坐标转换、切向 yaw、角度跨越
±π、横移、PID 积分与饱和、终点停车、reset、真实策略输入槽对照。
Python 编译与 benchmark static check 通过。

| 方法 | 完整时长尝试 | 跑完整段且无跌倒 | 说明 |
| --- | ---: | ---: | --- |
| Raw | 4 | 4 | A0/B0、A2/B2、直线 000、圆 009 |
| DWBC | 1 | 1 | A0/B0 |
| Visual | 4 | 1 | 圆 009 跑完；其余三条跌倒，失败均保留 |

这些是运行与接入证据。底盘实际 yaw 仍可能明显偏离切向目标，Visual 当前
也没有达到稳定播放条件；全向运动学控制器的输出不能代替原策略的执行能力。

逐帧核对了底盘 waypoint、切向 yaw、速度向量和、限幅及有限状态；离线
scorer 重算结果与 receipts 一致。Raw 的完整 viewer 与 headless trace hash
一致；Visual viewer 和 Raw 原生 stand 模式另有各 10 步检查。

- [完整验证记录](../benchmark/results/omni_follower_20260915/verification.json)
- [Raw A0/B0 控制器诊断图](../benchmark/results/omni_follower_20260915/raw_a0b0_follower.png)
- [诊断图 PDF](../benchmark/results/omni_follower_20260915/raw_a0b0_follower.pdf)

## frozen_trajectory_library2 A5/B3 诊断

该 cell 不是单纯增加 Kp 就能解决。原始 1.0 倍时间律要求：

- EE 水平参考速度峰值 0.489 m/s；
- 切向角速度分段 P95 为 1.48–3.21 rad/s，峰值 6.72 rad/s；
- 0.36 m 切向偏移后的底盘 waypoint 速度 P95 约 0.98 m/s；
- 约 15.8% 路段的曲率半径小于 0.36 m 的偏移距离。

旧限制 0.35 m/s、0.60 rad/s 时，平移和 yaw 命令分别约 95% 和 81% 时间
饱和，末端 waypoint 误差 1.47 m，平均绝对 yaw 误差约 105°，末端跌倒。

Raw 当前限制按 checkpoint 的实际训练范围提高到表中数值，并修复了 yaw
跨 π 后反向追赶。1.0 倍运行 `tuned_v2` 完成 1272 步、无跌倒；平均/末端
waypoint 误差约 0.40/0.50 m，平均绝对 yaw 误差约 68°。它能继续跟随，
但这条紧曲率轨迹仍无法让实际 yaw 始终贴住切向。

新增显式诊断参数 `--playback-speed`。0.15 倍运行 `slow_015` 完成 8194 步、
无跌倒；平均 waypoint 误差约 0.21 m，平均绝对 yaw 误差约 45°。该模式
保留空间路径，但改变冻结时间律，receipt 会写明
`diagnostic_slow_playback_not_taskspec_timing_comparable`，不能用于正式方法比较。
0.10 倍运行出现低层策略反向自旋，yaw 反而变差，因此不推荐继续降速。

```bash
cd /home/simon/Projects/Simon/wbc_rl_mpc

# 原冻结时间律；当前参数可跑完整段
./run_roboduet_raw_library.sh --viewer --cell 5 3

# 只用于更容易观察空间路径
./run_roboduet_raw_library.sh --viewer --cell 5 3 --playback-speed 0.15
```

诊断结果位于 `benchmark/results/omni_cell53_diagnosis_20260915/`；
`baseline`、`tuned_v1` 和 `slow_010` 的失败或退化结果均保留。
