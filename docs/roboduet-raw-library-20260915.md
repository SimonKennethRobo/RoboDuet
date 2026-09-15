# RoboDuetRaw：冻结轨迹库运行入口

2026-09-15。沿用用户最新验收：能运行、目视上大致跟随，不以 tracking success 为门槛。

**当前算法已更新为共用全向 waypoint PID**：EE 地面投影沿切向后退至
footprint 外，yaw 跟随切向，实测速度与参考速度合成后加 PID 修正。
下面的近/远距离两段逻辑保留为历史记录；最新定义、限幅和运行字段见
[全向 waypoint PID](omni-waypoint-follower-20260915.md)。启动命令不变。

**后续修正：A0/B0 底盘不动。** 原版近距离门槛使 A0/B0 整段的底盘速度
命令为零。默认 `follow` 现已让底盘跟随近距离参考的 XY 位移，原启动命令
无需增加参数。修正后 A0/B0 跑完 935 步、无跌倒，底盘相对起点最大平移
约 0.278 m，EE 位置 RMSE 约 0.076 m。新结果位于
`benchmark/results/raw_base_motion_20260915/`；下方首次验证记录保留供对照。

## 直接启动

在 `/home/simon/Projects/Simon/wbc_rl_mpc` 执行：

```bash
# 近距离课程轨迹，先看这条
./run_roboduet_raw_library.sh --viewer --cell 0 0

# 移动范围更大的课程轨迹和随机直线
./run_roboduet_raw_library.sh --viewer --cell 2 2
./run_roboduet_raw_library.sh --viewer --trajectory random-line-000

# 查看名称 / 依次播放全部 68 条
./run_roboduet_raw_library.sh --list
./run_roboduet_raw_library.sh --viewer

# 去掉 --viewer 可无窗口运行，选择参数可重复
./run_roboduet_raw_library.sh --cell 0 0 --trajectory random-line-000
```

脚本可以从任意目录通过绝对路径运行，使用已有 isaacgym conda 环境中的
MuJoCo 和 CPU 策略推理，无需启动 ROS/MPC。实时窗口需要可用桌面。
橙线为原参考，绿线为实际 TCP，黄球为当前参考目标。
关闭当前窗口会进入下一条；Ctrl+C 停止本批次。

默认输出：`benchmark/results/roboduet_raw_library/<时间戳>/`。
每条轨迹有 `run.log`、`trace.npz`、`receipt.json`，批次进度写入
`batch_state.json`。`--output /新目录` 可指定输出，已有目录不会被覆盖。
`--max-steps N` / `--max-tasks N` 用于限制短测。

## 本次改动

- 保留原始 dog actor、arm actor、arm adaptation/history encoder，以及
  arm actor 输出的身体 pitch/roll 规划。当前 checkpoint 的
  `plan_vel=False`，原生模式本来就不自主规划底盘平移。
- 默认 `follow` 底盘辅助在近处保留起始臂端与底盘的水平偏移，让底盘跟随
  参考的 XY 位移并保持初始 yaw；支持前后和横向移动，平移速度上限
  0.25 m/s，带位置反馈、速度阻尼和加速度限制。
- 当目标离底盘超过 0.60 m，或方位角超过 1.30 rad 时，切换到已有的
  `EEBaseFollower` 接近目标，并保持该模式直到本条任务结束。
  远处接近的距离目标为 0.50 m，前进上限 0.35 m/s，yaw 上限
  0.60 rad/s，不主动倒车。两种辅助都保留原 arm/posture 策略。
- 默认将送入双策略的目标限制在该 checkpoint 的训练球坐标和姿态范围内。
  球坐标仍使用原实现的 ground + 0.38 m 中心，姿态观测仍使用原
  `quat_to_angle` 的投影轴角，而不是用标准 Euler 直接替代观测。
- 冻结 TaskSpec、时间律、物理初态、原始参考和 scorer 保持原定义。
  `controller_ee_position_m` / `controller_ee_quaternion_xyzw` /
  `controller_target_projected` 单独记录投影后的控制输入；底盘实际命令
  记录在 `base_feedforward_command`。
- 与 Visual 共用既有批量 runner；Visual 的启动命令和默认参数保持兼容。

原生姿态规划、零平移、原始目标输入可用以下诊断组合重放：

```bash
./run_roboduet_raw_library.sh --viewer --cell 0 0 \
  --base-mode stand --target-mode native
```

默认实现应称为 **RoboDuetRaw 双策略 + 启发式底盘跟随辅助**，不能将额外的
底盘跟随能力写成 `plan_vel=False` 原始 arm actor 自主学到的能力。
本次没有重训、替换权重，也没有用 IK 取代 learned arm actor。

## 首次验证与产物（近距离修正前）

结果根目录：`benchmark/results/raw_repair_20260915/`。
最终代表轨迹批次：`representative_final`。

| 轨迹 | 控制步数 | 结果 |
| --- | ---: | --- |
| A0/B0 | 935 | 跑完整段，无跌倒/非有限状态 |
| A2/B2 | 497 | 跑完整段，无跌倒/非有限状态 |
| random-line-000 | 908 | 跑完整段，无跌倒/非有限状态 |
| random-circle-009 | 1029 | 跑完整段，无跌倒/非有限状态 |

近距离 A0/B0 位置 RMSE 约 0.087 m；长程和圆轨迹仍有明显滞后、绕行和姿态
误差。这些结果用于确认可运行与查看运动效果，不表示跟踪达标。

全库另做了 68 条各 2 步短测，位于 `all68_smoke`；这不是全部 68 条完整
时长验证。`viewer_smoke` 是 Xvfb 下 10 步真实 viewer 检查。
`visual_regression_smoke` 验证共享 runner 改动后 Visual 仍能运行。

回放均由最终 trace 中的物理状态生成，960×540、15 fps：

- [A0/B0 回放](../benchmark/results/raw_repair_20260915/representative_final/runs/curriculum-a0-b0/replay/mujoco_tracking.mp4)
- [随机直线回放](../benchmark/results/raw_repair_20260915/representative_final/runs/random-line-000/replay/mujoco_tracking.mp4)

30 个相关测试通过。新增测试直接提取原生源码，比对 20D arm 观测、600D
arm 历史、56D dog 当前观测、双策略推理输入和 pitch/roll 规划；覆盖原始目标
不被投影覆盖、近处/远处跟随及任务 reset。近距离修正后已重新通过这些测试。

旧实现副本、`a0b0_before`、试验阶段 `representative_v1/v2` 均保留。
最终批次、短测、回放各自保留自己的 trace/model/config/source hash。

## 近距离修正后的回放

- [A0/B0 底盘参与跟随](../benchmark/results/raw_base_motion_20260915/a0b0_v1/runs/curriculum-a0-b0/replay/mujoco_tracking.mp4)
- `a0b0_v1` 为完整 headless 运行，`viewer_a0b0` 为同轨迹完整 Xvfb viewer
  运行，`representative` 为其余三条代表轨迹复查。
- 原 `raw_repair_20260915/` 中的回放对应修正前版本。

修正后四条代表轨迹均跑完整段，无跌倒或数值故障：

| 轨迹 | 底盘相对起点最大 XY 位移 | EE 位置 RMSE |
| --- | ---: | ---: |
| A0/B0 | 0.278 m | 0.076 m |
| A2/B2 | 1.380 m | 0.590 m |
| random-line-000 | 2.272 m | 0.096 m |
| random-circle-009 | 0.788 m | 0.084 m |

完整 viewer 与 headless 的 A0/B0 trace hash 一致。验证记录：
[verification.json](../benchmark/results/raw_base_motion_20260915/verification.json)。
