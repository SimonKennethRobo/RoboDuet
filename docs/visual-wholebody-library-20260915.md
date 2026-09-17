# Visual WholeBody：冻结轨迹库运行入口

**2026-09-15 后续更新**：默认底盘辅助已切换到
[共用全向 waypoint PID](omni-waypoint-follower-20260915.md)，目标 yaw 取轨迹
切向。本文原 follower 的配置和结果保留为历史记录；启动命令不变。

2026-09-15。目标是能运行并目视检查大致跟随，不以 tracking success 为门槛。

## 直接运行

在 `/home/simon/Projects/Simon/wbc_rl_mpc` 执行：

```bash
# 打开窗口，运行一条课程轨迹
./run_visual_wholebody_library.sh --viewer --cell 0 0

# 另一条已实际运行的课程轨迹
./run_visual_wholebody_library.sh --viewer --cell 2 2

# 指定随机直线；名称由 --list 列出
./run_visual_wholebody_library.sh --viewer --trajectory random-line-000
./run_visual_wholebody_library.sh --list

# 依次显示库中全部 68 条轨迹
./run_visual_wholebody_library.sh --viewer

# 无窗口运行，可重复 --cell / --trajectory 选择多个任务
./run_visual_wholebody_library.sh --cell 0 0 --cell 2 2
```

Shell 入口可从任意目录通过绝对路径执行，自动使用现有 isaacgym conda
环境中的 MuJoCo、CPU 策略推理与 GLFW，无需另起 ROS/MPC。
实时窗口需要桌面显示服务。橙线是冻结参考，绿线是实际 TCP，黄球是当前目标。
关闭当前窗口会进入下一条；Ctrl+C 停止本批次。

可选 `--output /新目录`；默认输出为
`benchmark/results/visual_library/<时间戳>/`。每条轨迹写入 `run.log`、
`trace.npz`、`receipt.json`，批次状态写入 `batch_state.json`。
输出目录必须不存在，旧结果不会被覆盖。

`--max-tasks N` 限制任务数量，`--max-steps N` 只用于短测。
省略两者时跑到原任务结束或跌倒终止。任务超时、失败记录仍保留，
`success=false` 不阻止继续查看下一条轨迹。

## 本次修复

- 新增 `benchmark/wbc/visual_mujoco.py`，沿用原 Visual checkpoint 和持久化
  DLS IK；从 checkpoint 相邻 `run_config.json` 读取观测契约、默认关节角、
  动作缩放、PD 参数和控制频率。
- 恢复原生播放入口的当拍动作。原 `ManipLoco.step()` 在
  `global_steps < 10000 * 24` 时使用最新动作，checkpoint 加载不恢复这个计数；
  之前 MuJoCo adapter 则无条件延迟一拍。默认采用原生播放时序，
  `--action-delay 1` 保留延迟诊断入口。
- 接入原仓库 `EEBaseFollower`，把非零 vx/yaw 命令、步态相位与时钟送入
  71D proprioception；保留 18D privileged 槽和 10 帧历史。
- 接入原 follower 的机械臂工作空间投影。控制目标额外记录到
  `controller_ee_position_m` / `controller_target_projected`；评分参考保持冻结值。
  接触观测在共同 MuJoCo plant 上按实际足端接触力的 1.5 N 阈值判断。
- 新增 `benchmark/data/run_visual_library.py`：直接消费
  `benchmark/data/frozen_trajectory_library/suite/trajectory_suite.json` 和
  `reference.npz`，验证库、TaskSpec 和参考 hash，并归档原 suite。
  不重新生成参考、不改时间律、不按方法重设物理初态或重新锚定。

这里使用 **Visual low-level policy + 仓库已有的启发式 base follower + IK**。
没有加载图像输入的 high-level policy，也没有重新训练。
现有 `cross_method_cli --method visual_wholebody` 同样使用修复后的 adapter。

## 实际运行记录

根目录：`benchmark/results/visual_repair_20260915/`。

| 输入 | 控制步数 | 运行表现 | 相对目录 |
| --- | ---: | --- | --- |
| A0/B0 | 935 | 跑完整段，无跌倒 | `a0b0_follow_v1` |
| A0/B1 | 1572 | 跑完整段，无跌倒，偏差较大 | `representative/runs/curriculum-a0-b1` |
| A2/B2 | 497 | 跑完整段，无跌倒 | `representative/runs/curriculum-a2-b2` |
| random-line-000 | 908 | 跑完整段，无跌倒，存在绕行及滞后 | `representative/runs/random-line-000` |
| random-circle-009 | 970 | 后段跌倒，已保留失败 | `representative/runs/random-circle-009` |

五次运行均没有非有限状态。全部 68 条另做了每条 2 步的输入/执行短测，
全部能加载并推进，记录在 `all68_smoke`；不能将短测说成全部轨迹跑完整段。

静止隔离探针只改变动作延迟时，旧实现在 50 步跌倒，取消延迟后完成 250 步，
最小 upright 为 0.973。该探针仍有末端漂移，记录在 `diagnostics/isolation.json`。
实时 viewer 已通过 Xvfb 下 10 步测试，路径为 `viewer_smoke`；没有声称验证用户桌面。

直线回放：
[`random-line-000/replay/mujoco_tracking.mp4`](../benchmark/results/visual_repair_20260915/representative/runs/random-line-000/replay/mujoco_tracking.mp4)。
这是已记录物理状态的离线回放，960×540、15 fps、18.2 s；黄色参考、青色实际 TCP。

27 个相关测试通过，包括直接提取原 `ManipLoco.compute_observations()`
比较完整 799D 输入和原 actor 动作，以及验证 follower 不改冻结目标。
运行测试时先收集 `test_mujoco.py`，保证 IsaacGym 在 Torch 之前导入：

```bash
cd /home/simon/Projects/WBC/RoboDuet
env LD_LIBRARY_PATH=/opt/miniconda3/envs/isaacgym/lib OMP_NUM_THREADS=1 \
  /opt/miniconda3/envs/isaacgym/bin/python -m pytest -q \
  benchmark/wbc/test_mujoco.py benchmark/wbc/test_visual_mujoco.py \
  benchmark/wbc/test_mujoco_adapters.py benchmark/wbc/test_cross_method_cli.py
```

旧 adapter 副本、早期失败、不同阶段 receipt/source hashes 均保留于该结果根目录。
本次没有修改 checkpoint、公共 MJCF、训练代码或已冻结的轨迹文件。
