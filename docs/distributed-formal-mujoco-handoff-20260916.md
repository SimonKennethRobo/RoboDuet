# 分布式正式 MuJoCo 测评交接（2026-09-16）

本文件是下一 session 的直接执行入口，覆盖
`docs/benchmark-handoff-20260914.md` 第 38 节中已经过期的地形和单机启动参数。
用户已经确认：不再把外部方法的复现完整性作为门槛，所有方法均视为已复现；下一
session 收到“开始执行”后应直接完成下述 P0/P1、smoke 和正式启动，不重新做方案讨论，
也不运行旧的训练脚本。

## 1. 最终测评口径

- 方法：`roboduet`、`roboduet_raw`、`umi`、`visual_wholebody`、
  `wb_locoman`、`qm_control`、`deep_whole_body_control`、`ma2022`。
- 任务：冻结的 `frozen_trajectory_library2`，68 条轨迹。
- 场景：每条轨迹均运行 `nominal` 和 `push`，合计 544 个 method-task job、
  1088 个 scenario result；失败必须保留在分母。
- 统一 TaskSpec/reference/trace/scorer 不变。扰动为 base COM、world +X、80 N、
  0.1 s，目标冲量 8 N·s。
- `umi` 是唯一允许的 method-specific dynamics 例外，继续使用
  `training_nominal`：腿 damping 0.1 Nms/rad、frictionloss 0.025 Nm、足端
  slide friction 1.0。其余环境、任务、扰动和 scorer 仍统一。
- 所有 scenario 保存原始 `trace.npz` 和 receipt；本轮必须录制 MP4。

### RoboDuet（ours）冻结身份

`roboduet` 必须使用以下 policy 与 MPC 配置，不能退回默认 `task.info`：

```text
policy key: robot_lab_rear_r30o_s42_11497
policy.pt SHA-256: dfc2bd1bb34e8490730d8d84f2e3bfd90eed29045c2b519726372f92100d2ba7
config.yaml SHA-256: 9edf2219aab2ad2ff265d8da6e6c6856070ace7762218257ccebd50f89cc0563
MPC task: /home/simon-nfs/Projects/Simon/wbc_rl_mpc/go2_x5_ocs2/config/robot_lab_rear_r30o_s42_11497/task_sota.info
task_sota.info SHA-256: 8c36bf47fe59353f0984d27fcc46477ed9e3f70850f01963a6a030b201c3b938
OCS2 transport: synchronous
OCS2 command mode: full
```

本机对应源路径前缀为 `/home/simon/Projects/Simon/wbc_rl_mpc`。已完成的单条实跑 receipt
证明 `source_task_sha256 == runtime_task_sha256`，见：

```text
benchmark/results/paper_terrain_rollout_v3/20260916_000139/nominal/receipt.json
```

## 2. 已确认的最终 MuJoCo 环境

正式环境 ID：`go2-x5-mujoco-paper-rolling-v3`。

- 公共 Go2-X5 MJCF，physics dt 2.5 ms，`implicitfast`，重力 `[0, 0, -9.81]`。
- 16 m × 16 m、129 × 129 的确定性连续 heightfield，seed 20260915。
- 高度范围严格为 `[-0.10, +0.10] m`，RMS 约 0.04384 m，中心采样为 0。
- 无离散碎石；不要恢复 2026-09-15 的临时 ellipsoid rocks 预览。
- 地面摩擦 `0.8 0.02 0.01`，`condim=6`。
- 论文外观：无棋盘纹理的哑光 sage-stone 地面、低反射、蓝灰天空、柔和低角度
  主光和冷色补光。外观写入 environment manifest，但不改变碰撞高度数组。
- 当前冻结预览的 heightfield SHA-256：
  `7cf6b86449b68e77b3733f77d63a3d70aef7b452b411107233fc59c04b009754`。
- 当前本机 scene SHA-256：
  `bc704172e852b08e249c7e15a18c22efe9927c2aff11f178737c18ade514b710`。
  scene 内含绝对 mesh 路径；NFS 正式 campaign 应重新物化一次并冻结新的 scene hash，
  不要求它与本机预览 scene hash 相同，但所有分片必须消费同一份 NFS scene。

实现入口：

```text
benchmark/wbc/formal_mujoco.py
benchmark/data/run_formal_mujoco.sh
benchmark/wbc/FORMAL_MUJOCO.md
```

论文风格预览：

```text
benchmark/results/paper_terrain_rollout_v3/20260916_000139/paper_render/paper_frame_5s.png
benchmark/results/paper_terrain_rollout_v3/20260916_000139/paper_render/mujoco_tracking.mp4
```

单条 A0/B0 nominal 在该 ±10 cm 地形上已完成 935/935 个有效采样，final progress 1.0、
无跌倒、无数值故障，EE position RMSE 0.04669 m。严格 endpoint-hold gate 仍为 timeout，
所以这只证明运行链路和地形可用，不代表完整正式结果。

## 3. 录像和截图约定

- `benchmark/wbc/mujoco_video.py` 的正式 MP4 仅叠加黄色 reference EE path 和青色
  actual EE path；`target_base_pose` 固定为 `false`。
- 实时 viewer 中蓝色 follower target-base footprint 也默认关闭，仅直接调用
  `benchmark.wbc.mujoco --viewer-base-target` 时才显示。
- `--no-hud` 可生成论文截图用的干净 MP4；正式 benchmark 视频默认保留方法、场景、
  时间和 EE error 的 HUD。
- 论文渲染建议 1920×1080；完整 1088 个视频先使用默认 960×540，避免无必要放大 NFS
  占用。代表性结果可从 trace 无损重渲染成 1920×1080。
- 每个视频 artifact 必须记录 source trace hash、scene hash、分辨率、fps，以及
  `overlays.target_base_pose=false`。

## 4. 当前实现状态与必须先补的 P0/P1

已有单机 runner、原子 `formal_state.json`、resume、失败保留、独立进程组、ROS domain
隔离和录像后处理。上一轮 v1 单机 campaign 位于
`benchmark/results/formal_mujoco_rough_v1/20260915_214906/`，结果为 540/544 job 完成、
4 个 `qm_control` job 失败；该结果只用于验证调度器，不能与新 v3 环境结果混合。

正式分布式启动前仍必须实现以下内容；当前不能假装这些入口已经存在：

1. 在 `cross_method_cli.py` 增加 ours 专用的 OCS2 task 参数，并只在 `roboduet` 命令中
   传入 `task_sota.info`。
2. 在 `formal_mujoco.py` 增加对应参数，把绝对路径和 SHA-256 冻结进 campaign state，
   resume 时严格校验；receipt 必须再次满足 source/runtime hash 相等。
3. 把 `sys.executable` 显式传给 cross-method 子进程，去掉远端对本机硬编码
   `/opt/miniconda3/envs/isaacgym/bin/python` 的依赖；录像 Python 路径也必须可配置并预检。
4. 增加确定性的 task-based sharding：`task_index % shard_count == shard_index`。同一任务的
   八种方法保留在同一分片，便于完整性检查。
5. 每个节点只写自己的 `nodes/<hostname>/formal_state.json` 和 job 目录；禁止多个节点
   并发改写同一个 state 文件。
6. 增加只读 aggregator，校验 suite、scene、heightfield、task_sota、方法、场景哈希一致，
   拒绝重复/缺失 method-task-scenario 行，再生成总 `formal_results.json`。
7. 为 task override、shard 不重不漏、跨节点 manifest 一致性、失败保留和录像 overlay
   增加测试。

建议新增入口名（尚未创建）：

```text
benchmark/data/run_formal_mujoco_cluster.sh
benchmark/wbc/aggregate_formal_mujoco.py
```

## 5. 集群和 NFS

网关：

```bash
ssh simon-nfs@corelab-amd-server.lan -p 65522
```

从网关到计算节点也使用 SSH 端口 `65522`，不能使用默认 22。共有 7 台主机、8 张 GPU：

| shard | hostname | GPU | 建议首轮节点 worker |
|---:|---|---|---:|
| 0 | `simon-5090.lan` | RTX 5090 ×1 | 1 |
| 1 | `corelab-4080s.lan` | RTX 4080 Super ×2 | 1 |
| 2 | `corelab-5080-1.lan` | RTX 5080 ×1 | 1 |
| 3 | `corelab-5080-2.lan` | RTX 5080 ×1 | 1 |
| 4 | `actlab-a4.lan` | RTX 3080 10 GB ×1 | 1 |
| 5 | `actlab-a5.lan` | RTX 3080 10 GB ×1 | 1 |
| 6 | `actlab-a6.lan` | RTX 3080 10 GB ×1 | 1 |

7 个 task shard 对 68 条任务的分配应为五个 10-task shard、两个 9-task shard；每个
task 在本节点运行全部八种方法和 nominal/push。MuJoCo/ROS 部分主要受 CPU 和进程隔离
约束，不要因为 `corelab-4080s` 有两张卡就未经 smoke 直接启动两个节点级 shard。

所有节点共享：

```text
/home/simon-nfs/Projects/WBC/RoboDuet
/home/simon-nfs/Projects/Simon/wbc_rl_mpc
```

本机 `/home/simon/Projects/WBC/RoboDuet` 是该 NFS RoboDuet 的 sshfs 挂载。用户将在下个
session 前同步外层 workspace；2026-09-15 的审计中远端尚不存在完整
`Projects/Simon/wbc_rl_mpc`，且已检查的 `isaacgym`/`isaacgym22` 环境没有 Python
`mujoco`。因此同步完成不等于运行环境已经就绪。

旧 `tmp/run_cluster.sh` 仅可参考 hostname、Conda 和 node-local IsaacGym 映射；它会启动
无关训练，绝对不能直接运行。

## 6. 下一 session 的直接执行顺序

### P0：同步后逐节点预检

从网关对 7 个节点逐一检查：

- 两个 workspace 目录和冻结 suite 均可见；
- policy/config/task_sota SHA 与第 1 节一致；
- 选定 Python 能 import `mujoco`、`numpy`、`torch`；
- ROS Jazzy 和 OCS2/bridge executable 可用；
- `ffmpeg` 可用，EGL 能离屏渲染一帧；
- NFS 输出可写、剩余空间足够；
- ROS install 中不存在失效的 `/home/simon/...` build/install 前缀。若存在则在
  `/home/simon-nfs/...` 下重建，不复用失效 install。

远端环境名可参考旧脚本：`corelab-5080-1`、`corelab-4080s`、`actlab-a5`、
`actlab-a6` 通常使用 `isaacgym22`，其余通常使用 `isaacgym`；必须以实际 import
结果为准。不要用裸 `python`。

### P1：实现并验证缺失入口

完成第 4 节的 task override、shard、聚合器和测试。正式 campaign root 建议为：

```text
benchmark/results/formal_mujoco_paper_rolling_v3_distributed/<campaign_id>/
```

由 orchestrator 只物化一次 `environment/scene.xml` 和顶层 immutable manifest；节点只写
`nodes/<hostname>/`。为每节点分配不重叠的 ROS domain 区间。

### P2：分布式 smoke

先在两台节点运行 1 条相同冻结任务的互补 shard，至少包含 `roboduet`、一个纯 policy
方法和 `qm_control`，同时运行 nominal/push 并录像。必须核对：

- task_sota source/runtime SHA 相等；
- 两节点 suite、scene、heightfield 和环境 contract 一致；
- trace/receipt/MP4/PNG 均存在且有限；
- 视频无 target-base pose，地形外观正确；
- ROS domain 无串线，结束后无孤儿进程；
- 用 smoke 的实际 MP4 大小估算 1088 个视频的 NFS 总占用。

### P3：正式启动并持续监控

smoke 通过后立即在 7 个节点启动全部 shard，不再请求是否开始。监控到所有 shard 进入
terminal 状态；节点或方法失败仍保留原始日志与分母。只对明确失败项使用显式
`--rerun-failed`，不得用覆盖式重跑隐藏首轮结果。

### P4：聚合和交付

聚合器必须证明恰有 1088 个唯一 scenario key，列出所有失败、缺失、timeout、fall、
numerical fault，并输出按方法和难度的 frozen metrics。集群作业“退出 0”或视频存在都
不等于测评成功；最终总结应同时报告完成率、严格 success、失败类型、RMSE、能量、
滑移和稳定性指标。

## 7. 工作区保护

当前 RoboDuet 工作区不是干净状态。与本轮相关但尚未提交的文件包括 formal runner、
视频 renderer、MuJoCo runner、测试和文档；同时存在用户的其他修改：

```text
benchmark/wbc/test_mujoco_adapters.py
sysid/run.sh
sysid/run_policy_library_benchmark.py
benchmark/data/run_qm_control_library.py
benchmark/run.sh
```

不要覆盖 `benchmark/run.sh`，不要把无关 sysid 修改混入提交。外层
`/home/simon/Projects/Simon/wbc_rl_mpc` 是多仓库/非单一 Git 容器；提交前必须分别确认
真实 repo root 和 dirty state。

## 8. 当前验证证据

- `benchmark/wbc/test_formal_mujoco.py` 与 `test_cross_method_cli.py`：`6 passed`。
- v3 scene 已由 MuJoCo 编译；无碎石，高度范围和外观进入 environment manifest。
- `roboduet + task_sota` A0/B0 nominal：完整 935 steps、无 fall、无 numerical fault。
- 1920×1080 无 HUD 论文预览和 960×540 正式录像路径均已验证。
- 这些证据是单机单任务 smoke，不替代即将执行的 68×8×2 正式分布式结果。
