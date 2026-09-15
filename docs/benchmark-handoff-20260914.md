# Legged Manipulation Benchmark：IsaacGym 完整交付与后续 MuJoCo 交接

日期：2026-09-14，Asia/Shanghai。

主仓库：`/home/simon/Projects/WBC/RoboDuet`。

Visual WholeBody 的 2026-09-15 最新运行修复见第 28 节；其验收采用用户最新要求：能运行、目视大致跟随，不以 tracking success 为门槛。

RoboDuetRaw 的同类运行修复见第 29 节；A0/B0 底盘不动的后续修正见第 30 节，沿用上述验收标准。

**最新跟随算法见第 31 节**：按用户要求改为 EE 地面投影、footprint 偏移、
切向 yaw 与共用全向 waypoint PID；第 28–30 节的旧跟随规则作为历史保留。

部署仓库：`/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar`。

## 0. 最新执行交接（2026-09-14 16:52 CST，以本节为准）

本文第 1--12 节保留立项、合并前审查和设计依据；第 13--19 节记录实施过程；第 20 节记录最新范围裁剪。后续 session 应先读本节和第 20 节，不要把早期待办当成当前范围。

### 当前仓库与保护边界

- RoboDuet：`/home/simon/Projects/WBC/RoboDuet`，分支 `integration/legged-manip-benchmark`，HEAD `d2204dcfea03aaa1ccf1aac4acadec10ffe04cfc`。
- 当前实现未提交且必须整体保留。modified：`benchmark/README.md`、`benchmark/wbc/{cli,evaluation,scenarios,scoring,suite,test_contracts}.py`、本文档、`go1_gym/envs/config/wbc.py`、`go1_gym/envs/roboduet/{legged_robot,numerical_safety,wbc_env}.py`、`modules/{curriculum,trajectory_generator}.py`、`scripts/rl_sar_obs.py`。untracked：`benchmark/wbc/{mujoco,test_mujoco,trace,workspace}.py`、`modules/test_trajectory_generator.py`、`scripts/visualize_trajectory_bank_rerun.py`。
- `rl_sar`：分支 `feat/go2_x5_wbc_simon`，HEAD `77fbebcff5903e008a9d9663643e6d4ec2f1692f`。用户改动 `policy/go2_x5/base.yaml`、`src/rl_sar/test/gait_observation_probe.cpp`、`src/rl_sar/test/test_gait_observation.py` 必须保留。
- 本轮运行了有界 IsaacGym benchmark、rl_sar fixed-input probe 和一个 0.4 s MuJoCo scripted-IK TaskSpec 闭环。没有启动训练、Slurm 或硬件；没有执行 RC_s17/F1 learned manipulation 闭环。

### 最新范围与控制器方向

- 下一会话的唯一主交付是**完整 IsaacGym legged-manipulation benchmark 系统**；本阶段先不继续 MuJoCo。
- IsaacGym 必须在同一 TaskSpec、初始化、push 和 scorer 下支持两类上层控制器：`floating-base OCS2 MPC` 与 `IK`；两者都要能驱动并比较不同 locomotion policy。
- 不采用 RC_s17/F1 learned arm/MPC 路线。未来恢复 MuJoCo 时，直接复用当前已经能运行的 floating-MPC MuJoCo 仿真，由该 MPC 控制不同 locomotion policy。
- 当前范围不包含 multi-seed aggregate，也不包含 inference/solver latency、P95/P99、deadline miss、solver iteration 或 fallback 性能分析。
- legged-manipulation robustness 当前只保留 `push`。payload/link mass、friction、terrain、observation latency/frame drop、action delay、model mismatch、matched disturbance seed 和 nominal-vs-perturbed paired comparison 均移出当前交付。
- 不再实施 self-collision 统一统计，也不再补 MuJoCo 足端接触/reach-table diagnostics。现有 null/unavailable 语义保留即可。

### 已完成

1. Frank M12 三提交链已合入 integration 分支；WBC benchmark 已使用 `cell x bank row` TaskSpec、共享 simulator 配对、wave 调度、partial/run-state、严格 JSON、HTML 和 reference archive。
2. 修复 success/fall 聚合、active NaN/Inf、无效样本 mask、autoreset 前 terminal snapshot、完整 time-law/task hash 和逐任务失败保留。
3. raw trace 已升级为 `legged-manip-trace-v3`，保留 v1/v2 读取兼容；除原 EE/base/DOF/action/能量/接触/终止字段外，加入 coordination、Jacobian、joint-limit 和 IK 状态。腿部能量按 physics substep 积分；mixed `M` 下不可观测的 arm actual torque 保持 null。
4. A=0..5 轨迹达到 `0.25..5.0 m` XY 净位移；A=5 覆盖全 XY 方向、高曲率和逐点 terrain-relative `0..1.5 m` Z。gamma 容量为 2048，time-law 容量为 512。
5. `bounded-se3-arc-time-law-v1` 已修复 `v_max` 被归一化抵消的问题：`T` 是最短时长，SE(3) arc speed/acceleration 和 Cartesian reference acceleration 都有硬上限，长路径自动延长 duration。
6. benchmark 内部训练 episode horizon 已与 TaskSpec deadline 解耦；高级索引导致 trajectory placement 未写回的问题已修复并有回归测试。
7. nominal 初态从一个 canonical materialized env 复制到所有 task slot/candidate slice，TaskSpec 使用相同 environment-local 表示。相同任务跨 candidate 顺序、`--total_envs` 和 wave packing 保持同一 ID/hash/archive；policy 接管前 arm/dog observation/history 不一致会 fail fast。
8. P2 逐步指标已接入在线与离线 scorer：精度残差、leg torque/contact/base path、`d_lat/timing/progress/rho/base utilisation/feedforward/manipulability`、Jacobian singular value、joint-limit margin 和 IK saturation/validity；在线/离线逐字段一致。
9. `workspace-probe-v1` 定义了 `fixed_base`、`bounded_posture`、`bounded_whole_body` 三种体素 scope，并保留失败目标作为 attempted-volume 分母。当前完成的是 backend-neutral contract/aggregation，执行 runner 尚未接线；self-collision 不可观测时保持显式 unavailable。
10. MuJoCo 共享后端已新增 `benchmark/wbc/mujoco.py`：直接校验并读取相同 suite/TaskSpec/reference archive，通过 rl_sar TorchScript dog-policy 边界和确定性 DLS IK arm baseline 执行，写 trace-v3 并调用同一 scorer。receipt 保存 task/suite/reference、policy/config 和完整 MJCF resource tree 哈希。它不是 learned arm policy。
11. rl_sar no-height 87D fixed-input probe 发现 Python mirror 曾组装 89D；`scripts/rl_sar_obs.py` 已按生产 C++ 语义删除 height command/pose/pose-error 三处高度槽。修复后 2813 cases 通过，Python mirror 最大误差 `6.184e-7`。

### 当前有效 receipts

| 用途 | 路径 | 结论边界 |
| --- | --- | --- |
| 最新 A5 长程真实运行 | `benchmark/results/timing_a5_smoke_v6/20260914_141731/` | complete；A5/B0 3100 样本后 timeout，A5/B5 174 样本后 trajectory cutoff；均无 fall/fault；0/2 success 只说明旧 checkpoint 的 OOD smoke 失败 |
| 两 task / 两 wave | `benchmark/results/fairness_two_waves_v3/20260914_141438/` | 与下一行 task/spec/suite/reference/initial state 完全一致 |
| 两 task / 单 wave | `benchmark/results/fairness_one_wave_v3/20260914_141449/` | 事件完全一致；跨容量动作最大差 `6.02e-4`，保留 raw trace 作为 GPU PhysX 容差证据 |
| candidate 换序 | `benchmark/results/fairness_order_swap_v1/20260914_141531/` | 与 `fairness_pair_v2/20260914_140858/` 指纹和事件一致；当前两 candidate 是同一 bundle |
| P2 在线/离线字段核对 | `benchmark/results/p2_metrics_smoke_v1/20260914_140534/` | 21 个新增字段最大相对差 `5.96e-8`；仅验证指标口径 |
| P2 end-to-end 运动学字段 | `benchmark/results/p2_feasibility_smoke_v1/20260914_160024/` | trace-v3，online/offline 字段和事件一致；IK 明确 not-applicable |
| P2 legacy IK 字段 | `benchmark/results/p2_ik_feasibility_smoke_v1/20260914_160340/` | 20 step；online/offline 最大相对差 `9.77e-8`；只验证 IK/feasibility 记录链 |
| rl_sar 87D fixed input | `benchmark/results/rl_sar_fixed_input_87d_20260914/` | 2813 cases；C++ probe 与 Python mirror 通过；只验证 observation/gait 接口 |
| MuJoCo 同 TaskSpec smoke | `benchmark/results/mujoco_taskspec_smoke_v7/` | 20 × 20 ms，same task `timed-trajectory-2de5c9207314db07`；无 fall/fault、IK invalid=0，最终 timeout/未成功；scripted DLS baseline 实现证据 |

最新 A5 v6 的 source 正确标记 dirty；suite SHA-256 为 `9ec1277d8f1edeb15788271e8191bf41ad205f37701245d646c36e44c0a475cd`，reference archive SHA-256 为 `84b47db2e15dd9fc9b94c829d85cc6281e867e0227b182d678ad1c92be9b8eee`，raw trace SHA-256 为 `c2ea4a9728c82f56e599e4b78e8d7698dc461b76388a1b90b8a1227779686480`。在线/离线事件完全一致，共有数值字段最大相对差 `1.48e-6`。

### 已验证命令

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/simon/Projects/WBC/RoboDuet:${PYTHONPATH:-}"

pytest -q benchmark/wbc/test_mujoco.py benchmark/wbc/test_contracts.py \
  go1_gym/envs/config/test_height_reference.py \
  go1_gym/envs/config/test_numerical_safety.py \
  modules/test_trajectory_generator.py
python benchmark/ci/static_check.py
```

结果为 `41 passed`；相关 `py_compile`、static check 和 `git diff --check` 通过。MuJoCo v7 receipt 的 76 字段结果与单独离线重算完全相等。只有 IsaacGym 自带 `np.float` 弃用 warning。

### 下一步，不要重做已完成项

1. 在 IsaacGym benchmark 中建立明确的 upper-controller adapter，至少支持 `floating_base_ocs2_mpc` 和 `ik`，并让同一个上层控制器能够替换、评估多个 locomotion policy。
2. 保持现有 TaskSpec、canonical initial state、wave 调度、terminal snapshot、trace-v3 和 scorer；控制器差异只能经过 adapter，不得另建任务或评分语义。
3. 把 `workspace-probe-v1` 接到 IsaacGym，完成 fixed-base、bounded-posture、bounded-whole-body 三种 scope 的真实执行与 reachable-volume 输出；不实现 self-collision 指标。
4. 将 push 作为唯一 robustness 场景写入 TaskSpec，确保不同 locomotion policy/upper controller 接收同一确定性 push schedule；其余随机化与延迟场景不进入当前系统。
5. 取得至少两个真正不同且接口兼容的 locomotion policy，完成 OCS2/IK × loco-policy 的小型矩阵，核对任务 ID、初态、reference、push、有效样本和终止事件，再生成统一 JSON/HTML。
6. 完成单一冻结 suite 上的逐任务和总体比较。无需 multi-seed aggregate 和 solver/inference 性能统计；可以保留按冻结任务做的 paired difference/置信区间，但不得写成跨 seed 泛化。
7. 本阶段不再修改或运行 MuJoCo。现有 MuJoCo scripted-IK work 作为历史实现保留；后续单独阶段改接已能运行的 floating MPC MuJoCo 仿真。

后续 session 可直接使用：

> 在 `/home/simon/Projects/WBC/RoboDuet` 读取 `AGENTS.md`，再读 `docs/benchmark-handoff-20260914.md` 第 0 和第 20 节。保留全部 dirty 文件，不重做 Frank 合并、TaskSpec/timing、terminal/raw trace、P2 coordination/feasibility、公平性 smoke、87D probe 或现有 MuJoCo smoke。下一会话只交付完整 IsaacGym benchmark：在同一 TaskSpec/初态/push/trace/scorer 下接入 `floating-base OCS2 MPC` 与 `IK` upper-controller adapter，使它们可以控制并比较不同 locomotion policy；接通 IsaacGym workspace probe、运行最小 OCS2/IK × loco-policy 矩阵并生成统一 JSON/HTML。只保留 push robustness；不做 MuJoCo、RC_s17/F1、multi-seed aggregate、solver/inference timing、self-collision 或其他扰动。不要启动训练。

## 1. 历史用户目标与当时执行状态

用户要求审查 `frank/m12-benchmark-finish`，结合现有 benchmark，规划适合 legged manipulation 任务的合并与完善方案，并基于上述 `rl_sar` 实现 MuJoCo sim2sim benchmark，与其他 baseline 比较。

用户特别要求覆盖：成功率、完成时间、轨迹精度及方差、稳定性、能耗、关节加速度与 jerk、机械臂工作空间与运动学可行性、base 轨迹/运动范围/平滑性、base-arm 协调能力。

最新指令：**“整理交接文档，下个 session 开始执行”。**

- 本 session 只整理交接；未合并分支、未修改运行时代码、未启动仿真或训练。
- 下一 session 接续本任务时，直接按本文顺序实现并做有界验证；不必重新询问是否开始，也不必重新做一遍完整方案调研。
- 执行目标是 benchmark 集成、公共协议/评分器、MuJoCo 后端和已有 baseline 的接入。先在集成分支形成可检查的改动。
- 训练新 policy、云端/Slurm 大规模 sweep、硬件实验不属于这次交接的实施内容。本文没有安排正式比较的总预算，也没有指定最终候选 checkpoint。
- 小规模仿真验证是实现后的正常步骤；先检查资源和已有任务，使用明确的 env 数、任务数、时长及输出目录。
- 不把本文作为常驻禁止执行的标记。下一 session 应继续实施，而不是因为本 session 是文档准备就再次停在确认环节。

预期结果：**同一套任务、物理量记录和评分规则，由 IsaacGym 与 MuJoCo 两个后端执行，支持多种控制结构。**

## 2. 合并前仓库快照（历史）

以下是本次检查的快照；下一 session 先核对是否漂移。

| 项目 | 已核实状态 |
| --- | --- |
| RoboDuet 当前分支 | `v3-stage2`，比本地 `origin/v3-stage2` 超前 1 个提交 |
| RoboDuet HEAD | `41e2e596b4d76b349b8be4fb993b27f230ee26d0` |
| Frank 分支 HEAD | `079991b5249d5d59e8104f062829c1ca9299dc71` |
| 共同祖先 | `c71f829139e03d50b2ce6609f022c0cedfe8c489` |
| 祖先之后的独立提交数 | 当前分支 25，Frank 分支 3 |
| Frank 相对祖先的变更 | 14 文件，1990 additions，77 deletions |
| RoboDuet 写本文前 worktree | 干净；本文及 roadmap 入口是本 session 的文档改动 |
| rl_sar 当前分支 | `fix/stand-gait-consistency`，比对应本地 origin ref 超前 1 个提交 |
| rl_sar HEAD | `53204ad437909e3c418218f81d36088d64613d89` |
| rl_sar 已有用户改动 | `policy/go2_x5/base.yaml`，必须保留 |

Frank 的三个提交，按依赖顺序：

1. `ecb520c1869000450c3e59559092487a59ffb6c0` — `feat: impl stage2 benchmark (M12)`。
2. `d3aea8beef206d39bca04046671458308f80d374` — `feat: complete stage2 WBC benchmark`。
3. `079991b5249d5d59e8104f062829c1ca9299dc71` — `feat: parallelize WBC benchmark and add trajectory videos`。

这些是本地 refs 的检查结果；本轮没有 fetch 或声称远端不存在更新。

已经运行 `git merge-tree --write-tree --name-only HEAD frank/m12-benchmark-finish` 做合并预演。退出码 1 表示发现冲突；没有改 HEAD、index 或 worktree。冲突文件仅为：

- `go1_gym/envs/config/wbc.py`。
- `go1_gym/envs/roboduet/wbc_env.py`。

`scripts/load_policy.py` 可自动合并，但仍需语义验证。Git 对象库可能保留预演产生的临时 tree；它不是已完成的合并或可运行的源码版本。

## 3. 现有代码与可复用内容

### RoboDuet 当前分支

- `benchmark/cli.py`：`--wbc` 仍为未实现的占位入口。
- `benchmark/dog_policy/{cli,evaluation}.py`：速度、arm disturbance、body pose、gait 场景，配置恢复、兼容分组及并行执行。
- `benchmark/{candidates,metadata,compare}.py`：候选发现、git/runtime metadata、历史结果比较。
- `benchmark/reports/html.py`：已有 HTML/report 基础设施。
- `benchmark/ci/static_check.py`：主要覆盖 dog-only 报告工具；不能代替 WBC 指标和运行时测试。
- `modules/trajectory.py`：`TrajectoryBatch`，包括 `gamma_*` 与 `tl_t/tl_s/tl_sdot/T`。
- `modules/curriculum.py`：`TrajectoryBank`，当前构造器已接受 seed；路径为单文件模块，不是 `modules/trajectory/` 目录。
- `scripts/sim2sim_mujoco.py`：已有 Python sim2sim 辅助路径，主要面向 stage-1；可用于诊断，不应因此另起一套正式评分语义。

### Frank 分支新增

当前工作树还没有 `benchmark/wbc/`，检查其代码要用 `git show frank/m12-benchmark-finish:<path>`。

- `benchmark/wbc/evaluation.py`：dog/arm 配对、兼容分组、GPU 推理、WBCAccumulator、逐任务 active mask。
- `benchmark/wbc/scenarios.py`：将 `cell × bank row` 展平后按容量分 wave，检查遗漏、重复和任务顺序。
- `benchmark/wbc/suite.py`：稳定任务 ID、几何特征与 coverage、suite hash。
- `benchmark/wbc/video.py`：按任务几何特征选择代表视频，选择不依赖 policy 成绩。
- `benchmark/wbc/cli.py`：`--total_envs`、`--validate_only`、`--smoke`、bank 配置、结果与视频输出。
- 默认 6 × 6 × 8 = 288 个逻辑任务；这是覆盖规模，不是 288 次独立训练或重复试验。

### rl_sar 当前分支

- `src/rl_sar/test/mujoco_closed_loop.cpp`：已有 `rl_mujoco_eval`，复用生产 RL SDK 的 observation/model/output；支持有界场景、seed、扰动和 CSV。
- `src/rl_sar/CMakeLists.txt`：已定义 `rl_mujoco_eval` target。
- `src/rl_sar/library/core/rl_sdk/{rl_sdk.cpp,rl_sdk.hpp}`：观测、历史、模型加载、动作输出与 `SetExternalArmTarget`。
- `src/rl_sar/fsm_robot/fsm_go2_x5.hpp`：已有 `RLFSMStateOCS2Manip`，通过 OCS2 bridge 接入 base commands + arm targets。
- `src/rl_sar/library/core/ocs2_bridge/`：协议与通信。消息头注明有另一份 canonical copy，变更消息布局需要同步两份并升级版本。
- `src/rl_sar/src/rl_sim_mujoco.cpp`：交互运行时，包括 physics/control/policy 线程。

上述“已有”均指源码检查；没有在本轮编译或验证 manipulation 闭环。

## 4. 必须修复的发现与证据

本节 `benchmark/wbc/*`、相关 compare 行号指 Frank HEAD。行号只供定位，修改后按函数名追踪。

### F1：完成判据不等于任务成功

位置：`evaluation.py:439` 的 `per_env_summary()`，`:484` 的 `wbc_summary_for_indices()`。

目前主要检查 `final_progress >= success_progress` 且没有 fall/trajectory early termination。默认 progress 阈值 0.8，没有同时要求位置、姿态精度、稳定保持，也没有真正完成时间。

用 AST 提取原 Accumulator/WBCAccumulator 后在 CPU 构造输入，实际输出：

```text
completion_probe: {"completed": true, "ee_pos_rmse_m": 0.20000000298023224, "ee_rot_rmse_rad": 1.0}
```

该例 final_progress 为 0.85，无 fall/early termination。它证明判据缺项；不代表任何真实 policy 达到了这些分数。

修复：benchmark 拥有独立的 success/termination evaluator；point 与 timed trajectory 使用不同成功条件。保留旧 progress 作为诊断字段，不复用训练 curriculum success 作为正式任务成功。

### F2：相同 bank row 不保证相同任务初态与世界目标

位置：`scenarios.py:149` 的 reset/settle；`_load_paired_tasks()`；`wbc_env.py:1306` 的 `_place_and_reset_trajectories()` 和 `_default_trajectory_anchor()`。

各 candidate 先运行自身 policy settle，再以此刻 shoulder/base pose 和 reach 模型确定 anchor。任务位置因此依赖 candidate 的预热行为/配置；换顺序、分组或 wave 也可能改变随机数分配。

修复：独立生成 TaskSpec；用共同的物理初态、环境局部世界坐标和参考轨迹执行。允许不同环境有平移后的 env origin，但去除 origin 后任务必须一致。预热使用固定初始化流程，并明确 policy 接管时刻；不能在 candidate 自行移动后重采目标。

任务注入必须同步刷新当前参考、进度、episode clock、终止缓存、观测历史及相关命令状态，避免第一步读旧参考。共享 simulator 的 curriculum 更新和训练 command scheduler 不能改变正式任务。

### F3：suite hash 未覆盖完整时间律和重放条件

位置：`suite.py:52` 的 `trajectory_features()`，`:85` 的 suite hash。

content hash 仅覆盖路径 p/quaternion/s；条目虽含 duration，但缺少完整 `tl_t/tl_s/tl_sdot`、初态、anchor 和扰动。另一个 CPU 复现固定几何与总时长、改变时间律后得到：

```text
different_time_law_same_suite_sha256: True
```

修复：保存并 hash 完整轨迹、时间律、初态、场景/扰动、成功协议和资源引用；评分用的完整任务可脱离原生成器独立重放。不能用 `bank_seed != 0` 单独证明 held-out；需记录训练 bank 来源或明确无法验证训练集交集。

### F4：物理指标口径需修正

位置：`evaluation.py:322`、`:330`、`smoothness_summary()`。

- `base_lin_vel` 是 body frame；直接差分不是世界坐标物理加速度。用相同世界坐标点的速度差分，或显式加入旋转坐标系项。
- 当前 `motor_power_mean_w` 仅为 arm 关节绝对机械功率均值；不能叫 whole-body 能耗。
- 实际物理运动、joint target/action 平滑性分别记录；不要把 action 差分当 joint jerk。
- 能量应在物理子步积分；固定采样时刻、单位、坐标点、滤波与重采样方案。不要跨 reset 求导。

### F5：compare 丢失 success/fall

位置：`benchmark/compare.py` 的 `_preferred_result_rows()` 与 `_summary_rows()`。

它优先使用 `wbc_trajectories`，但该层字段为布尔值 `completed/fall`，primary metrics 读取 `completion_rate/fall_rate`。原函数 CPU 复现：

```text
compare_probe: {"completion_rate": null, "fall_rate": null, "ee_pos_rmse_m": 0.20000000298023224}
```

修复：定义逐任务结果与 aggregate schema，显式从布尔事件累计分子/分母；HTML/compare 使用相同 scorer 输出。不要将逐任务行和 cell aggregate 重复计权。区分宏平均与样本加权 pooled RMSE，明确字段名。

### F6：无效数据可能穿过 mask，metadata 与失败记录不足

位置：`evaluation.py:220` 的 `_masked()`，`cli.py:305` 以后的 JSON/metadata。

```text
masked_nan_probe: [nan]
```

原函数通过乘法屏蔽值，`NaN * 0` 仍为 NaN。缺样本时很多函数又返回 0，会看起来像完美跟踪。

修复：先区分 inactive 样本与 active 非有限状态；inactive 用 where/select 排除，active 非有限状态生成 fault 事件。缺失指标写 null 并带有效样本数；JSON 严格禁止 NaN/Inf。异常任务也落盘，保留分母和失败原因。

WBC CLI 当前未复用现有完整 metadata helper。需补齐两仓库源码、实际 resolved config、policy、robot/model/TCP、执行器、任务和原始结果的 provenance。失败时不应丢失已完成 wave 的所有结果。

### F7：共享环境兼容性不完整

位置：`evaluation.py:34` 的 `WBC_COMPAT_PATHS`，`:115` 的加载入口，`cli.py:180`、`:231`。

共享环境由组内第一个 candidate 的 cfg 创建。已有字段不足以覆盖 sim dt、decimation、action scale、PD/actuator、IK 参数、TCP、reach table 内容、所有观测语义和动态资源。

修复：分开声明“公共评估物理条件”和“candidate 控制/观测契约”。允许不同 observation layout 的方法比较，但不能直接共用错误的 observation/control adapter。无法证明共享执行等价时分组或独立进程执行，同一任务与评分规则不变。

## 5. 合并方案

从核实后的 `v3-stage2` 建集成分支，建议名 `integration/legged-manip-benchmark`。有需要可建独立 worktree；先检查名称/路径是否存在。不要覆盖当前目录或 rl_sar 的 dirty base.yaml。

保留 Frank 三提交的历史，在集成分支接入完整依赖链，再做修复。不要只 cherry-pick `079991b`。不要在当前主工作树有未保存改动时盲目 checkout/merge；本交接文档可能尚未提交，新 worktree 不会自动含它，应先从原路径读取并保留。

| 文件/功能 | 处理方式 |
| --- | --- |
| `benchmark/wbc/` | 接收框架、wave 与视频；实现 F1–F7 后才用于正式评分 |
| `benchmark/cli.py` | 接入 WBC，保留 dog-only 与 inspect/compare 入口 |
| `benchmark/reports/html.py`、`compare.py` | 接收 WBC 展示，统一 schema、聚合和协议兼容校验 |
| `scripts/load_policy.py` | 接收可选 `device` 参数，保留当前 checkpoint/layout 校验 |
| `go1_gym/envs/config/wbc.py` | 增量加入 bank seed；当前已有 9-plan 与 bypass 配置，不重复改变默认行为 |
| `go1_gym/envs/roboduet/wbc_env.py` | 接入自定义任务 hook，合并全部 reset bookkeeping，核对 fixed gait duration |
| README/roadmap | 更新实际能力；不能沿用旧文档“暂不做 WBC”的优先级 |

最重要的冲突是 trajectory reset 提取：Frank 把放置/清零搬到 `_place_and_reset_trajectories()`；当前代码 `wbc_env.py:1420` 后新增了 `traj_dlat_sq_sum`、`traj_rho_*`、`traj_base_util_*`、`traj_v_ff_sum`、`traj_v_base_sum`、power/manipulability 等清零。必须把完整集合保留下来。

运行时契约以 checkpoint + 实际代码为准：

- 当前 M12 `plan()` 为 arm 6D 加 plan 6D/9D；9D plan 顺序是 `dv(x,y,yaw), height,pitch,roll, gait_frequency,stance_width,stance_length`。
- 当前 `commands_dog` 的 body command 顺序与 plan action 顺序不同。不要按旧说明直接复制索引；检查 `dog_cmd_idx` 与 `posture_specs`。
- 前 6 个 arm actions 可能为 `ik_residual` 或 `end_to_end` 等不同解码模式，宽度相同不等于含义相同。
- 正常执行顺序为 arm observation/inference → 一次性全环境 plan → dog observation/inference → `HistoryWrapper.step(dog, physical_arm)`。保留上一动作/plan 历史语义。
- 不为了评分方便修改训练 obs、action layout、reward 或 curriculum。
- 保留项目的 DOF-before-root reset 顺序和 actor 创建期 rigid-body DR 规则。

## 6. 目标架构与数据契约

```text
固定 TaskSpec + ControllerBundle + EvaluationProtocol
                     |
          +----------+-----------+
          |                      |
    IsaacGym runner       rl_sar MuJoCo runner
          |                      |
          +----------+-----------+
                     |
             逐任务事件与原始轨迹
                     |
              同一套离线 scorer
                     |
        task results / aggregate / compare / HTML
```

允许渐进式抽取模块，先兼容已有 CLI；不要为此重写整个 dog-only benchmark。可将公共 schema、metrics、task loading 放在 `benchmark/` 中独立于 IsaacGym import 的模块，便于 CPU 重算和 MuJoCo 复用。

### TaskSpec

至少记录 task_id、schema/protocol version、task family、robot/world resource hashes、物理初态、初始参考、完整参考 p/q/t 或路径加时间律、扰动时间表、episode seed、deadline、终止与成功参数。

记录环境坐标转换规则与 TCP。公共任务生成不能读取 candidate 的在线状态/成绩。记录 generator version，但执行应能直接读取冻结产物。

### ControllerBundle

支持独立 dog/arm checkpoint 路径与 ID、单一全身 policy、脚本/IK、MPC + model/config。保存每个资源 hash、实际 observation/action layout、history/recurrent reset、control/policy period、命令与关节映射、PD/actuator 和后处理参数。

不能把“相同 logdir + 同一 ckptid 的 dog/arm”作为唯一候选形式。辨识模型必须与对应 policy 配套，不静默换成另一个已部署 policy。

### 原始记录与结果

- 最小逐步记录：simulation time、reference/actual EE pose、base pose/twist、q/dq、实际 actuator torque、目标 q/dq/torque、contacts、控制状态与终止事件。
- 根据指标需要记录物理子步数据或无损累积的能量/平方和/计数；不得只保存打印出来的均值。
- 在 autoreset 前捕获 terminal snapshot。失败、solver reject、numerical fault、初始化失败、未执行任务分别记录。
- 每个任务输出 success、时间字段、end reason、有效 exposure、各指标与原始文件引用；aggregate 可完全离线重算。
- metadata 引用 RoboDuet/rl_sar 两仓库 commit、dirty/source snapshot、policy/config/resource/task hashes、runtime/build/MuJoCo 版本、实际配置和完整调用参数。
- 视频是新一轮 replay 时应保留独立 replay 指纹/状态；不能宣称它必然是定量试验同一次 rollout。

## 7. 指标协议

| 维度 | 要实现的指标与解释 |
| --- | --- |
| 完成 | success rate、完成时间分布、timeout/fall/fault/solver failure 等原因分布 |
| 精度 | EE position/geodesic orientation RMSE、MAE、P95、peak、容差内时间比例；几何横向误差与按时间跟踪误差分开 |
| 稳定性 | 单次误差残差波动、同任务重复执行方差、不同训练 seed 方差分别统计 |
| 平滑性 | arm/legs joint acceleration 与 jerk，EE/base 线和角加速度/jerk，target/action 差分分别报告 |
| 能量 | arm/legs/whole-body 的积分绝对机械功、正机械功、均值/峰值功率、torque RMS/饱和时间比例 |
| 工作空间 | 固定 base、可变姿态、可移动 base 的 SE(3) 目标成功地图；覆盖精度、时限与姿态条件 |
| 可行性 | 关节限位裕度、碰撞、奇异性、IK/轨迹连续解可行性；区分 solver 未找到解和不可行证据 |
| Base | 世界轨迹、姿态、路径长度、漂移、区域占用、平滑性、足端滑移与支持约束 |
| 协调 | 必须移动 base 的任务收益、EE 保持补偿能力、可达范围扩展、相同任务质量下的额外运动与能量 |
| 鲁棒性/计算 | 扰动恢复、负载/地形/摩擦/延迟退化、推理/求解 P95/P99、deadline miss/fallback 次数 |

### 成功与时间

- point reaching：在 deadline 前达到位置和姿态容差并持续保持指定时间；成功判定完成时刻包含所需 hold。可另存 first entry time。
- timed trajectory：按给定时间律执行，要求预定的 tracking tube 占比、终点精度/末端 hold、路径覆盖与安全条件。不能到 80% 就成功，也不能通过事后时间对齐消除滞后。
- `position_tolerance_m`、`rotation_tolerance_rad`、`hold_time_s`、`tracking_tube_fraction`、`deadline_s` 等参数写在 protocol/profile。
- 用户尚未指定精度阈值。开发 profile 可用清楚标记的暂定值，例如 3 cm / 5 deg / hold 0.5 s，实施者应按任务尺度检验并冻结；这些不是现有训练/论文已认可的标准。
- 失败样本不进入“成功条件下完成时间均值”，但必须保留在总分母。展示 `P(success by time t)`，避免只成功少数简单任务的方法被误判为最快。
- 失败后的缺失轨迹不能静默补 0 或从任务集合消失。评分协议应明确 RMSE 的有效时窗及失败惩罚/独立失败率，防止早失败产生虚假的低误差。

### 方差、平滑性与能量

对真实参考计算残差，再统计残差方差；不要把轨迹位置变化当噪声。角误差使用 SO(3) 测地角，涉及角残差协方差时使用明确坐标系的旋转向量。

求导固定采样带宽、时间步和滤波方式，报告单位和 valid count；不得跨 reset 求导。物理高频 raw 与固定带宽比较指标可以同时保留。不同轨迹速度/时长下不要把所有 jerk 混成无条件的单一排名。

绝对机械功代理：`E_abs = integral(sum_i(abs(tau_i * dq_i)) dt)`；另可记录正机械功。没有电机损耗模型时不能称为电池耗电。MuJoCo 有非单位 transmission/gear 时不能直接把 actuator scalar force 当 joint torque，应核对映射或使用对应 joint-space actuator force。

站立操作不按接近零的移动距离算 COT；移动任务可补充 COT。能量始终与任务成功/精度一起报告，防止早失败被奖励。

### 工作空间与协调

固定 base arm workspace、允许 base 姿态变化的 workspace、允许迈步平移的任务 workspace 分开。后两者必须限定操作区域、时间、初态和目标姿态，否则访问体积可以靠漫游增大。

`rho` 是方法诊断，不是独立运动学可行性判据。主要可行性评估应使用公共机器人模型、TCP、限位和碰撞定义。Manipulability 需注明 Jacobian 坐标/平移旋转尺度，不能盲目跨机器人比较原始 determinant。

可辅助分解世界坐标 EE 线速度：

```text
v_EE = v_base + omega_base × r_base_to_EE + J_arm_linear × dq_arm
```

这只是解释量；在 EE hold 任务中抵消 base motion 是好行为。通过任务/消融比较收益，不把某个贡献比例或更大的 base 速度直接当“协调更好”。

## 8. 任务集与 baseline 设计

第一版至少覆盖：

1. 近距离 point reach/hold：arm 原本可达，测精度、稳态和无必要 base 运动。
2. 超出初始 arm workspace 的 reach：全身在指定区域内可达，需要平移或调姿。
3. 世界坐标 EE hold + base 移动/外部扰动：测补偿能力。
4. timed SE(3) tracking：多方向、曲率、速度、姿态变化率与较长路径。
5. 近限位/碰撞约束、负载与地形变化：与 nominal 分开分层报告。

后续接触操作单独扩展 force tracking、物体位姿/操作成功率。空中 EE tracking 结果不能代表已完成接触 manipulation。

评估记录保留 task family、几何/时间难度、workspace 分区和扰动强度。可复用 Frank 的 cell grid，但不能只按训练 curriculum level 组织所有跨方法结论。

首批 baseline 家族：

- 固定 base + IK：诊断 arm 自身能力。
- 固定 locomotion policy + IK + 简单 base 规划/前馈：协调基线。
- RoboDuet dog/arm pair。
- 同一 locomotion policy + RL–MPC，使用匹配的辨识模型。
- 能接入同一机器人与协议的 unified whole-body RL 或 optimization controller。

先盘点真实存在的源码、checkpoints 与部署契约，不能为凑 baseline 擅自训练。外部方法的 observation/preview/ground-truth access 要明确；缺失传感器不能偷偷补 oracle。优先统一机器人、物理条件和可用信息，各方法保留自身控制结构。

两条对比线分开：固定底层 locomotion 比上层协调；完整 controller bundle 比系统性能。站立操作、同步移动操作和接触操作有独立 eligibility，不能把不支持某任务的系统混入同一主排名而不说明。

正式统计使用共同 task × episode seed 配对，预先冻结任务权重；重复 episode 和不同训练 seed 分层。成功率给区间，连续指标可对配对差值做按任务/seed 分层的 bootstrap。历史结果仅在协议、任务、机器人和评分语义兼容时比较。

## 9. rl_sar MuJoCo 实现重点

优先扩展 `src/rl_sar/test/mujoco_closed_loop.cpp` 或从其抽出可复用 runner；复用现有 RL SDK、模型/历史和动作解码。公共 scorer 留在独立模块，MuJoCo 不重写第二套指标。

已有评估器必须去掉/显式参数化的专用假设：

- `Forward()` 要求 actor output 恰为 12D。
- `Reset()` 写死 `ObservationBuffer(...,30,"time")`，与 SDK 的配置推导可能不同。
- `use_policy_gait_commands` 默认 false，会覆盖 gait frequency、swing height、stance 和 gait duration。
- 当前 `completed = !failed` 仅表示存活到场景结束；没有 manipulation success。
- 目前 arm_motion 是脚本扰动，不是 upper arm policy/末端任务跟踪；没有直接证明已能跑 M12。

接口应支持分层、全身 joint-target 和 torque controller，在执行器层统一作用于 robot。不能强制每个 baseline 都走 RoboDuet 的 9D plan。PD、action scale、observation rate 是 controller contract，比较时明确固定/允许变化的条件。

提供两种调度语义：

1. 同步 simulation-time 调度：用于控制性能比较，可重复初态/时序；记录计算耗时。
2. 部署调度：复用 FSM/bridge/异步行为，用于 real-time deadline、通信、fallback 和实际闭环系统评估。

同步模式允许慢求解器等多久、是否模拟 deadline miss 必须写入协议；不能把暂停仿真等待求解的成绩声称为实时性能。

仿真和 wall-clock 时间戳分开。现有 bridge 的 steady-clock/staleness 语义不能直接换成 simulation time 而不调整双方。需要协议改动时同步两份消息头并升级版本。

检查 TCP offset、quaternion 顺序、关节顺序、base-origin/IMU frame、height reference、history 顺序、policy/control/physics dt、扭矩限制和 sensor latency。先固定输入验证 obs/action/decoder，再做闭环。

定量 rollout headless、有界、独立输出；代表视频单独 replay。原始轨迹采样需保证派生 position/velocity/force 处于明确时刻，遵守 MuJoCo forward/integration 顺序。

## 10. 下一 session 的执行顺序与验收

### P0：环境与候选盘点

- 核对两个仓库 branch、HEAD、dirty diff、worktree、GPU/进程状态。
- 阅读当前 `AGENTS.md`；实际 IsaacGym/MuJoCo 调试时使用对应 skill。
- 保存已有改动的可恢复快照，尤其 rl_sar `base.yaml`；使用明确 candidate 参数，避免借改默认文件切换方法。
- 在集成分支工作；列出可用 dog/arm/MPC candidate 的资源与兼容性，缺失项准确标记。

### P1：Frank 集成 + 正确性基础

- 整合三提交和环境 hook；解决 reset 与 layout 冲突。
- 完成 F1–F7 的核心修复，抽出公共 task/result schema 与基础 scorer。
- 回归 dog-only CLI、加载、报告；验证合并没有改变训练配置/布局的预期行为。
- 验收：上面的四个 CPU 反例都有明确的新预期；任务 ID/hash 覆盖时间律，success/fall 聚合正确，NaN/缺样本不伪装为好成绩。

### P2：公共协议与 IsaacGym 有界验证

- 冻结第一版 dev/smoke TaskSpec，补齐成功时间、精度/方差、arm/leg/base/EE 平滑性与能量记录。
- 使用真实 loader/HistoryWrapper 跑小规模任务；不能只用 mock/import 宣称 IsaacGym 路径验证通过。
- 检查同 candidate 单独/批量、candidate 换序、不同 wave 容量的 task/初态/参考一致性。数值结果使用声明的容差，不要求不同 GPU batching 必然位级相同。
- 验收：terminal snapshot 无 reset 污染；按任务保留失败；raw 重算与在线 summary 一致；JSON/report 可读。

### P3：MuJoCo 后端 + 首个 manipulation 闭环

- 复用 rl_sar 实现 TaskSpec 执行、controller adapter、相同 trace/result schema。
- 完成固定输入 obs/action/decoder 对齐；真实 checkpoint 未找到时先完成公共 runner/脚本 baseline，不声称 learned pair 已验证。
- 接入可用 M12 pair 或已有匹配方法，在相同 nominal task 上跑短 reach/hold/trajectory。
- 验收：两后端由同一 scorer 出结果，资源与任务指纹完整；失败/超时/fallback 可追踪；构建与运行命令可复现。

### P4：协调/工作空间套件与 baseline 对比

- 扩展第 8 节任务与实际可用 baseline，增加可行性、工作空间和计算指标。
- 先完成少量共有任务的端到端比较、eligibility 与报告，再固定正式试验矩阵和预算。
- 验收：所有 candidate 在相同物理任务/信息约束下评分，学习/优化/脚本方法均可表达，无暗中改变默认 policy/model 的行为。

实现时及时更新本文的阶段状态、实际路径、验证 receipts 和未解决项。通过必要验证后继续下一阶段；不要因完成一次 smoke 就把全套 benchmark 标为完成。

## 11. 启动检查与环境提示

只读起步命令：

```bash
git -C /home/simon/Projects/WBC/RoboDuet status --short --branch
git -C /home/simon/Projects/WBC/RoboDuet worktree list
git -C /home/simon/Projects/WBC/RoboDuet rev-parse HEAD frank/m12-benchmark-finish
git -C /home/simon/Projects/WBC/RoboDuet log --oneline --left-right HEAD...frank/m12-benchmark-finish
git -C /home/simon/Projects/WBC/RoboDuet diff --stat HEAD...frank/m12-benchmark-finish
git -C /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar status --short --branch
git -C /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar rev-parse HEAD
```

本项目 IsaacGym 环境是 `isaacgym`；不要照抄 Frank README 中的 `conda run -n roboduet`。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/simon/Projects/WBC/RoboDuet:${PYTHONPATH:-}"
```

若使用独立 worktree，PYTHONPATH 指向实际实施目录。仿真入口保持 `import isaacgym` 在 torch 前。已有 CPU 复现仅抽取指标类/函数，不涉及创建仿真；不能用它替代运行时验证。

下一 session 可直接使用的起始提示：

> 读取 `/home/simon/Projects/WBC/RoboDuet/docs/benchmark-handoff-20260914.md`，以第 16 节为当前状态继续执行。保留全部未提交改动，不要重做已经完成的 Frank 集成、terminal snapshot、raw trace、配对同步和轨迹覆盖；先修复 timing difficulty，再跑最新长程轨迹的真实 IsaacGym A=5 smoke，之后按第 16 节指标优先级继续。

## 12. 证据边界与参考

本节是合并前的历史证据边界，已被第 13--16 节取代；不要据此判断当前工作树仍未合并或未实现。

已完成：分支 diff 审查、merge-tree 冲突预演、当前 rl_sar 源码检查、四个纯 CPU 反例复现。

尚未完成：实际合并、实现修复、policy bundle 盘点、IsaacGym/MuJoCo 构建/闭环 smoke、baseline 比较、任何训练或硬件实验。本交接不提供这些结果的性能结论。

参考资料：

- 本仓库 `AGENTS.md`、`benchmark/README.md`、`docs/benchmark_roadmap.md`；部分旧说明与 M12 当前代码有差异，以实际契约为准。
- [MuJoCo simulation loop](https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html#simulation-loop)：状态、派生量与控制调用时序。
- [Whole-body end-effector pose tracking](https://arxiv.org/html/2409.16048v2)：工作空间和全身操作比较参考；该文明确操作策略本身不含 locomotion，接入时按任务能力分组。

## 13. 2026-09-14 执行进展

已建立 `integration/legged-manip-benchmark`。文档提交为 `c5da12a`，Frank 三提交链通过 merge commit `a59e73c` 接入；两个预期冲突已按当前 9D plan、固定 gait duration 和完整 trajectory/coordination accumulator reset 语义解决。

P0 已完成当前快照复核。`rl_sar` 仍在 `53204ad`，用户的 `policy/go2_x5/base.yaml` 改动未触碰。检查时 GPU 0 上另有一个 4096-env IsaacLab 训练进程，因此这里只运行 1-env、20-control-step 的有界 IsaacGym smoke。

P1 已开始，当前工作树实现了首轮正确性修复：

- 公共开发版 timed-trajectory success 协议使用 3 cm、5 deg、80% tracking tube、99% progress 和 0.5 s endpoint hold；阈值仍明确标为 development，不是论文最终标准。
- active NaN/Inf 记录为 numerical fault；inactive NaN 通过 `where` 排除；无有效样本输出 null，JSON 禁止 NaN/Inf。
- reference hash 覆盖 `gamma_*`、`tl_t/tl_s/tl_sdot`、`L/T`；任务 ID 还覆盖同步后的物理初态、environment-local anchor、deadline、扰动表和 success 协议。`bank_seed != 0` 不再自动声明 held-out。
- settle 改为候选无关的 zero-action 流程；同一任务的 root/DOF 初态按 env origin 平移后同步，目标 anchor 以 environment-local 坐标冻结；任务注入时立即刷新 t=0 reference 和相关诊断。
- compare 从逐任务 `completed`/`fall` 布尔事件计算 rate；旧的 80%-progress-only 逻辑已移除。
- body-frame base velocity 不再直接求导；arm/whole-body torque 在 mixed `M` 控制下不可观测时输出 null，避免把 arm position target 当 torque。当前功率仍明确标注为 control-step sample，物理子步能量积分留在 P2。
- 每个 wave 写 partial results/suite/run-state，metadata 保存 strict JSON、resolved config、checkpoint/config hashes、命令和两个仓库的 source snapshot；完成的 suite 另存完整 `gamma_*` 与 `tl_*` 的 backend-neutral NPZ replay archive。

CPU regression：`pytest -q benchmark/wbc/test_contracts.py` 为 5 passed；`python benchmark/ci/static_check.py` 通过。真实 loader 验证发现 `runs/2026-08-31/stage2_v3_trajtracking_refactor_131152` 的 arm checkpoint 与恢复后的 runtime layout 不兼容（checkpoint history 11859 / actor 329，runtime 3953 / 195），没有绕过。`runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018` 的 90D dog、201D arm、15D arm-action pair 通过 loader。

IsaacGym smoke receipt：`benchmark/results/integration_start_smoke_v2/20260914_115232/`。命令使用 1 env、1 task、2 settle steps、20 eval steps；`run_state.json` 为 complete，metadata 记录 clean source `7db10b1`，任务 `timed-trajectory-32408c2cf0e33d3d` 无 numerical fault，0.4 s benchmark deadline 正确记录为 timeout。完整 reference archive 为 `task_references_group_01.npz`，SHA-256 `7f5917e40b8b2e2d9a297d6e9686aeef0145077b47753ec07c6328a3a4e23c01`。它只证明当前集成路径能创建真实 simulator、加载 policy pair、执行并写严格结果；0/1 complete 在人工短 deadline 下不是 policy 质量结论。

## 14. 2026-09-14 P2 terminal/raw-trace 进展

P2 的 reset 边界与离线重算基础已继续实现：

- `BenchmarkWBCEnv` 在 `reset_idx()` 之前只读保存 terminal physical sample；普通训练 `WBCEnv` 继续使用 no-op hook，DOF-before-root reset 及其他 simulator 写入顺序未改变。
- 在线 accumulator 对终止 env 使用该快照，明确输出 `n_valid_metric_samples`、`terminal_snapshot_used` / aggregate count；timeout、trajectory early termination 和 numerical fault 不再依赖 reset 后可能已被清零的张量。
- `--record_raw_traces` 可按 policy/wave 写 backend-neutral NPZ（当前 schema 为 `legged-manip-trace-v2`，scorer 仍兼容 v1）。内容覆盖 reference/actual EE pose、base root state、q/dq、actuator command/limit、joint target、policy action、足端接触力/速度、有效 mask 与终止事件；结果行保存 archive hash 和 task index。
- `python -m benchmark.wbc.trace <archive>` 可在不创建 IsaacGym/MuJoCo 的情况下重算核心成功事件、EE accuracy、控制步功率与 EE/base/arm smoothness；raw trace 也已带上 `d_lat/timing/rho/base-feedforward/manipulability`，这些 coordination 字段的完整离线汇总仍待补齐。
- 腿部绝对/正机械能现在直接在每个 physics substep 上积分 `tau*dq*sim_dt`，逐任务保存 joule；这与控制步瞬时功率是两个独立口径。混合 `M` 控制的 arm slice 是 position target，因此 arm/whole-body power/energy 仍输出 null，未伪装成物理力矩。
- 配对检查发现原同步代码错误地用 env index 索引扁平 `dof_state`，实际只复制了少数 DOF 行；现改为 `(num_envs,num_dof,2)` view 后按 env 复制。indexed state write 后增加候选无关的 materialization step，再同步物理状态和 observation/task caches。每个 wave 在 policy 接管前显式比较 arm/dog `obs` 与 `obs_history`，不一致会 fail fast。
- nominal timed-trajectory suite 明确关闭 arm link mass/COM 与 mount asset randomization；此前这些 actor-creation DR 会让配对 candidate 在相同 action 下立刻经历不同动力学。未来 perturbed suite 必须把配对后的扰动值写入 TaskSpec，而不能重新打开独立随机采样。

回归为 benchmark contracts 加 numerical-safety 共 13 passed，静态检查通过。最新真实 receipt 为 `benchmark/results/raw_trace_smoke_v3/20260914_122018/`：1 env、1 task、2 settle steps、最多 500 control steps，任务在第 33 步发生 trajectory early termination；在线结果保留 33/33 个有效样本并标记 `terminal_snapshot_used=true`。离线重算与在线 success/end reason/counts 完全一致，accuracy/power/energy/smoothness 在相对 `1e-6` 内一致。该样本的 substep-integrated leg absolute mechanical energy 为 `119.7859 J`；raw archive SHA-256 为 `fe0a2b49a3eba85dcd030f58d967429be6ff6cbf57b081c7508dca59dbf9c067`。该 run 的 source metadata 正确标记 dirty，因此是实现验证 receipt，不是可发布性能结果。

配对公平性 receipt 为 `benchmark/results/paired_fairness_smoke_v6/20260914_122951/`，把同一 bundle 作为 `pair_A/pair_B` 放入一个 shared simulator。二者 task ID、TaskSpec hash、冻结 initial state 和 anchor 相同；500 个共同样本的 environment-local reference position 最大差 `2.98e-7 m`，quaternion/time-law 差为 0，第一步 policy action 最大差 `1.91e-6`，第一步 DOF/root-local state 最大差分别为 `3.58e-6 rad` / `1.57e-5`。两者均保留 500 样本并以 benchmark timeout 结束。该结果证明本次固定 bundle/单任务/单 wave 配对成立；尚未完成 candidate 换序、不同 wave capacity 和多任务 suite 的一致性矩阵。

## 15. 2026-09-14 连续移动轨迹覆盖

最高几何难度不再局限在约 1 m 的 XY 包围盒。生成器新增显式 `xy_displacement` 契约：A=0..5 从 `0.25 m` 线性增加到 `5.0 m`，并消除多频正弦自身的随机首尾漂移，因此 A=5 的每条轨迹都具有约 5 m 的首尾 XY 净位移。方向扩散也从 0 增加到 `pi`，最高难度在整个 XY 平面采样行进方向。为避免长程漂移把曲率稀释，另叠加随 A 增强的确定性横向波形（最高 `0.55 m`、3 cycles）；A=5 同时把原始轨迹 Z 归一到 `0..1.5 m`。bank 轨迹放置时，Z 按每个参考点 XY 位置查询地形表面并逐点相加，所以该范围是相对局部地面而非固定 world-Z；自定义 probe 仍保留完整 XYZ offset 语义。该共享 curriculum 变更同时影响后续训练轨迹库和 benchmark held-out 轨迹库，已有 checkpoint 在这些任务上的结果应明确标为长程/大高度分布外评测，不能当作同训练分布成绩。

Rerun 样例位于 `benchmark/results/trajectory_samples/grid6x6_xy5m_z1p5_highcurv_smoothz_seed12345/`，包含 36 条 6x6 网格轨迹、原始 NPZ、manifest 和 `trajectory_gallery.rrd`，并绘制每条路径的 z=0 地面参考框。高度使用一次平滑的“中位高度→最高→最低→中位高度”扫掠，避免把 1.5 m 覆盖变成高频垂直抖动。正式 bank 口径审计了 A=5 的 384 条（6 timing cells x 64）：首尾 XY 位移为 `4.99998..5.00000 m`，地面相对 Z 最低/最高约 `0..1.5 m`，8 个方向桶计数为 `[59,48,50,46,44,53,47,37]`。每条轨迹的 curvature p99 最低 `3.679 rad/m`、中位数 `7.712 rad/m`；最大 SE(3) 路径长度 `15.1439 m`、最大 gamma 点数 1515。容量从 1536 提高到 2048，benchmark loader 对旧 checkpoint 也强制该下限，固定审计余量 533 点。288-task/8-row suite 构建及新增高度、5 m 位移和高曲率 fail-fast 检查通过；生成器测试 9 passed，benchmark contract 测试 10 passed（分进程运行以满足 IsaacGym-before-torch 导入约束）。

## 16. 下一 session 最新交接

### 16.1 当前 Git/工作树

- 仓库：`/home/simon/Projects/WBC/RoboDuet`
- 分支：`integration/legged-manip-benchmark`
- HEAD：`d2204dcfea03aaa1ccf1aac4acadec10ffe04cfc`
- 当前实现尚未提交；不要 reset/checkout 覆盖。工作树包含本轮 benchmark 实现及此前同一轮未提交修改。
- 已修改：`benchmark/README.md`、`benchmark/wbc/{cli,evaluation,scenarios,suite,test_contracts}.py`、本文档、`go1_gym/envs/config/wbc.py`、`go1_gym/envs/roboduet/{legged_robot,numerical_safety,wbc_env}.py`、`modules/{curriculum,trajectory_generator}.py`。
- 新文件：`benchmark/wbc/trace.py`、`modules/test_trajectory_generator.py`、`scripts/visualize_trajectory_bank_rerun.py`。
- `git diff --stat`（不含 untracked 文件）为 13 files、859 insertions / 76 deletions；开始前重新运行 `git status --short`，因为这些都是需要保留的用户工作。

### 16.2 已完成，不要重做

1. Frank M12 benchmark 链已并入当前 integration 分支；WBC 任务按 `cell x bank row` 展平、共享 simulator 配对、wave 调度、HTML/代表视频基础已存在。
2. TaskSpec/reference hash 已覆盖完整几何、时间律、初态、anchor、deadline、扰动表和 success protocol；严格 JSON、partial/run-state、reference archive 已接通。
3. terminal autoreset 前快照、active numerical fault、inactive mask、逐任务失败保留、raw trace 和无 simulator 离线重算已实现。
4. 腿部机械能按 physics substep 积分；mixed `M` 下无法获得 arm 实际 torque 时保持 null。
5. paired DOF/root 同步、派生 cache 重建、候选无关 materialization step、policy 接管前 arm/dog observation/history 一致性检查已实现。
6. nominal benchmark 禁止候选间独立的 actor-creation link/mount DR；未来 perturbed suite 必须把成对扰动写入 TaskSpec。
7. 轨迹 A=0..5 的 XY 净位移为 `0.25..5.0 m`；A=5 全 XY 方向、含显式高曲率片段，Z 相对逐点局部地面覆盖 `0..1.5 m`。bank 路径使用 pointwise terrain height，自定义 probe 保留旧 XYZ offset。
8. `max_gamma_points` 默认和 benchmark 旧快照下限均为 2048；suite 对 A=5 的 5 m、Z 范围和 curvature p99 做 fail-fast。
9. Rerun 生成脚本已可生成 6x6 gallery、原始 NPZ、manifest 和动态 `.rrd`。

### 16.3 当前证据

- 修改轨迹前的真实 IsaacGym receipts：
  - `benchmark/results/raw_trace_smoke_v3/20260914_122018/`
  - `benchmark/results/paired_fairness_smoke_v6/20260914_122951/`
  - 它们证明 terminal/raw/paired 基础路径，但不能证明最新 5 m / 1.5 m / high-curvature 轨迹可闭环执行。
- 最新纯生成/CPU 验证：
  - `pytest -q modules/test_trajectory_generator.py`：9 passed。
  - 独立 pytest 进程运行 `benchmark/wbc/test_contracts.py`、`test_height_reference.py`、`test_numerical_safety.py`：22 passed，只有 IsaacGym `np.float` 既有弃用 warning。
  - 288-task、8 rows/cell suite 构建与 coverage validation：PASS；A=5 共 48 个任务，Z 约 `0..1.5 m`，最小 curvature p99 `4.8797 rad/m`。
  - 384 条 A=5 正式 bank 审计：XY `4.99998..5.00000 m`；方向 8-bin `[59,48,50,46,44,53,47,37]`；curvature p99 最低/中位 `3.679/7.712 rad/m`；最长 `15.1439 m`；最大 1515 点。
  - `python benchmark/ci/static_check.py`、相关 `py_compile`、`git diff --check`：PASS。
  - 最新 Rerun：`benchmark/results/trajectory_samples/grid6x6_xy5m_z1p5_highcurv_smoothz_seed12345/trajectory_gallery.rrd`。
- 证据边界：轨迹扩展后尚未运行真实 IsaacGym policy rollout，不得声称 checkpoint 能完成长程/地面到 1.5 m 任务；已有 checkpoint 对 A=5 是明确的长程/大高度 OOD 测评。

### 16.4 下一步优先级

P0，先修 timing difficulty 的真实 bug：`TimingGenerator.generate()` 当前先用 `v_max` 乘 `softplus(raw)`，随后又整体乘 `L / area`，导致 `v_max` 被完全约掉；B 轴目前主要改变频率形状，并没有兑现配置声明的速度上限/难度。需要重新定义时间律契约，记录实际 mean/P95/peak speed 与 acceleration，增加单元测试，并据此校准 5--15 m 路径的 duration/deadline，避免空间覆盖被隐式变成不合理高速任务。修改协议后更新 suite/task schema/version 与 handoff。

P1，做最新轨迹的有界真实 IsaacGym 验证：使用已验证兼容 bundle `runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018`，至少显式跑一条 A=5/B=0 和一条 A=5/B=5；检查实际 world reference 相对 terrain height、未被 2048 buffer 截断、terminal snapshot/raw offline 一致、early termination 原因、viewer/video/Rerun 可读。保持小 env，不启动训练。receipt 必须标 dirty implementation smoke，不作性能结论。

P2，补齐指标：

- EE position/orientation error 的 P95、peak、residual std/variance，以及跨任务/seed 方差。
- joint torque RMS、峰值和 saturation fraction；足端 slip/support/contact 指标接入 WBC per-task summary。
- base path length、净位移、drift/covered area，及 arm-only reach 与 whole-body completion 的 coordination benefit。
- raw trace 离线汇总完整覆盖 `d_lat/timing/rho/base utilisation/feedforward/manipulability`，在线/离线字段逐项对照。
- workspace success map/reachable volume、joint-limit/self-collision/singularity/IK feasibility 分类。
- inference/solver P95/P99、deadline miss/fallback；perturbation/load/terrain/latency robustness。
- 置信区间、paired bootstrap、multi-seed aggregate。

P3，完成公平性矩阵：同 candidate 单独/批量、candidate 换序、不同 `--total_envs`/wave capacity、多任务 suite，比较 task IDs、initial states、reference archive、首步 observation/action 和最终事件；保留声明容差，不要求不同 GPU batching 位级一致。

P4，MuJoCo 公共后端和 baseline 对比仍未实现。开始时必须读取 `mujoco-skill`，保护 `/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/policy/go2_x5/base.yaml` 的用户改动，并维持 TaskSpec/trace/scorer 同构。不要把接口、固定臂或脚本 baseline 证据升级成 moving-arm learned closed-loop 结论。

### 16.5 下个 session 可直接复制的提示

> 在 `/home/simon/Projects/WBC/RoboDuet` 读取 `AGENTS.md` 和 `docs/benchmark-handoff-20260914.md` 第 16 节。保留当前未提交工作树。先修 `TimingGenerator` 中 `v_max` 被归一化约掉的问题，为 5 m XY、0--1.5 m ground-relative、高曲率轨迹建立可解释的速度/加速度/duration 契约和测试；然后用兼容 bundle 对 A=5/B=0、A=5/B=5 做小规模真实 IsaacGym smoke，核对 reference、2048 容量、terminal/raw trace 和 early termination。通过后继续 P2 指标与公平性矩阵，不启动训练，不夸大旧 checkpoint 的 OOD 表现，并把 receipts 回写本文档。

## 17. 2026-09-14 timing 契约与 A=5 IsaacGym 验证

第 16.4 节 P0/P1 已完成，以下状态取代其中的待办描述。

`TimingGenerator` 不再把 `v_max` 乘入 profile 后又归一化掉。新 `bounded-se3-arc-time-law-v1` 契约把 `T` 定义为最短时长，将 `v_max` 和 `a_max` 分别作为 metre-equivalent SE(3) 弧长速度与弧长切向加速度硬上限；长路径自动延长 duration。`TrajectoryFactory` 在 geometry/timing 合成后还会测量真实笛卡尔 reference 加速度，高曲率导致的向心项超过 `linear_a_max` 时再次整体拉伸时间。suite/task schema 升级为 `wbc-task-spec-v2`，manifest 记录弧长速度/加速度及笛卡尔速度/加速度的 mean/P95/peak 与配置上限。

当前 B=0..5 使用 `v_max=0.25..0.75 m/s-equivalent`、`a_max=0.20..1.20 m/s^2-equivalent`、`linear_a_max=0.75..2.50 m/s^2`。固定 seed 12345 的 288-task/8-row suite 已通过完整 coverage 与上限 fail-fast：time-law 均为 401 点，gamma 最大 1234 点，未触及 512/2048 容量；总 duration 为 8.0--81.75 s。A=5/B=0 duration 为 47.00--81.75 s、笛卡尔 acceleration peak 不超过 0.7499 m/s^2；A=5/B=5 duration 为 24.89--57.60 s、peak 不超过 2.4996 m/s^2。单元/contract/reset/numerical 回归为 13 + 24 passed，静态检查、py_compile 和 `git diff --check` 通过。

实现过程中真实 smoke 发现并修复两个不能掩盖的运行时问题：benchmark 曾继承训练配置的 20 s episode timeout，会提前截断长 reference；现在 benchmark env 由 TaskSpec deadline 管理并把内部 episode horizon 设为 10000 s。其次，`traj_batch.gamma_p[env_ids]` 的高级索引产生副本，placement helper 的 XY anchor/terrain-Z 修改此前没有写回实际 batch；现已显式写回并增加回归。TaskSpec 同时把未参与 ground-relative 执行的 3D anchor 改为实际使用的 `anchor_env_local_xy_m`，任务 hash 不再包含虚假的 Z anchor。

最终真实 IsaacGym receipt：`benchmark/results/timing_a5_smoke_v5/20260914_135950/`。命令使用兼容 bundle `runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018`，2 env，A=5/B=0 与 A=5/B=5 各 1 条，2 settle steps，3100 control steps，raw trace 开启。`run_state.json` 为 complete，source 正确标记 dirty；reference archive SHA-256 为 `d11078fcdc875dc5955e278f2270fd0c733aa8b44203cacdb9060ba476d6ac68`，raw trace SHA-256 为 `04fe6906ee9b36f66d04d976c4036e342be5a54c032d2d9bb785f0541fa9ece4`。

- A=5/B=0：1168 gamma points、401 time-law points、L=11.6715 m-equivalent、duration=60.6503 s、arc peak=0.25 m/s-equivalent、linear acceleration peak=0.6615 m/s^2。3100/3100 样本有效，无 fall/fault/trajectory cutoff，62.0 s benchmark deadline 后 timeout；最终 progress 0.9887，但未满足精度/hold 成功协议。
- A=5/B=5：1234 gamma points、401 time-law points、L=12.3302 m-equivalent、duration=36.1645 s、arc peak=0.5080 m/s-equivalent、linear acceleration peak=2.4995 m/s^2。在 8.72 s、436/436 有效样本时 trajectory early termination；terminal snapshot 已使用，无 fall/fault。
- 两条实际 environment-local world reference 与冻结 archive + XY anchor 的最大差分别为 `7.98e-7 m` 和 `2.28e-7 m`。平面 terrain 上，B=0 完整经过的 reference Z 为约 `0..1.5 m`；B=5 在 early termination 前经过约 `0.752..1.500 m`。在线/离线核心事件、样本数和 terminal 标志完全一致，accuracy/power/energy 数值最大相对差 `1.21e-6`。

0/2 completion 是旧 checkpoint 在明确长程/大高度 OOD 条件下的 smoke 结果，不是候选性能比较。下一优先级仍是第 16.4 节 P2 指标补齐，然后做 P3 公平性矩阵；MuJoCo/P4 尚未开始。

P2 第一批指标随后已接入在线 accumulator 与 backend-neutral scorer：EE position/orientation error 的 P95、peak、residual std/variance；腿部 torque RMS、绝对峰值和 99%-limit saturation fraction；接触期 foot slip、contact/support/no-support fraction；base XY path length、net displacement、excess path、bounding-box area 及 signed/absolute Z drift。raw trace schema 升级为 `legged-manip-trace-v2`，新增 actuator torque limits 和 foot velocities，并继续接受 v1 archive。

验证 receipt 为 `benchmark/results/p2_metrics_smoke_v1/20260914_140534/`：1 env、A=0/B=0、100 control steps、raw trace SHA-256 `8ac982b5fdf0fa3ba4b951c374331b2cf808d389278433737c7daf4038038284`。21 个新增逐任务字段的在线/离线结果一致，最大相对差 `5.96e-8`。该实现样本中 leg torque saturation fraction 为 0.0075、contact-phase foot slip mean 为 0.08487 m/s、base XY path/net/excess 为 0.58060/0.45356/0.12705 m；这些数值只验证口径与重算，不作 policy 质量结论。

## 18. 2026-09-14 P3 公平性矩阵进展

继续检查不同 wave packing 时发现，IsaacGym 在不同 env slot 中 materialize 同一 nominal 初态会产生很小的浮点差；旧实现把每个 slot 的 settle 后状态和 anchor 直接写入 TaskSpec，导致第二个任务在“单 wave 的第二个 slot”和“两 wave 的第一个 slot”之间得到不同 task ID。这不是参考轨迹变化，但会破坏任务身份与容量无关的契约。

当前实现把 benchmark 初态明确为一个 canonical materialized physical state：从首个 active env 取得 root/DOF/派生 cache，按 environment origin 平移并复制到所有 active task slot 和 candidate slice。TaskSpec 使用同一 canonical environment-local 状态与 anchor 表示，避免 world-origin float32 减法噪声进入任务哈希。DOF state 仍先于 root state 写入 simulator；task injection 后的 candidate observation/history fail-fast 检查保留。

不同容量的真实 IsaacGym receipts：

- 两个 task 分两 wave：`benchmark/results/fairness_two_waves_v3/20260914_141438/`，`--total_envs 2`。
- 两个 task 放在一 wave：`benchmark/results/fairness_one_wave_v3/20260914_141449/`，`--total_envs 4`。

两次运行的 task ID 列表、TaskSpec hashes、suite SHA-256 `3fabba792e60c0cd14bf7c53a80403504863329d0451c77ef9b27e78477e1d36`、reference archive SHA-256 `acbb37435c0d45e0efecd59ff1ac8a12f6ba661152511411e3a0f007fda8714d` 和记录的 initial states 均完全一致。按 task ID 对齐后，两个 candidate 的 20-step timeout/fall/fault/early-termination/validity 事件完全一致；environment-local reference 最大差为 `2.02e-7 m`。GPU PhysX 的 rollout 并非跨 env count 位级确定：观测到 DOF 最大差 `4.71e-5 rad`、root-local state 最大差 `2.87e-4`、policy action 最大差 `6.02e-4`；包含接触力在内的所有浮点 trace 字段最大绝对差为 `1.94e-2`。因此正式公平性检查应对任务/参考/事件采用严格相等，对连续动力学量采用声明容差并保留 raw trace。

候选顺序交换 receipt 为 `benchmark/results/fairness_order_swap_v1/20260914_141531/`，与原顺序 `benchmark/results/fairness_pair_v2/20260914_140858/` 对比。task/suite/reference 指纹完全一致，事件完全一致；environment-local reference 最大差 `2.98e-7 m`，DOF/root-local/action 最大差分别为 `2.30e-5 rad`、`1.08e-4`、`3.56e-4`。这里两个 candidate 指向同一 policy bundle，只验证调度、切片和名称换序路径；不同真实 candidate 的换序矩阵仍需在取得第二个兼容 bundle 后执行。

最新 canonical 初态实现也已重跑完整长程 smoke：`benchmark/results/timing_a5_smoke_v6/20260914_141731/`。`run_state.json` 为 complete，suite SHA-256 为 `9ec1277d8f1edeb15788271e8191bf41ad205f37701245d646c36e44c0a475cd`，reference archive SHA-256 为 `84b47db2e15dd9fc9b94c829d85cc6281e867e0227b182d678ad1c92be9b8eee`，raw trace SHA-256 为 `c2ea4a9728c82f56e599e4b78e8d7698dc461b76388a1b90b8a1227779686480`。A=5/B=0 保留 3100/3100 样本并在 62.0 s deadline timeout，最终 progress `0.9870`；A=5/B=5 在 3.48 s、174/174 样本时 trajectory early termination。两者均无 fall/numerical fault。raw reference 与 archive time-law 插值后再加 XY anchor 的最大 environment-local position 差分别为 `6.97e-7 m` 和 `2.74e-7 m`；B=0 完整经过约 `0..1.5 m` 的 ground-relative Z。在线/离线事件完全一致，共有数值字段最大相对差 `1.48e-6`。结果仍是旧 checkpoint 的 OOD 实现验证，0/2 success 不支持性能结论。

最终本地回归为 trajectory generator `13 passed`，benchmark contract + height reference + numerical safety `24 passed`；`benchmark/ci/static_check.py`、相关 `py_compile` 和 `git diff --check` 均通过。第 17 节的 P2 单任务收据生成在 canonical 初态修复之前，但最新 A=5 v6 已同时执行 v2 trace 与新增 P2 指标路径。

下一优先级是继续第 16.4 节 P2 未完成项：coordination 字段完整离线汇总、workspace/feasibility、计算 deadline/fallback、robustness 与多 seed 统计。P3 在同一 bundle 上的单独/配对、换序和跨容量核心检查已完成；不同兼容 candidate 仍受资源可用性限制。MuJoCo/P4 尚未开始。

## 19. 2026-09-14 P2 feasibility 与 MuJoCo TaskSpec 接入

P2 coordination 已完整进入 trace-v3 和离线 scorer：`d_lat`、timing error、progress、rho/comfort utilisation、base feedforward/actual utilisation、manipulability，以及 observed Jacobian/joint-limit/IK 分类。full Jacobian 最小奇异值因为平移/旋转行单位混合只作诊断；rotational Jacobian 最小奇异值为无量纲阈值量。分类仅描述 rollout 经过的构型，不证明未访问目标存在 IK 解。self-collision 在 Isaac asset collision filter 或 MuJoCo 当前未插桩时保持 unavailable，不产生虚假零计数。

`benchmark/wbc/workspace.py` 定义 `workspace-probe-v1` 的 cell-centred grid 和三种 scope；aggregate 会将所有失败目标留在 attempted volume 分母。执行器尚未接入，因此目前没有 reachable-volume 结果。

真实 IsaacGym 实现 receipts：

- `benchmark/results/p2_feasibility_smoke_v1/20260914_160024/`：end-to-end action mode，100 step，trace-v3；online/offline 事件及共同数值字段一致，IK 状态为 not-applicable。
- `benchmark/results/p2_ik_feasibility_smoke_v1/20260914_160340/`：legacy IK action mode，20 step；online/offline 事件一致，新增数值字段最大相对差 `9.77e-8`。其中 IK saturation=1、observed feasible fraction=0.05 只说明这个短实现样本，不能作为方法结论。

MuJoCo 接入前先重建 `rl_gait_observation_probe` 和 `rl_mujoco_eval`。no-height 87D bundle 的固定输入验证最初发现 Python mirror 仍保留三个 height 槽而组成 89D；现在 command height、actual pose height、pose-error height 都按生产 C++ 的 compact layout 移除，并按 command scale width 截断。最终 receipt `benchmark/results/rl_sar_fixed_input_87d_20260914/receipt.json` 为 complete：2813 cases，clock 最大误差 `6.009e-8`、phase 最大误差 `4.987e-10`、Python mirror 最大误差 `6.184e-7`。它只证明 observation/gait 固定输入边界。

新增 `benchmark/wbc/mujoco.py` 直接读取并验证 IsaacGym suite、TaskSpec 和 reference NPZ；相同 reference 通过 rl_sar TorchScript dog policy 与 scripted DLS IK arm adapter 在 MuJoCo 执行，并生成同一 trace-v3/scorer 输出。进度使用与 RoboDuet 相同的 forward-only 0.15 m SE(3) window projection。receipt 同时保存 scene XML、完整 MJCF resource directory、policy/config、Python observation mirror 和生产 `rl_sdk.cpp` 哈希，以及 20→18 DOF adapter 明细；TaskSpec 中超出当前 18-actuator 模型的两个 DOF 显式记录为 ignored。

最新 receipt 为 `benchmark/results/mujoco_taskspec_smoke_v7/receipt.json`：任务 `timed-trajectory-2de5c9207314db07`，20 个 20 ms policy step，MuJoCo physics dt 2.5 ms。run complete、无 fall/numerical fault、IK invalid fraction=0；人工 0.4 s deadline 后 timeout，未成功。runner SHA-256 为 `9def54c50aef7abe28e1178445fc9baad1f062275040f15c21545c0a1df27e25`，trace SHA-256 为 `2d2f3dc22f12c650cd93d3f3bc4dedc491d1e7caca747706e0730cd2d4d58eae`，reference SHA-256 为 `3428af78ad04c21d65fcacc20dbdc660060e17f83b48c6cd68088a45ae98d046`，MJCF resource-tree SHA-256 为 `89e35eaea24205432df08aadaa4ac4d437ff45afb2fe9a0f2512b9a5769cea19`，dog policy SHA-256 为 `6d3af1d39a51822ab1e46b48140ab136c226204afb48303ee758ee68cc00c194`。receipt 的 76 字段结果与独立调用 trace scorer 完全相等。

这证明共享 TaskSpec、reference、MuJoCo physics、rl_sar dog inference、scripted arm controller、trace 和 scorer 的有界链路可运行。它不是最终控制架构、sim-to-real 或硬件证据；后续工作按第 0 和第 20 节的新范围执行。

## 20. 2026-09-14 最新范围裁剪与下一会话验收目标

本节取代第 16.4、18 末尾和 19 末尾的旧优先级；既有实现与 receipts 继续保留，但以下被裁剪的项目不再阻塞当前 benchmark 完成。

### 20.1 当前明确不做

- 本阶段不继续 MuJoCo，不把 MuJoCo 批处理、足端接触或 reach-table diagnostics 作为 IsaacGym benchmark 的完成条件。
- 不接 RC_s17/F1 learned arm/MPC controller。后续 MuJoCo 阶段直接使用目前已经能运行的 floating MPC MuJoCo 仿真，由同一个 MPC 控制不同 locomotion policy。
- 不做 self-collision 统一统计；已有 unavailable/null 字段可以保留，但不要求补执行证据。
- 不做 multi-seed aggregate。
- 不做 policy inference、IK/MPC solver P95/P99、deadline miss、iteration、fallback 或求解效率指标。
- robustness 只保留 push；不做 payload/link mass、friction、terrain、observation latency/frame drop、action delay、model mismatch、matched disturbance seed 或 nominal-vs-perturbed paired comparison。
- 不启动新训练；使用已有且接口兼容的 locomotion policy 与 upper-controller implementation 完成系统交付。

### 20.2 下一会话唯一主目标

交付一个可实际运行的完整 IsaacGym benchmark，能够在冻结的 TaskSpec 上选择两种 upper controller：

1. `floating_base_ocs2_mpc`：floating-base OCS2 MPC 产生 base command 与 arm target/command，再由选定 locomotion policy 执行腿部控制。
2. `ik`：使用现有 IK 路径产生 arm command，并通过同一 locomotion-policy interface 执行 base/leg command。

两种 adapter 必须复用相同的任务、初态、reference、push schedule、step ordering、terminal capture、trace-v3 和 scorer。controller 与 locomotion policy 是两个独立选择维度，结果中必须明确记录二者身份、配置和 hashes，避免把 upper-controller 改动与 loco-policy 改动混在一个 candidate 名称中。

建议的运行矩阵是：

```text
upper controller ∈ {floating_base_ocs2_mpc, ik}
loco policy      ∈ {至少两个真实不同且兼容的已有 policy}
scenario         ∈ {nominal timed trajectory, deterministic push timed trajectory}
```

不要求 multi-seed 扩展。每个矩阵单元必须使用同一冻结 task list；若 OCS2 runtime 不能在共享 GPU env pool 内逐 env 实例化，可以按 controller/loco pair 分组顺序运行，但 TaskSpec、initial state、reference archive 和 push schedule hashes 必须完全相同。

### 20.3 完整 IsaacGym benchmark 的验收条件

- CLI 能显式选择 upper-controller adapter 和 locomotion policy，不依赖含糊的 run 名推断控制结构。
- OCS2 和 IK 都有固定输入 contract test：reference/state 输入、base command、arm command、action width、frame、单位和更新频率全部 fail-fast 校验。
- 至少完成 OCS2 与 IK 各一个真实 IsaacGym bounded smoke；每个 smoke 都必须有 advancing steps、有限状态、trace-v3、exit/run-state 和 provenance receipt。
- 至少两个不同 locomotion policy 完成相同任务的最小 comparison matrix；换序后 TaskSpec/reference/push/event identity 仍成立，连续动力学使用已经声明的 PhysX 容差。
- push 是唯一 robustness variation；push 的时刻、作用 body、方向、幅值、持续时间和 frame 写入 TaskSpec/hash，并在 raw trace 中记录实际事件。
- `workspace-probe-v1` 在 IsaacGym 中可执行 fixed-base、bounded-posture 和 bounded-whole-body 三种 scope，失败目标保留在分母，并输出 reachable-volume map/summary。self-collision 不在验收范围。
- results 包含逐任务结果、controller × loco-policy aggregate、nominal/push 场景标识、raw-trace 引用、严格 JSON 和 HTML；失败任务不得被过滤。
- 在线与离线 scorer 的共同字段、事件和样本计数一致；成功、fall、timeout、trajectory cutoff 和 numerical fault 继续分别报告。
- 回归测试、static check、`py_compile` 和 `git diff --check` 通过；交接文档写入最终命令、版本、hash、artifact 路径和证据边界。

### 20.4 后续 MuJoCo 阶段

IsaacGym 完整交付之后，再单独接 MuJoCo。届时不走 RC_s17/F1 路线，也不重新设计 MPC：直接复用已能运行的 floating MPC MuJoCo 仿真，将 locomotion policy 做成可替换 adapter，并继续复用 TaskSpec、trace-v3 和 scorer。当前 `benchmark/wbc/mujoco.py` 的 scripted DLS smoke 作为公共数据链路参考保留，不是后续最终控制架构。

## 21. 2026-09-14 IsaacGym system matrix 实施与审计结果

已新增显式 `--system_matrix` 路径，将 upper controller 与 locomotion policy
分成两个独立维度。`floating_base_ocs2_mpc` 使用现有 native C++ OCS2
SQP/MRT、ROS2 target relay 与 ZMQ wire contract；`ik` 使用
`WBCEnv._solve_arm_dls_ik_step` 和既有 base staging。两者都经过同一个
`HistoryWrapper`、dog-policy loader、TaskSpec、terminal snapshot、trace-v3
和 scorer。外部 physical base/posture command 统一由
`WBCEnv.apply_external_upper_commands()` 做 shape/finite/range/rate/smoothing
校验。OCS2 state/cmd 固定为 224/116 byte，state 使用 env-local position、
xyzw→wxyz quaternion、body-frame velocity 和 12+6 DOF；测试覆盖 wire
宽度、header、frame conversion、输出宽度/finite/mixed-command fail-fast，IK
另校验 `ik_residual` action mode 与 task count。

系统矩阵 receipt 为
`benchmark/results/system_matrix_v2/20260914_172059/`。实际命令使用
`--upper-controllers floating_base_ocs2_mpc ik`、两个 `--loco-logdirs`
（e2e、waypoint）、`--scenarios nominal push`、`--cells 0,0`、每单元
1 task、20 control steps。`run_state.json` 为 complete，8/8 matrix cells；
每个 task 都有 20/20 finite metric samples、无 fall/numerical fault/trajectory
cutoff，并按 0.4 s deadline 独立记录 timeout。所有失败 task 均保留在
results；严格 JSON、controller×loco aggregate、逐 task、raw-trace 引用和
HTML 均已生成。核心 artifact SHA-256：

- `results.json`: `d35a2b856dd1c8e69314eea9c81a6cd90654b4f9b1578bc3e285f763c8749e49`
- `trajectory_suite.json`: `4bcaab7a43ec0d1c2340771d2a3c6f5275e6557a85cb6612c5a9e6b04683bc85`
- `metadata.json`: `077aea27d64e9d186e73891d223d2c0b9eb86a67216c2b95e6e71d9ce5899ba3`

两个 locomotion checkpoint 确实不同：e2e dog SHA-256 为
`51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670`，
waypoint dog SHA-256 为
`a0cda68818c792b14fc6d140c8ed5021f2bb32fb4c5ad2ba1aa7828f9db67d9b`；
对应 parameters SHA-256 分别为
`6fc10bd18d37b00355b3a4c0cd4b4ffb55d91c39a724b2affbb28c9321c3e7bb`
和 `8c831a4962b2aa867ea96ab3b9ffc11562f0951eaa65349042df870ecc8b7bfa`。
nominal TaskSpec SHA-256 为
`236fa4afd323a2e21c5004d82758e2d653d1454fea3a097d89bc9f42b55d3810`；
push TaskSpec SHA-256 为
`ad7d145635b36ad0220401617d3e30c96761c266ec8dd033cff7fb609429ac2e`。
二者 reference content SHA-256 都为
`eb78b19852607c1a0f95e517bef8d443f368160e1865d4de04beb7474ff9be33`，
差异只来自显式 disturbance schedule。push 为 t=0.10 s 开始、0.10 s
持续、base COM、environment-world +X、80 N；每条 push raw trace 实际记录
5 个 20 ms active samples。

独立 `benchmark.wbc.trace` 重评分已写入同目录 `rescored.json`。8 个 task
的 success/fall/timeout/trajectory-cutoff/numerical-fault/end-reason 和样本数
与在线结果完全一致；456 个共同数值字段最大绝对差
`1.52587890625e-05`，最大相对差 `8.480402759870448e-06`。这是 float32
online reduction 与 NumPy offline reduction 的量化差，不是事件差异。

`workspace-probe-v1` 已接入相同 controller/dog step。receipt 为
`benchmark/results/system_workspace_smoke_v1/20260914_172914/`；包含 OCS2
和 IK、e2e locomotion policy、fixed-base/bounded-posture/
bounded-whole-body 三种 scope、每 scope 一个 0.1 m cell-centred voxel 和
20 control steps。`run_state.json` complete，`workspace_results.json`
SHA-256 为
`ae45bbb5cb4de49560bb6b4b30205088c94e929ea681a32c1565b07f554f22f9`。
六个目标都保留在各自 `0.0010000000000000002 m^3` attempted volume
分母，短 smoke 均 timeout、reachable volume 为 0；fixed-base 两个目标
实测最大 XY displacement 都为 0。这只证明三种 scope 和失败分母可执行，
不是 workspace 性能结论。fixed-base 通过每个 physics decimation 前恢复
post-settle 六自由度 root state 实现；bounded-posture 将 planar/yaw command
锁零并保留有界 height/pitch/roll；bounded-whole-body 保留全部有界 command。

native OCS2 运行前，在
`/home/simon/Projects/Simon/wbc_rl_mpc/ros2_ws` 仅重建了 build/install
产物：`ocs2_pinocchio_interface`、`ocs2_mobile_manipulator`、
`ocs2_mobile_manipulator_ros`、`go2_x5_ocs2`、`go2_x5_ocs2_bridge` 共 5
packages 通过。因系统 Pinocchio 4.1 不兼容旧 self-collision helper，runner
只在 receipt runtime 目录复制 task 并关闭明确不在本轮范围的
self-collision term；原 task SHA-256
`8fe707b12fb03fa96d2c2cc9ca586823988a74af27cbf91b34c823d876f09b96`，
runtime task SHA-256
`b5e698d92a48394a65255854d3bb7f6158f2195910729ab17d53396ea6407415`。
未修改外部 source。每个 bounded run 结束后 target relay/native bridge 均已
清理，无残留 benchmark OCS2 进程。

### 21.1 换序审计中尚未满足的条件

两个真实 policy 的反序 receipt 为
`benchmark/results/system_matrix_order_swap_v1/20260914_173025/`。
8/8 TaskSpec/content/reference identity、push samples 和所有离散事件一致；
IK 的 reference、EE/root/DOF/action trace 逐元素完全一致。但 native OCS2
的连续动力学明显超出此前 PhysX tolerance。检查发现 adapter 可能消费
异步 publisher 中上一 observation 的 command，现已在每步发送 state 前
drain queued command，并只接收 `policy_time` 严格递增的 fresh full-mode
command。

freshness 修复后的两次 OCS2-only 正序/反序 receipts 分别为
`benchmark/results/ocs2_freshness_order_a_v1/20260914_173326/` 和
`benchmark/results/ocs2_freshness_order_b_v1/20260914_173414/`。4/4
TaskSpec/reference/push/event identity 仍严格一致，但 continuous trace 仍未
达到 PhysX tolerance；例如 e2e nominal 的 base-root 最大绝对差约 0.128，
policy-action 最大差约 1.893。原因是 native SQP/MRT 的 observation/policy
time 和 policy update 由独立进程 wall clock 驱动；两次独立 launch 即使
任务相同也不形成确定性同步 controller sequence。因此第 20.3 的“换序后
连续动力学满足声明 PhysX 容差”仍是唯一明确未通过项，不能把当前结果称为
完整验收通过。下一步应先给 native bridge 增加 simulator-time synchronous
request/response contract（或等价的确定性 policy-update barrier），再只重跑
两种 policy 的 OCS2 正序/反序矩阵；无需重新训练、扩 robustness 或启动
MuJoCo。

最终本地回归：benchmark/trajectory tests `33 passed`；
`benchmark/ci/static_check.py`、相关 `py_compile`、`git diff --check` 全部通过。
本轮没有启动训练，也没有继续 MuJoCo、RC_s17/F1、multi-seed、solver timing
或 self-collision 统计。

## 22. 2026-09-14 native OCS2 simulator-time 同步闭环

第 21.1 节唯一未通过的换序条件现已解决。benchmark 的
`floating_base_ocs2_mpc` 不再通过两个独立 ROS publisher/subscriber 的 wall
clock 推进 solver。新增 benchmark-only native C++ runner
`/home/simon/Projects/Simon/wbc_rl_mpc/go2_x5_ocs2_bridge/tools/wbc_benchmark_sync.cpp`，
在同一进程内持有 `SqpMpc`、`MPC_MRT_Interface` 和 `WbcBridgeCore`。Python
每个 IsaacGym control step 通过单个 blocking ZMQ REQ/REP frame 同时发送
224-byte state、7D xyz+xyzw target、sequence 和由 sequence 推导的 controller
time；C++ 完成一次 `advanceMpc()`、`updatePolicy()` 和 policy evaluation 后才
回复 116-byte command。每个 task reset 将 sequence 归零，C++ 在 `seq==0`
时同步重置 MPC node 和 bridge state，避免 nominal/push 或不同 task 之间继承
solver warm start。adapter 要求 reply sequence 严格相等、
`policy_time == sequence * 0.02 + 0.01`、full mode 且 solver valid，否则直接
失败。runtime task 将 DDP/SQP `nThreads` 都固定为 1；self-collision 仍只在
receipt task copy 中关闭，范围没有扩展。

external bridge package 已新增上述 runner、CMake target 和 `ocs2_sqp`
dependency，并在
`/home/simon/Projects/Simon/wbc_rl_mpc/ros2_ws` 重建
`go2_x5_ocs2_bridge`。runner source SHA-256 为
`391b39056b224ba32fead4925698b1c8ac6166fe303dca5f70088403b4921170`，
installed executable SHA-256 为
`508dadf78c871366141e2d8063773561d6b0e45474a0b38d2f40cb66e107a1c0`；
source task、runtime task 和 URDF SHA-256 分别为
`8fe707b12fb03fa96d2c2cc9ca586823988a74af27cbf91b34c823d876f09b96`、
`ea209624e545c7be7b764f7386a0af304c6f1fcef4cb634f09ba0136cb0948bb` 和
`ba20ded72f49ff90ff1ac02d00ac23bc4f473b3fe6e2c71029597c423321d51d`。
这些资源和 adapter source/executable 都进入 controller provenance。

首次 direct smoke
`benchmark/results/ocs2_direct_sync_smoke_v1/20260914_181709/` 在任何
IsaacGym evaluation step 前以 C++ `-11` 失败，`run_state.json` 正确保留
failed。原因是直接构造的 SQP solver 尚未安装 interface reference manager；
补上 `setReferenceManager(interface.getReferenceManagerPtr())` 后，3-step receipt
`benchmark/results/ocs2_direct_sync_smoke_v2/20260914_181815/` complete，1/1
cell、3/3 finite samples，runner 正常清理。随后两个独立 20-step e2e nominal
receipts
`benchmark/results/ocs2_direct_repeat_a_v1/20260914_181835/` 和
`benchmark/results/ocs2_direct_repeat_b_v1/20260914_181846/` 的所有 raw-trace
数值与离散字段逐元素完全一致。

最终两策略 OCS2 正序/反序 receipts 为：

- `benchmark/results/ocs2_direct_order_a_v2/20260914_182350/`：e2e 后 waypoint；
- `benchmark/results/ocs2_direct_order_b_v2/20260914_182413/`：waypoint 后 e2e。

两次都为 complete、4/4 cells，覆盖两个真实 locomotion policy 和
nominal/push，每 cell 1 task、20 control steps。按
controller/policy/scenario 对齐后，4/4 suite records（TaskSpec、initial state、
reference archive、push schedule）完全相等；nominal/push TaskSpec SHA-256
仍分别为
`236fa4afd323a2e21c5004d82758e2d653d1454fea3a097d89bc9f42b55d3810`
和 `ad7d145635b36ad0220401617d3e30c96761c266ec8dd033cff7fb609429ac2e`。
四对 raw trace 的全部数值和离散字段逐元素完全一致，连续动力学最大绝对差
为 0，严格强于第 17 节声明的 GPU PhysX tolerance；无 benchmark OCS2
残留进程。两次 receipt 核心 SHA-256 分别为：

- order A `results.json`:
  `60329117dfbce8f87adb6d35aac852e3ce876269cb61f48cc23d4f8cf0c3bb0f`；
  `trajectory_suite.json`:
  `476e190bbd824fe34738a30674190c5afe485c74acad2236aba68a9154421547`；
  `metadata.json`:
  `de6713bc8e9d6ddd5783939535096c7f5e6d02d43203d224383e9b96077f2b9d`。
- order B `results.json`:
  `66699cf08f4c7321a9b1a6dce51c0ad84ad2ebf407a49f193f0a27336d03581a`；
  `trajectory_suite.json`:
  `c2b82b0da569e14d03a8ea39e8fe387a58d06c5303a455f4ff3bf01c296daa2a`；
  `metadata.json`:
  `b51aa78926a4c39a119464c9d4db9266df7797322a364f3ee1b3b06a212aec26`。

两个目录均新增独立 `rescored.json`。每次 4 个 task 的 success/fall/timeout/
trajectory-cutoff/numerical-fault/end-reason 和样本数与在线结果完全一致；
各 224 个共同数值字段最大绝对差 `7.62939453125e-06`、最大相对差
`5.390138203143529e-07`。因此第 20.3 节的 IsaacGym system-matrix 换序条件
现已通过。该结论仍是固定 seed、单个短任务 cell 的实现验收，不构成
multi-seed、长程性能、solver timing、self-collision、MuJoCo 或硬件证据；
本轮没有训练。最终本地回归为 benchmark/trajectory `34 passed`，external
bridge conversion test `1/1 passed`；`benchmark/ci/static_check.py`、相关
`py_compile` 和 `git diff --check` 全部通过。

## 23. 2026-09-14 IsaacGym 参考/执行轨迹实时可视化

system-matrix CLI 新增 `--visualize-trajectories`。启用后必须打开 IsaacGym
viewer（与 `--headless` 同时使用会 fail fast），viewer env 0 中用橙色绘制完整
reference path、绿色绘制实际 end-effector 历史。现有 cyan target、preview
pose/orientation axes 和红色 tracking-error segment 保留。实际路径在每个
TaskSpec 开始前清空，不跨 wave、nominal/push 或其他 task；长轨迹显示最多均匀
抽样 512 个点，内部仍按 control/render frame 累计。开关、两种 RGB 颜色和
viewer env index 写入 `metadata.json`。

真实 viewer smoke 使用 Xvfb（当前 agent shell 没有桌面 `DISPLAY` 授权）运行：
`benchmark/results/trajectory_viewer_xvfb_smoke_v1/20260914_204134/`。
`run_state.json` complete，IK + e2e locomotion、nominal、A=0/B=0、1 task、
5/5 control steps；真实 `gym.add_lines()` overlay 路径无异常。桌面终端应直接
省略 `--headless` 并添加 `--visualize-trajectories`，无需 Xvfb。此次 smoke
验证 viewer 绘制链路可执行，不是 tracking 性能证据。receipt 的
`results.json`、`trajectory_suite.json`、`metadata.json` SHA-256 分别为
`eab43498acbb04f57beababa36c247e896f6ad8bfe3ff2276a95e3a207917f15`、
`b81280e8493c55b788daf5a9421ff55439cdb5ebefe349f531bd55b833f35d11`、
`6606968734e51b422403421964dd56d58d72756ad8406240d61a32b28643ce9f`。
最终 benchmark/trajectory 回归 `36 passed`；static check、相关
`py_compile` 和 `git diff --check` 通过。

## 24. 2026-09-14 OCS2 整段轨迹输入与 TCP 起点修复

第 22 节的同步 runner 虽然每步严格 request/reply，但旧 v2 contract 每步只
发送当前 7D pose，并在 C++ 中把它包装成恒定 1 s target；这不是完整轨迹
MPC。现在 `wbc-upper-controller-v3` 在每个 task 的 sequence 0 一次发送完整
时间律 `tl_t` 以及对应的 xyz+xyzw SE(3) knots，后续请求只发送当前 state。
C++ runner 要求 sequence 0 至少 2 个、最多 4096 个严格递增且 finite 的
knots；非零 sequence 若携带 reference 会直接失败。MPC 的
`TargetTrajectories` 在 task reset 时只构造一次，之后按 simulator time 滚动。

同时修复了三个会放大 TCP 误差的问题：system matrix 使用 OCS2 task 自身的
安全 arm home `[0, 0.9, 0.9, 0, 0, 0]`，避免 joint2/joint3 从零下限屏障
启动；TaskSpec 升为 v3，记录完整 XYZ translation 和 quaternion left
multiplier，使冻结轨迹的 t=0 pose 与 post-settle TCP pose 对齐；原生 bridge
按 IsaacGym `base` root origin 解释 state，并使用 URDF 中 base 到
x5_base_link 的 `[0.05, 0, 0.10] m`，不再套用硬件 IMU 到 mount 偏移。
runtime OCS2 task 另外把 `armNominalPosture.weight` 从 1.0 设为 0.0，使 TCP
tracking 不再依赖实际 locomotion policy 精确实现理想 floating-base 位移，并
把 running/final SE(3) 权重从 position/orientation `10/5` 提高为 `50/25`；
所有修改都位于 receipt 的 immutable task copy 并由 hash 记录。

同一 e2e locomotion policy、cell `(0,0)` 的 20-step nominal 诊断中，旧逐点
contract 的 EE position RMSE 为 `0.31709 m`，首帧误差 `0.23444 m`。整段输入
但尚未修正初态时 RMSE 为 `0.30076 m`；安全 home 后为 `0.23394 m`。完整
SE(3) 起点对齐后首帧位置/姿态误差降至约 `0.009 m / 0.038 rad`。最终配置的
4 s receipt 为
`benchmark/results/ocs2_full_reference_tracking_weight_4s_v1/20260914_210525/`，
EE position RMSE/peak 为 `0.08040/0.11319 m`，orientation RMSE/peak 为
`0.34014/0.51987 rad`，无 fall 或 numerical fault。

完整路径 receipt 为
`benchmark/results/ocs2_full_reference_complete_path_v1/20260914_210621/`。
该任务 reference duration 为 `17.690706 s`，实际完成 910 个同步 control
steps、最后 simulator reference time `18.200226 s`，没有协议中断、fall 或
numerical fault。这证明整段 reference 在超过 OCS2 1 s horizon 后仍持续被
MPC 使用。性能尚未通过严格 tracking gate：position RMSE/peak 为
`0.11928/0.23220 m`，orientation RMSE/peak 为 `0.43921/1.05044 rad`，
tracking-tube fraction 为 `0.00110`，最终为 timeout/unsuccessful。当前结果
支持“整段轨迹输入和起点连续性已修复”，不支持“OCS2 加现有 locomotion
policy 已经达到轨迹跟踪验收阈值”；剩余主要问题是 ideal floating-base MPC
与实际 learned locomotion response 的动态失配，需要单独做 matched response
model 或 closed-loop command-dynamics 校准。

完整路径 artifact SHA-256：

- `results.json`: `f72a8805cc32beccd03a6103e6cf305fa6e98e5f5651c6f9a38623f4c91027f1`
- `metadata.json`: `7514f6939f421cea766bca2377ef63eb72abee4c60e5ef2de7aa04cc5c5b558c`
- `trajectory_suite.json`: `901d982ba19f04ac3762b83f327ed9e9d987191f2d69f0ae9ac5384e7d2242a6`
- raw trace: `bdf3937669828e732e645200285ad0e768c6e4007fa924180872ff6960061e9e`
- native runner source: `f843ba6c420ed33f7b6d5baaa6a354a2a8191578fe65bbd492ba2b84d74b5453`
- installed runner executable: `255f4cf12b0d8ca69f9cc6e7187561fd6e6d6a74825f09e4f431679ca37b9405`

最终 v3 换序 receipts 为
`benchmark/results/ocs2_full_reference_order_a_v1/20260914_211047/` 和
`benchmark/results/ocs2_full_reference_order_b_v1/20260914_211117/`。两次均
complete，分别按 e2e→waypoint 与 waypoint→e2e 执行；两个 policy 的
TaskSpec SHA-256 都为
`6fd956d39b4add32a06ee5d4abaa2e57cd25aa67aad8acaed3502ae9bd700e03`。
按 policy 对齐后，两对 raw trace 的所有 numeric 和 discrete arrays 均逐元素
完全一致，最大绝对差为 0，且无残留 `wbc_benchmark_sync` 进程。最终回归为
benchmark/trajectory/MuJoCo contract `36 passed`，external bridge conversion
test `1/1 passed`；static check、相关 `py_compile` 和 `git diff --check` 通过。

本轮没有启动训练。

## 25. 2026-09-14 OCS2 独立进程异步可视化路径

system-matrix 新增 `--ocs2-transport {synchronous,async}`，默认仍为
`synchronous`，用于可复现的定量 receipt。`async` 启动独立
`wbc_benchmark_async` C++ 进程；IsaacGym 以 state PUB/bind、command
SUB/connect 与它通信，MPC 侧采用镜像 socket，并与 `wbc_rl_mpc` 一样在两侧
使用 HWM 1 / `CONFLATE=1` 的 latest-state/latest-command 语义。

每个任务使用 32-bit task epoch 加 32-bit step 组成 wire sequence。step 0
重复发送整段带时间戳 SE(3) reference，直到 MPC 完成一次初始化握手；稳态只
发送 state，IsaacGym 按 50 Hz 墙钟节拍推进，并在 SQP 没有及时更新时复用同一
任务内最近的有效 command。超过 `--ocs2-command-timeout-s`（默认 0.5 s）
没有新 command 会直接失败。metadata 的 controller provenance 记录 received、
reused、最大滞后步数和最大 solve time，因此异步运行不会被误当成逐步严格同步
证据。

外部 `go2_x5_ocs2_bridge` 已加入并安装 `wbc_benchmark_async` target。CPU
contract 回归为 `21 passed`；独立协议 smoke 验证了完整轨迹握手和典型 1-step
command lag。真实 IsaacGym 4-step headless smoke 位于
`benchmark/results/ocs2_async_smoke_v2/20260914_212906/`，状态 complete，
1/1 cell、4/4 finite samples，最大 command lag 为 1 step，最大单次 MPC
solve time 为 `6.235 ms`，runner 在结束后无
残留进程。该短 smoke 只证明进程、协议和闭环接线，不证明轨迹跟踪性能或 GUI
帧率。

## 26. 2026-09-15 MuJoCo native OCS2 TaskSpec 闭环

`benchmark/wbc/mujoco.py` 新增
`--upper-controller floating_base_ocs2_mpc`，复用第 25 节的独立 C++ MPC 和
ZMQ transport。system-matrix v3 suite wrapper 现在可直接读取；重放时会校验
suite/TaskSpec/reference hashes，并应用完整 XYZ anchor 与 quaternion left
multiplier。MPC 在 task 开始时接收由 time-law knots 插值得到的完整 401-point
xyz+xyzw reference，后续每个 20 ms step 只接收 MuJoCo 的 base、12 腿关节和
6 臂关节状态。MPC base velocity/posture command 进入 rl_sar dog observation，
arm absolute-q command 进入 X5 PD target。可选 `--viewer` 绘制橙色 reference、
绿色 executed TCP path 和黄色当前 target。

验证使用 `smooth_S_s11_selected_20260914` rl_sar TorchScript dog bundle、同一
`timed-trajectory-ac28cbf79041b70f` 和 `scene.xml`。CPU 回归 `25 passed`；
Xvfb viewer 3-step smoke 与 100-step/2 s headless smoke 均完成。完整 receipt
位于 `benchmark/results/mujoco_ocs2_async_full_v2/`：910/910 control steps、
无 fall/numerical fault，MPC 初始化 1 次，最大 solve time `5.949 ms`、最大
command lag 1 step。最终 progress `0.95962`，position RMSE/peak
`0.14544/0.23289 m`，orientation RMSE/peak `0.41349/0.79610 rad`，tracking
tube fraction `0.00110`，最终 timeout/unsuccessful。因此当前证据证明 MuJoCo
native OCS2 闭环和完整轨迹数据链路可运行，但跟踪性能仍未通过验收阈值。

## 27. 2026-09-15 MPC pose-only 完整轨迹模式

native sync/async runners 现在接收显式 command mode 参数；Python CLI 在
IsaacGym system matrix 和 MuJoCo runner 中均新增
`--ocs2-command-mode {full,pose_only}`。默认 `full` 保持已有 benchmark 行为。
`pose_only` 严格复用 bridge 的 `CMD_MODE_POSE_ONLY=1` 合同：MPC 仍优化并接收
完整 SE(3) reference，执行侧将 `vx/vy` 置零，但保留 yaw-rate、base
height/pitch/roll 和 X5 absolute-q targets。transport 同时校验 reply mode 为 1
且 planar velocity 为零，避免配置与实际 runner 不一致。

MuJoCo 完整 pose-only receipt 位于
`benchmark/results/mujoco_ocs2_pose_only_full_v1/`。使用与第 26 节相同的
401-knot reference 和 rl_sar dog bundle，共完成 910/910 control steps；无
fall/numerical fault，`v_ff_xy_mean=0`，MPC 最大 solve time `6.140 ms`、最大
command lag 1 step。position RMSE/peak 为 `0.13296/0.24468 m`，orientation
RMSE/peak 为 `0.31767/0.74200 rad`，最终 progress `0.89614`、tracking-tube
fraction `0.00110`，最终 timeout/unsuccessful。尽管主动平移命令为零，接触和
姿态耦合仍造成 `0.41449 m` base XY 净漂移；pose-only 不是 fixed-base 物理
约束。该结果证明 pose-only 数据链路可运行，不证明跟踪性能通过验收。

## 28. 2026-09-15 Visual WholeBody 接入冻结轨迹库

新增 `benchmark/wbc/visual_mujoco.py` 和
`benchmark/data/run_visual_library.py`。恢复原生播放入口的当拍动作，接通
已有 `EEBaseFollower`、步态观测和工作空间投影；底盘会主动随 EE 目标移动。
直接使用 `benchmark/data/frozen_trajectory_library/suite` 中已冻结的 TaskSpec，
参考、时间律、初态和评分保持原定义，控制器投影目标单独记录。

从 `/home/simon/Projects/Simon/wbc_rl_mpc` 运行：

```bash
./run_visual_wholebody_library.sh --viewer --cell 0 0
./run_visual_wholebody_library.sh --viewer --trajectory random-line-000
./run_visual_wholebody_library.sh --list
```

全部 68 条通过每条 2 步短测；5 条完整时长尝试中，4 条跑完整段无跌倒，
`random-circle-009` 在后段跌倒。用户本次明确不要求 tracking 达标，
因此保留诊断评分，不继续为成功率调参。27 项相关测试通过，Xvfb viewer
10 步测试和直线 MP4 回放已生成。

这里是 Visual low-level policy + 原仓库启发式 follower + IK，未加载图像
high-level policy。具体命令、限制和产物见
[Visual WholeBody 运行说明](visual-wholebody-library-20260915.md)。

## 29. 2026-09-15 RoboDuetRaw 冻结轨迹库运行修复

保留 Raw 原始双策略和 learned body posture plan，给 `plan_vel=False`
checkpoint 增加明确标注的启发式底盘跟随辅助。近处目标交给机械臂处理，
目标超出近处工作空间后底盘开始跟随；策略输入投影到该 checkpoint 的训练
范围内，冻结参考和初态保持不变。没有用 IK 替换 arm actor，没有重训。

从 `/home/simon/Projects/Simon/wbc_rl_mpc` 运行：

```bash
./run_roboduet_raw_library.sh --viewer --cell 0 0
./run_roboduet_raw_library.sh --viewer --trajectory random-line-000
./run_roboduet_raw_library.sh --list
```

最终代表轨迹 A0/B0、A2/B2、random-line-000、random-circle-009 全部跑完整段，
无跌倒和非有限状态。全部 68 条另做了各 2 步短测。30 项相关测试通过，
包含原生 arm/dog 观测、历史和姿态规划对照；已检查 viewer 并生成离线回放。
长程和圆轨迹仍有明显跟踪偏差，按用户最新要求不继续以评分达标为门槛。

命令、辅助控制边界和结果路径见
[RoboDuetRaw 运行说明](roboduet-raw-library-20260915.md)。

## 30. 2026-09-15 RoboDuetRaw A0/B0 底盘参与跟随

用户反馈第 29 节命令 `--viewer --cell 0 0` 的底盘不动。核对 trace 后确认，
此前“超过 0.60 m 才跟随”的门槛使 A0/B0 整段底盘速度命令为零。
现在默认 `follow` 在近处也跟随参考 XY 位移，保留起始臂端偏移与底盘 yaw，
平移速度限制为 0.25 m/s；远处继续使用原有接近目标逻辑。
`--base-mode stand` 仍保留零平移诊断行为。启动命令不变。

修正后 A0/B0 完成 935 步、无跌倒/数值故障，底盘相对起点最大平移
0.278 m，EE 位置 RMSE 0.076 m。30 项相关测试重新通过，新增近距离
参考移动时底盘必须收到非零平移命令的检查。结果和回放位于
`benchmark/results/raw_base_motion_20260915/`，旧版产物保留。
完整 viewer 与 headless A0/B0 的 trace hash 一致；A2/B2、随机直线 000、
随机圆 009 也都复查跑完整段且无跌倒/数值故障。

## 31. 2026-09-15 共用全向 waypoint PID

用户要求不规划速度的方法采用统一底盘跟随逻辑，并确认前馈速度为
“实测底盘速度 + 轨迹参考速度”。新增 `benchmark/wbc/omni_waypoint_follower.py`，
接入 Raw、Visual、DWBC 的速度输入。几何约定为 EE 投影沿轨迹切向后退
footprint 半长加余量，底盘目标 yaw 始终取轨迹切向。世界系位置/yaw PID
包含积分限幅、速度/加速度限制、静止与终点停车及任务 reset。

Raw 原有近/远距离切换已移除；Visual 保留原 arm IK/工作空间投影；DWBC
原来固定为零的三个速度观测槽接入命令。UMI 没有速度命令接口，现有 MPC
方法保留各自底盘规划。原始权重、TaskSpec、参考与 scorer 未修改。

新结果根目录 `benchmark/results/omni_follower_20260915/`：

- `raw_final` 四条代表轨迹全程完成、无跌倒/数值故障。
- `dwbc_final` 的 A0/B0 完成 935 步，无跌倒/数值故障。
- `raw_viewer` 为 A0/B0 完整 Xvfb viewer 检查。
- `visual_final` 四条中随机圆 009 全程完成；A0/B0、A2/B2、随机直线 000
  发生跌倒。Visual 原生训练 `vy=0`，施加较小横移/后退限幅仍未解决，
  当前只能确认控制器接入，不能写成 Visual 全库可稳定运行。

窗口新增蓝色 waypoint、目标 footprint 和切向线。receipt 记录共用算法、
完整参数与 source hash；trace 新增 `follower_*` 诊断，原参考不变。
完整公式、参数及适用范围见
[全向 waypoint PID 说明](omni-waypoint-follower-20260915.md)。

## 32. 2026-09-15 library2 A5/B3 跟随诊断

用户指出 `frozen_trajectory_library2 --cell 5 3` 后段底盘和 yaw 均未跟上。
trace 确认旧版平移/yaw 命令分别约 95%/81% 时间饱和；参考速度峰值
0.489 m/s，切向角速度峰值 6.72 rad/s，而旧限制只有 0.35 m/s 和
0.60 rad/s。0.36 m footprint 偏移还使 waypoint 速度 P95 达到约
0.98 m/s；约 15.8% 路段曲率半径小于该偏移。

另外修复了 yaw 误差的结构问题：目标与实测 yaw 现在连续展开，实际落后
超过 π 后不会因 wrapped shortest-path 误差反向追赶。Raw 限制调整到
0.50 m/s 总速度、0.45 m/s 横移、0.35 m/s 后退、1.00 rad/s yaw，仍位于
checkpoint 训练采样范围。原时间律 `tuned_v2` 完成 1272 步、无跌倒，
平均/末端 waypoint 误差约 0.40/0.50 m，平均绝对 yaw 误差约 68°。

新增 `--playback-speed` 诊断选项，默认 1.0 不改变 TaskSpec 时间律。0.15
倍运行完成 8194 步、无跌倒，平均 waypoint 误差约 0.21 m、平均绝对 yaw
误差约 45°。慢放 receipt 明确标为非 TaskSpec timing comparable；0.10 倍
因低层策略出现反向自旋而退化，不推荐。当前证据说明调参能避免跌倒并减少
落后，但 A5/B3 的曲率和原时间律超出该低层策略保持切向 yaw 的执行能力。
结果位于 `benchmark/results/omni_cell53_diagnosis_20260915/`。
