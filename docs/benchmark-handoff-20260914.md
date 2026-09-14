# Legged Manipulation Benchmark：M12 合并与 MuJoCo 接入交接

日期：2026-09-14，Asia/Shanghai。

主仓库：`/home/simon/Projects/WBC/RoboDuet`。

部署仓库：`/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar`。

## 1. 用户目标与执行状态

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

## 2. 已核实的仓库状态

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

> 读取 `/home/simon/Projects/WBC/RoboDuet/docs/benchmark-handoff-20260914.md`，开始执行。先从当前 v3-stage2 集成 frank/m12-benchmark-finish，完成评分和任务公平性修复，再实现公共任务/指标与 rl_sar MuJoCo 后端。保留已有改动，按阶段完成有界验证并记录证据。

## 12. 证据边界与参考

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
