# 跨方法 MuJoCo benchmark：实现与实测交接（完成版）

日期：2026-09-15。工作区：`/home/simon/Projects/Simon/wbc_rl_mpc`。

本文件是 `baselines/DOC/CROSS_METHOD_MUJOCO_BENCHMARK_HANDOFF_4H.md`
的版本控制副本；工作区顶层不属于 Git 仓库。

原 4 小时实现计划已经执行完成。当前公共 benchmark 位于
`/home/simon/Projects/WBC/RoboDuet/benchmark/wbc`，分支为
`feat/cross-method-mujoco-qm-control`。以下第 1–7 节保留原设计依据和
执行计划，当前状态及可直接使用的产物以第 0 节为准。

## 0. 当前完成状态（2026-09-15）

八种交接方法均已接入同一 Go2+X5 MuJoCo plant，可通过统一入口运行：

`roboduet`、`roboduet_raw`、`ma2022`、`deep_whole_body_control`、
`visual_wholebody`、`umi`、`wb_locoman`、`qm_control`。

公共实现包括冻结 TaskSpec 校验、nominal/push 派生、trace-v3、统一
scorer、指标覆盖收据、离屏 MuJoCo MP4 和参考/实际轨迹图。视频使用
控制完成后记录的 root 与 18 关节状态进行确定性 MuJoCo 回放，因此不
改变控制时序或物理结果。

最终“稍难”实测采用 A0/B1 轨迹：

- nominal TaskSpec：`timed-trajectory-4a663eb29353ed21`
- push TaskSpec：`timed-trajectory-9279b263069f5c2a`
- suite SHA-256：`ea8a721d6bb101041a4ce26794744199f20f37b6a0dc069a93a1863a1a343f97`
- 路径长度：`3.325 m`，比上一条 A0/B0 的 `2.567 m` 增加 29.5%
- 峰值等效速度：`0.205 m/s`，比 A0/B0 的 `0.163 m/s` 增加 25.8%
- 参考时长/截止时间：`30.437 s / 31.0 s`
- push：环境世界系 base 中心施加 `80 N × 0.1 s = 8 N s`
- 控制与渲染代码提交：`aee26fda8ea1ee56782dd12146df188a62f8db0e`
- 结果文档提交：`8694d6f`

最终结果根目录：

`RoboDuet/benchmark/results/cross_method_mujoco_harder_release`

其中 `README.md`、`benchmark_summary.json`、`benchmark_summary.csv`、
`trajectory_comparison.png` 和 `VALIDATION.json` 是总入口。每个方法的
时间戳目录下均有 nominal/push；每个场景包含 trace、metrics、receipt、
`mujoco_tracking.mp4` 和 `trajectory_tracking.png`。16 个视频均为 H.264、
960×540、25 fps，黄色为完整参考轨迹，青色为实际 EE 历史。

| 方法（最终 run） | nominal：样本/终止，位置/姿态 RMSE，进度 | push：样本/终止，位置/姿态 RMSE，进度 |
|---|---|---|
| `roboduet` (`20260915_095023`) | 1550/timeout，0.4536 m / 1.7691 rad，0.0000 | 1550/timeout，0.4480 m / 1.7609 rad，0.0000 |
| `roboduet_raw` (`20260915_095046`) | 1550/timeout，0.4506 m / 1.6635 rad，0.0150 | 1550/timeout，0.4210 m / 1.6840 rad，0.4361 |
| `ma2022` (`20260915_095108`) | 1550/timeout，0.2931 m / 1.6271 rad，0.0060 | 1550/timeout，0.2615 m / 1.5244 rad，0.0150 |
| `deep_whole_body_control` (`20260915_095129`) | 235/fall，0.6128 m / 2.0019 rad，0.5534 | 228/fall，0.6026 m / 2.1587 rad，0.5564 |
| `visual_wholebody` (`20260915_095137`) | 16/fall，0.2680 m / 2.1117 rad，0.1865 | 16/fall，0.2513 m / 2.1144 rad，0.1925 |
| `umi` (`20260915_095144`) | 22/fall，0.8354 m / 2.4983 rad，0.1594 | 21/fall，0.7893 m / 2.4740 rad，0.1594 |
| `wb_locoman` (`20260915_095150`) | 24/fall，0.2914 m / 1.9231 rad，0.0000 | 26/fall，0.2766 m / 1.9241 rad，0.0000 |
| `qm_control` (`20260915_095347`) | 1551/timeout，0.1901 m / 0.8574 rad，0.6580 | 1551/timeout，0.1895 m / 0.8599 rad，0.6581 |

16 个场景均为 `success=false`、`numerical_fault=false`，通用物理指标
覆盖完整。提前跌倒和 timeout 均保留在结果及分母中。qm_control 在
完整时域方法中位置和姿态 RMSE 最低，但仍未满足 endpoint/tracking/hold
成功门槛；该批次不能支持任务成功或方法晋升结论。

机器验证收据为 `VALIDATION.json`，结果为 `ok=true`：8 个 manifest 均
来自上述干净控制提交，16 个 receipt 对应一致的 nominal/push TaskSpec，
视频流参数及视频/跟踪图哈希全部匹配。相关实现说明见
`RoboDuet/benchmark/wbc/CROSS_METHOD_MUJOCO.md`，本次实测报告见
`RoboDuet/benchmark/wbc/CROSS_METHOD_HARDER_A0_B1.md`。

## 1. 目标与依据

用户目标：实现效率优先，不追求彻底公平；各方法能在同一 MuJoCo 环境运行并测评，目标是约 4 小时内完成任意一种已有方法的接入。首轮包含必要的公共底座，至少交付一个真实方法；后续方法复用底座。没有可用权重/控制器时报告具体缺项，不启动训练。

两份来源的作用不同：

- **任务、参考轨迹、指标及已完成内容的依据：**[RoboDuet benchmark 交接](benchmark-handoff-20260914.md)。先读第 0 节概览，再读后续更新，尤其第 19、21–22、24、26–27 节；第 6–7 节用于理解契约和指标。文档顶部仍有历史阶段限制，状态以相关后续更新为准。本次用户要求已进入跨方法 MuJoCo 阶段，不继承早期“暂不做 MuJoCo”的限制。
- **旧多方法执行链的参考：**[OLD_CROSS_WBC_BENCHMARK_SUMMARY.md](OLD_CROSS_WBC_BENCHMARK_SUMMARY.md)。复用其中的公共 plant、方法 adapter、MPC sidecar 思路和实际可找到的代码；外机路径、旧任务/指标和 full 完成状态不能当成本机当前事实。

现在的工作是把有继承和包含关系的两套 benchmark 接起来：**RoboDuet 提供现行任务/指标体系，旧 Cross-WBC 提供多方法 MuJoCo 接入经验与可复用实现。** 不再建设第三套轨迹和评分体系，也不为统一架构大规模搬迁代码。

## 2. 最短实现路线

```text
RoboDuet 现行轨迹生成器 → 冻结 suite / TaskSpec / reference archive
                                     ↓
                   同一 MuJoCo plant + 方法 adapter / sidecar
                                     ↓
                         legged-manip-trace-v3
                                     ↓
                   RoboDuet 现有 scorer → 逐任务指标/汇总
```

优先扩展本机已有 `RoboDuet/benchmark/wbc/mujoco.py` 或可直接工作的公共 runner。需要工作区统一命令时，只加薄的 `cross_benchmark` 入口；新目录和框架重构不是前置条件。

物理循环负责模型、reset、参考、push 和记录；adapter 只负责状态/目标转换、方法内部状态和控制输出。直接调用现有 Python 推理即可，C++/ROS 方法沿用现有 sidecar。输出先兼容 `q_des + 方法 PD` 或 `tau`。保留各方法的观测、历史/RNN、控制频率、PD、IK 和求解参数。

共同环境先选 `baselines/mpc_baseline/mujoco_models/go2_x5_description/mjcf/scene.xml`，复用其资源和默认平地参数，物理步长 `0.0025 s`。若现成公共 runner 已使用另一份能工作的 Go2+X5 scene，可直接把那份固定为所有方法共用；记录最终选择即可，不做模型重建。使用公共 TCP 和关节名称映射，必要的坐标转换放在 adapter。

首个方法由下一 session 的用户指定；未指定则先接 `qm_control`，复用已有 sidecar 和安装前缀。当前 RoboDuet floating-MPC＋locomotion policy 的 MuJoCo 路径可作环境/记录参照，不为此重新设计 MPC 或改走 RC_s17/F1。

## 3. 参考轨迹：直接继承 RoboDuet

优先读取一份本机已有的现行冻结产物：`trajectory_suite.json` 加它引用的 reference NPZ，选择少量 task。需要新任务时才调用 RoboDuet 当前生成器，生成一次后供所有方法重放。

- 来源是 `modules/trajectory_generator.py`、`modules/trajectory.py`、`modules/curriculum.py` 和 `benchmark/wbc/suite.py`。
- 保留既有几何路径 `gamma` 与时间律 `tl_t/tl_s/tl_sdot`、A/B 难度和 bank row。现行几何包含约 `0.25–5 m` XY 净位移、高曲率、terrain-relative 高度覆盖；首轮选择较短/较易的现成 task，不要求全覆盖。
- 保留 `bounded-se3-arc-time-law-v1`：`T` 是最短时长，速度/加速度受限，长路径自动延长。实际 duration/deadline 从任务读取，不能统一截成 8 秒或重新匀速采样改变时间律。
- 初态、anchor、orientation multiplier 和 push 使用 TaskSpec 内容；映射到选定公共模型时做一次固定转换并记录，所有方法复用。不能在每个方法自行 settle 后重新以其末端位置生成目标。
- 首轮运行一个完整的短任务及其 push 变体。优先使用现有 push schedule；需要派生时复用现有 task 工具，保存派生后的任务，不另定一套扰动协议。

简单正弦/圆轨迹可以用于几秒接口诊断，**不能替代继承 RoboDuet 任务的测评交付**。取消上一版“导入耗时就改用新 8 秒轨迹”的安排。

## 4. 指标与腿臂协同：复用现有 scorer

统一输出 `legged-manip-trace-v3`，调用 `RoboDuet/benchmark/wbc/trace.py` 的 `score_trace_archive()` 及现有 `scoring.py`。沿用任务的成功/终止参数和现行运动学协议，不另写一套基础 RMSE 评分。

| 指标组 | 继承内容与接入要求 |
|---|---|
| 任务完成 | success、完成时间、tracking tube、progress、fall/timeout/cutoff/numerical fault；失败保留结果 |
| 跟踪 | EE 位置/姿态误差、残差波动、P95/peak，以及 `d_lat`、timing error；直接复用已有公式 |
| Base 与运动 | base path/net displacement/drift、实际关节与 EE/base 平滑性、已有力矩/能量/接触字段 |
| 腿臂协同 | `d_lat/timing/progress` 配合 base 轨迹；保留 `rho`/comfort utilisation、`base_util_mean`、`v_ff_xy_mean`/`v_base_xy_mean`、manipulability 等已有字段 |
| 运动学诊断 | Jacobian singular value、关节限位裕度、适用方法的 IK saturation/validity；沿用已有分类 |
| 工作空间 | 已有 `workspace-probe-v1` 的 fixed-base/bounded-posture/bounded-whole-body 是可复用扩展，首轮不重新跑体素扫描 |

协同指标的实现原则：

- MuJoCo 的 reference/actual EE、base pose/twist、q/dq 足以接上路径误差、进度、base 运动等通用部分；已有 runner 也提供 Jacobian/限位记录代码，优先复用。
- `rho` 依赖 reach model，base utilisation 依赖有意义的 base feedforward。能提供就按既有语义记录；方法没有对应变量时保留 null/not-applicable，不能用假零填成好成绩。首轮不能把所有协同字段都省掉而只交 RMSE。
- 实际 base 移动更多不等于协同更好；结合任务完成/精度、额外运动与能量解释。已有工作空间/消融结果可保留，跨方法首轮不新增整套协同消融实验。`pose_only` 也不等于物理固定底座。
- 只补所选方法接入需要的 trace 字段。已有 unavailable 字段不强行补齐，不增加 self-collision、严格信息对等或求解延迟统计等任务。

记录器可参考 `baselines/mpc_baseline/benchmark/trace_recorder.py`，但其中硬编码的协议阈值与当前 RoboDuet 有差异：复用写入逻辑，参数从现行 suite/scorer 取。旧 `cross-wbc-v1` trace 若要利用，只做一次字段转换；不能直接改 schema 名称冒充 trace-v3。

## 5. 已有成果与具体入口

以下已检查源码/文件存在性；运行结果的完成状态来自 RoboDuet 交接文档，本轮没有重跑。

| 复用内容 | 路径/当前状态 |
|---|---|
| 当前统一指标与原始记录 | `RoboDuet/benchmark/wbc/{trace,scoring,evaluation}.py`；trace-v3 已包含 P2 coordination/feasibility，文档记录在线/离线核对完成 |
| 当前任务/参考重放 | `RoboDuet/benchmark/wbc/{suite,mujoco}.py`、`baselines/mpc_baseline/benchmark/task_contract.py` |
| 当前 MuJoCo 闭环 | `RoboDuet/benchmark/wbc/mujoco.py`＋`scripts/sim2sim_mujoco.py`；已支持 floating OCS2＋不同 dog policy，早期 scripted IK smoke 不是最新状态 |
| 完整 MuJoCo 轨迹实例 | `RoboDuet/benchmark/results/mujoco_ocs2_async_full_v2/{receipt.json,trace.npz}`；第 26 节记录完整轨迹运行，跟踪未通过成功阈值 |
| pose-only 实例 | `RoboDuet/benchmark/results/mujoco_ocs2_pose_only_full_v1/`；第 27 节记录其与 full 的区别 |
| OCS2/IK × locomotion matrix | `RoboDuet/benchmark/wbc/{system_cli,controllers}.py`；同步 OCS2 及 workspace runner 已在后续章节记录完成，不重复早期待办 |
| qm_control | `baselines/mpc_baseline/qm_control_baseline/src/go2_x5_whole_body_mpc/scripts/benchmark_sidecar.py` 和同包 `launch/benchmark_sidecar.launch.py`；已有 `install_aligned` |
| WB-LocoMan | `baselines/mpc_baseline/wb-locoman_baseline/benchmark_sidecar.py` |
| Ma2022 | `rl_sar/deploy/ma2022/{adapter,play}.py`；保留 GRU，配上层控制器才是全身末端任务 |
| DWBC / Visual / UMI / 原版 | `baselines/Deep-Whole-Body-Control`、`visual_wholebody/low-level`、`umi-on-legs/mani-centric-wbc`、`RoboDuetRaw`；复用各自真实策略/观测/IK，旧总结只用于找接入线索 |

当前 `benchmark/wbc/mujoco.py` 已支持并校验公共 base-force
`disturbance_schedule`；nominal/push 均已在 8 种方法上实测。其他扰动
类型仍需先扩展并明确后端中立语义。

两套 MPC sidecar 已采用 `common-mujoco-controller-v1` 的 `describe/reset/step/close`；沿用其字段和更新周期，不新造通信协议。已有同步/异步路径哪个容易跑通就先用哪个，注明即可，不把严格确定性改造加入首轮。

Learning 方法可直接加载 checkpoint＋原配置，或复用现有 RL-SAR 部署；不强制先导出统一 TorchScript/ONNX。Visual/UMI 接底层 WBC，公共任务替代视觉高层输入。RoboDuet 原生的 controller × locomotion 组合，是跨方法候选的一种组织方式，不要求所有方法拆成同样结构。

## 6. 原 4 小时执行顺序（已完成，保留供追溯）

| 时间 | 交付进展 |
|---|---|
| 0–20 min | 确定所选方法入口和可用权重/环境；从现有 receipt 定位冻结 suite/reference，选一个短任务；确认当前模型/公共 scorer |
| 20–65 min | 复用现成 MuJoCo runner，解除仅限 RoboDuet 策略的必要假设；公共模型、参考和记录跑通 |
| 65–150 min | 接所选方法 adapter/sidecar，完成短程真实闭环，修复关节/TCP/观测等阻止运行的问题 |
| 150–205 min | 接 trace-v3 的通用协同字段、方法可用诊断和 TaskSpec push；调用原 scorer 跑 nominal/push |
| 205–240 min | 完成所选短任务整段运行或有记录的失败；离线重评分，保存实际命令、配置与结果，补简短 README |

实现效率来自复用已有任务和评分，并减少 task 数量；不是通过删掉轨迹生成/协同指标另起简化 benchmark。依赖冲突用现有环境或进程隔离解决，不全量重装。无需训练、全 288 条任务、多 seed、性能调参、完整 workspace 扫描、HTML 改版或视频系统。已有报告能直接输出就使用。

## 7. 原验收清单与下一步

首轮必须交付：

1. 一条实际可用的选方法运行命令，使用继承的冻结 TaskSpec/reference 和公共 MuJoCo 模型，覆盖 nominal/push。
2. trace-v3、原 scorer 生成的逐任务结果、有效配置/方法路径和运行日志。保留任务 ID、原参考及必要的模型映射；利用已有 metadata，不额外建设审计框架。
3. 结果含完成/失败、跟踪、base 运动及已接入的协同字段，缺失的专用字段明确说明。离线重评分可运行；失败也落盘。启动几步只是调试证据，不能当整段测评完成。
4. 简短 README 写明新增其他方法只需替换哪一个 adapter/配置，以及本轮实际测到的限制。

上述四项已经完成。后续 session 若继续，应从第 0 节的失败分析、控制器
适配或轨迹难度扩展开始，不再重复公共底座接入。可直接发送：

> 读取 `baselines/DOC/CROSS_METHOD_MUJOCO_BENCHMARK_HANDOFF_4H.md` 的第 0 节及 `RoboDuet/benchmark/wbc/CROSS_METHOD_HARDER_A0_B1.md`。基于现有 8 方法公共 MuJoCo benchmark 分析 A0/B1 的 timeout/fall，不重建 benchmark、不丢弃失败样本；修改后使用冻结 TaskSpec 复跑并更新视频、轨迹图和验证收据。
