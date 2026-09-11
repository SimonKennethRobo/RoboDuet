# 机械臂伺服辨识与 OCS2 实测闭环

本轮执行 `tmp/rlmpc-agent-handoff-20260911.md` 第 4 节下一步第 2 项。辨识对象是固定命令接口、RL policy、机械臂位置伺服与机器人；上层 MPC 不参与辨识。使用同一 `stage1_rlmpc_benchmark_2_223425` 的 last dog checkpoint，运行时观测为 112 维、history 为 50。

## 固定的命令接口

新实验共用 `benchmark/dog_policy/servo_runtime.py`：

```text
q_target(t + dt_policy) = clip(q_target(t) + dt_policy * v_arm, q_lower, q_upper)
arm_action = (q_target - default_q) / action_scale
```

目标跨 MPC 周期保留，不再每 0.1 s 重新锚定实测 q。每个 policy tick 都核对 `joint_pos_target[:,12:18]` 与请求目标；96 条采集轨迹的最大差异约为 `1.12e-7 rad`。保持 checkpoint 的 `control_type=M`，通过 IsaacGym actor 属性回读确认六轴 stiffness 为 `[50,50,80,30,20,20]`，damping 为 `[5,10,10,2.5,2,1]`。未通过修改 PD 增益来获得辨识结果。

本轮没有修改训练环境、policy、参考 OCS2 工作区。旧 `closed_loop_mpc.py` 原型保留，便于追溯不同接口与实验协议。

## 数据、模型和留出方式

- 主采集：`data/identification/arm_servo_20260911/`。72 条 × 24 s，6 个姿态中心，每个姿态包含 6 个关节单独激励、5 个底盘通道单独激励、1 个同时激励。36 条训练、12 条验证、24 条开发期测试；姿态中心和相位按整条轨迹分开。
- 最终确认：`data/identification/arm_servo_confirmation_20260911/`。模型冻结后另采 24 条 × 24 s，使用另外 2 个姿态中心、新相位与 chirp 频率；没有用于拟合或选择模型。96 条轨迹均无 reset。
- 数据记录实际 q、dq、目标角、底盘响应、根状态、末端位姿、有效区间和实际驱动属性。主采集关节目标速度最大约 `0.54 rad/s`；这并没有覆盖 MPC 全部允许的关节速度范围。
- `models.json` 保存训练/验证选择记录、系数、谱半径、各关节及各轨迹预测误差、policy/URDF/数据 SHA256。`confirmation.json` 保存独立确认结果与整条轨迹重采样的 bootstrap；两个新姿态中心不足以建立普遍泛化保证。

100 ms 离散模型为：

```text
z = [vx, vy, wz, z_base, pitch, q(6), previous_q(6), previous_target(6)]
w = [cmd_vx, cmd_vy, cmd_wz, cmd_height_offset, cmd_pitch, target_at_interval_end(6)]
z_next = A z + B w + c
```

机械臂行包含目标误差、目标增量、姿态相关的线性负载项，以及候选的一阶/二阶位置历史项。`separate` 分别拟合底盘与手臂；`coupled` 额外允许底盘行使用手臂姿态/增量、手臂行使用底盘响应。验证集在 100/300/500/1000 ms 多步自由预测上选择阶数和 ridge，排除不稳定或负伺服增益候选。耦合项只有在验证集归一化 MSE 降低超过 10% 时才选用；本次没有达到。

开发期曾检查过原 24 条测试轨迹来修订模型结构，因此最终结论使用后来采集的独立确认轨迹。自由预测只在窗口开始时注入一次实测状态，不注入后续实测底盘/手臂状态。100 ms 内的位置目标变化用终点表示，底盘命令用区间起点表示，是低阶离散近似。

确认数据上的六关节综合角度 RMSE：

| 预测时域 | 理想关节速度 | 分别辨识 | 显式耦合 |
| -------- | -----------: | -------: | -------: |
| 100 ms   |      0.229° |  0.206° |  0.203° |
| 300 ms   |      0.453° |  0.362° |  0.371° |
| 500 ms   |      0.624° |  0.503° |  0.513° |
| 1000 ms  |      0.573° |  0.696° |  0.700° |

分别辨识在 300–500 ms 降低约 20% 的综合误差，但 1 s 恶化，J3 也没有稳定改善。耦合/分别辨识的确认集归一化多时域 MSE 比值为 `1.0094`，本数据不支持启用显式耦合模型。

图和逐轴结果：[`prediction.png`](../data/identification/arm_servo_confirmation_20260911/prediction.png)、[`report.md`](../data/identification/arm_servo_confirmation_20260911/report.md)。

## OCS2 接入

`benchmark/ocs2_servo/solver.cpp` 直接链接参考工作区已有的 OCS2 `SqpSolver` / HPIPM。Python 与独立 C++ 进程通过本机 stdin/stdout 的 JSON 行交互；无 ROS topic 或 dummy rollout。每次规划使用 IsaacGym 实测反馈，所有实际运动都由 RL policy、位置伺服和真实仿真物理推进。

预测状态显式包含持续的手臂目标和上次输入；拟合模型不会把命令积分器折叠为伺服增益。离散辨识模型通过固定 100 ms Euler 网格适配给 OCS2，逐个区间检查时间网格和动力学缺陷；此适配不应被当成可任意更改步长的连续 ODE。

末端代价直接计算 `T_world_base * FK_URDF(q)`，包括 URDF 固定关节和原环境 EE 偏移。旋转误差使用三维旋转角，roll 在每个预测时域内保持当前实测值。每次规划比较 C++ FK 与 IsaacGym 实测末端位姿，超出 3 mm / 0.01 rad 就停止该试验。正运动学不经过辨识。

三组共用 1 s 时域、EE 权重位置 200 / 姿态 30、相同输入与平滑权重、相同命令/变化率/关节目标限幅。默认姿态命令边界沿用 screening 范围；没有引入旧诊断中只对辨识组扩大的范围，也没有把经验范围称为辨识出的 feasibility envelope。

参考版本 `SqpSolver::getOCPSolution()` 只把等式送入 HPIPM；裸不等式只进入残差统计。新接入显式添加 `StateInputSoftConstraint` / `StateSoftConstraint` 的 `RelaxedBarrierPenalty(mu=.001, delta=1e-6)`，并在输出前重新检查全预测时域约束边界。障碍函数本身是软约束，越界解会被拒绝；`ok` 表示数值有限、边界和动力学缺陷通过检查，不是最优性证明。

接入初期的 `benchmark/results/ocs2_servo_20260911/` 使用裸不等式，已停止并标为 superseded。第二轮 `ocs2_servo_validated_20260911/` 的到达任务碰到 30 次 SQP 迭代上限，边界检查拒绝了该解；保存同一个问题离线复算后，第 59 次迭代通过检查。最终所有组统一使用 120 次上限，并完整重跑，结果写入 `benchmark/results/ocs2_servo_final_20260911/`。前两轮不作为最终精度对照。

报告脚本检查每组 policy、URDF、二进制、初始状态、实际配置、权重和约束完全一致。仿真同步推进，没有模拟通信或计算延迟，也不据此评价实时部署性能。

## 完成的闭环结果与判断

最终为 2 个任务 × 2 个 seed（29、43）× 3 个模型，每条 24 s。12 条全部完成，2880 次 OCS2 输出均通过检查；最小预测约束余量 `3.94e-4`，最大离散动力学缺陷 `2.52e-15`。FK 与实测 EE 的最大差异为约 `1.00e-6 m / 2.08e-6 rad`。编译通过，4 项接口/拟合隔离/原生 OCS2 约束测试通过，Python 语法、shell 语法与 `git diff --check` 通过。

下表按两个 seed 的所有等长样本合并 RMSE，单 seed 数据见[闭环报告](../benchmark/results/ocs2_servo_final_20260911/report.md)。

| 任务      | 预测模型              | 末端位置 RMSE | 末端姿态 RMSE |
| --------- | --------------------- | ------------: | ------------: |
| hold_push | 辨识底盘 + 理想臂速度 |       0.30 cm |        0.35° |
| hold_push | 底盘/手臂分别辨识     |       1.31 cm |        2.13° |
| hold_push | 显式耦合              |       0.92 cm |        1.49° |
| reach     | 辨识底盘 + 理想臂速度 |      11.17 cm |        9.20° |
| reach     | 底盘/手臂分别辨识     |       7.59 cm |        7.75° |
| reach     | 显式耦合              |       7.36 cm |        7.44° |

`hold_push` 保持初始实测 EE 世界位姿，6/12/18 s 施加交替横向 20 N 推力，设定每次持续 0.25 s；调度分辨率为 20 ms。`reach` 在 3 s 内将目标平滑移动 `[0.65,0.15,0.08] m`，随后保持姿态与位置目标。

分别辨识使 reach 位置误差降低约 32%、姿态误差降低约 16%，但持姿明显退化。显式耦合在这两类闭环轨迹上比当前分别辨识候选略好或更好，但仍未解决持姿退化，独立确认集也没有证明其增量预测价值。因此可以确认辨识模型在移动到达任务上的收益，不能声称其普遍优于理想臂模型，也不能据此判定显式底盘—手臂耦合已经必要。

闭环实际访问的动作分布不同于独立辨识激励：MPC 允许的臂速度比采集覆盖范围更宽。报告同时保存了每组自己访问轨迹上的 100 ms 关节预测误差；这是机制诊断，不能代替同输入留出对照。1 s 留出预测退化、J2/J3 偏差以及动作范围外推仍需继续解决，尚未证明某一个因素就是持姿退化的根因。

旧 SLSQP hold 结果与本轮在命令接口、运动学和优化实现上均有不同，不应把跨协议的精度变化全部归因于 OCS2 或伺服辨识。

## 运行入口

先用系统 C++ 环境构建。可通过 `OCS2_WORKSPACE` 指向另一个已有安装的 OCS2 workspace，默认参考 `/home/simon/Projects/Simon/wbc_rl_mpc/ros2_ws`。

```bash
bash benchmark/ocs2_servo/build.sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出路径必须不存在；防止覆盖已有实验。
python -m benchmark.dog_policy.collect_servo_identification --output data/identification/arm_servo_new
python -m benchmark.dog_policy.servo_model data/identification/arm_servo_new
python -m benchmark.dog_policy.collect_servo_identification --confirmation --output data/identification/arm_servo_confirmation_new
python -m benchmark.dog_policy.report_servo_experiments --models data/identification/arm_servo_new/models.json --confirmation data/identification/arm_servo_confirmation_new

python -m benchmark.dog_policy.ocs2_servo_mpc --mode separate --task hold_push --seed 29 --seconds 24 --output benchmark/results/ocs2_servo_trial_new
python -m unittest benchmark.dog_policy.test_servo_ocs2 -v

# 完整配对：2 tasks × 2 seeds × 3 models，每条 24 s。
bash scripts/run_ocs2_servo_experiments.sh benchmark/results/ocs2_servo_batch_new
```

单次闭环可用 `--models` 指定模型文件，`--mode` 为 `base_only`、`separate`、`coupled`。其中 `base_only` 仍使用辨识底盘，手臂按理想速度预测。模型文件中的 `selected_on_validation=separate` 只表示在两种辨识结构间选择分别辨识，不意味着已经优于理想速度基线。
