# 指定 policy，自动采集、辨识和导出 MPC 配置

统一入口为 `sysid/identify_policy.py`，复用同目录下的采集、拟合和导出脚本。

## 一条命令跑完整流程

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
  --policy I_Q \
  --output tmp/experiments/identify_iq_v1
```

换策略时只修改 `--policy` 和实验输出目录，例如：

```bash
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
  --policy coord_G_27999 \
  --output tmp/experiments/identify_coord_g_v1
```

`--policy` 也接受完整策略目录或其 `policy.pt` 路径。文件需要按 RL-SAR 部署格式组织：

```text
go2_x5/
├── base.yaml
└── MY_POLICY/
    ├── config.yaml       # YAML 顶层键为 go2_x5/MY_POLICY
    └── policy.pt         # 导出的 TorchScript；不是 PPO 训练 state_dict
```

可以用 `--robot-dir` 更换按名称查找策略的目录，用 `--stack-root` 更换 OCS2 工作区，
用 `--scene` 指定 MJCF。文件路径可以是绝对路径；输出相对路径按当前 shell 目录解释。

新实验在任何仿真之前自动把 policy 部署包冻结到结果目录：

```text
identify_MY_POLICY/
└── policy_bundle/
    ├── manifest.json
    └── go2_x5/
        ├── base.yaml
        └── MY_POLICY/
            ├── config.yaml
            └── policy.pt
```

采集、续跑、MPC 导出和后续轨迹库评估均使用这份快照。`manifest.json` 保存源路径和
三个文件的 SHA-256；续跑会先验证快照，源部署目录后来修改或切换默认 policy 不会改变
已经开始的实验。旧实验没有 `policy_bundle/` 时保持旧的原路径及散列校验方式。

## 自动执行什么

1. 检查策略配置、观测布局、网络输入输出和 MuJoCo 站立预检。
2. 采集 72 条、每条最多 20 秒的激励轨迹：36 train、18 development、18 test。
3. 用 train 拟合 F0（无延迟一阶）、F1（带延迟一阶）、F1_gait 和 F2（带延迟临界阻尼二阶）。
4. 用 development 的 0.1/0.3/0.6/1.0 秒因果预测误差选择残差和最终模型；test 只验证冻结结果。
5. 输出逐通道 RMSE/MAE/bias、逐轨迹 RMSE、PNG/PDF 图，再导出全部候选和自动选择的 `.info`。

采集使用 step、chirp、multi-sine、真正的二值 PRBS、联合六通道及机械臂运动激励，与末端 benchmark 轨迹库独立。
train/development/test 使用不同随机相位、种子和频率缩放。拟合目标按“轨迹 × 预测时域”等权，
避免长轨迹或短时域独占目标函数。F2 为
`y_ddot = omega^2 (gain*u+bias-y) - 2*omega*y_dot`，不额外拟合阻尼比，控制模型复杂度。
当前完整辨识协议的激励幅度、波形、时长在 `sysid/identify_iq_mujoco.py` 的
`AMPLITUDES`、`protocol()`、`commands()` 中维护；部署限幅在 `sysid/run_iq_mpc.py`
的 `LIMITS` 中。默认沿用已有 I_Q 辨识实验的激励范围；更换 policy 后应核对其训练范围。

```text
identify_coord_g_v1/
├── pipeline_inputs.json             # 输入与源码散列，续跑时核对
├── pipeline_state.json              # 当前阶段、失败原因、完成状态
├── protocol.json
├── raw/                            # 采集数组和逐条 success/failure
├── models_v2/
│   ├── F0.json
│   ├── F1.json
│   ├── F1_gait.json
│   ├── F2.json
│   └── selection.json
├── models_v2_prediction_validation.json
├── identification_quality/
│   ├── quality.json                # 全模型、全时域、逐通道/逐轨迹指标
│   ├── REPORT.md
│   ├── rmse_by_horizon.{png,pdf}
│   └── prediction_traces.{png,pdf}
├── logs/                           # collect.log / fit.log / report.log / export.log
└── mpc/
    ├── task_coord_G_27999_ideal.info
    ├── task_coord_G_27999_first_order.info
    ├── task_coord_G_27999_first_order_delay.info
    ├── task_coord_G_27999_gait.info
    ├── task_coord_G_27999_second_order.info
    ├── task_coord_G_27999_selected.info
    └── manifest.json
```

在另一个终端用 `tail -f <输出目录>/logs/collect.log` 或 `fit.log` 查看实时进度。
失败轨迹保留在原始数据和验证统计中；流程完成表示产物齐备，不表示该 policy 的预测或跟踪性能合格。
新 `.info` 保存在实验目录，通过运行参数加载。OCS2 缓存按任务内容和模型库散列隔离。

## 先检查或冒烟测试

```bash
# 只检查接口、执行三秒站立预热
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
  --policy I_Q --stage check --output tmp/experiments/identify_iq_v1

# 使用同一目录继续，只采一条轨迹；不执行拟合或导出
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
  --policy I_Q --smoke --resume --output tmp/experiments/identify_iq_v1

# 接着采完剩余轨迹、拟合和导出
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
  --policy I_Q --resume --output tmp/experiments/identify_iq_v1
```

中断后同样追加 `--resume`，已经完成的数据会验证散列并跳过，完成的拟合/导出也会验证后跳过。
如需分阶段执行，使用 `--stage collect`、`--stage fit`、`--stage report`、`--stage export`。
已有目录需要 `--resume`；更换策略、配置、场景或相关源码后使用新目录。

## 加载新模型运行 MPC

运行脚本从实验 manifest 自动选择匹配的 policy、机器人目录和 `.info`：

```bash
/opt/miniconda3/envs/base312/bin/python sysid/run_iq_mpc.py run \
  --root tmp/experiments/identify_coord_g_v1 \
  --controller selected --scenario walking_curve --seconds 12 \
  --viewer --output tmp/experiments/identify_coord_g_v1/mpc_run_01
```

`selected` 是开发集自动选中的模型。也可显式选择 `first_order`、`first_order_delay`、
`gait` 或 `second_order` 做消融。每次闭环测试使用不同的 `--output`。
辨识流程默认无窗口；上述 `--viewer` 打开 MPC 闭环的 MuJoCo 窗口。

## 在冻结轨迹库上比较辨识开关

下面从已有的 `frozen_trajectory_library2/suite` 无放回随机抽取相同的 10 条任务，
对四个 policy 各运行 `ideal` 和 `selected`，共 80 次正式闭环测试：

```bash
/opt/miniconda3/envs/base312/bin/python sysid/run_policy_library_benchmark.py \
  --experiments tmp/experiments/identify_iq_v2 \
    tmp/experiments/identify_NH_D_s11 tmp/experiments/identify_robotlab_v2 \
    tmp/experiments/identify_coord_I_26499 \
  --library benchmark/data/frozen_trajectory_library2 \
  --seed 20260915 --count 10 --workers 4 \
  --output tmp/experiments/policy_sysid_library2_comparison
```

输出目录需全新；续跑同一目录追加 `--resume`。`--prepare-only` 只冻结输入，
`--smoke-only` 对每个组合运行两步预检；预检不计入正式结果。
`--resume --report-only` 根据已落盘的执行记录更新汇总。

测试复用库内既有 TaskSpec、初始物理状态、时限、原始 trace 和离线 scorer。
两侧统一使用辨识实验的部署限幅；F1_gait 接收策略在状态采样边界的实测步态相位。
策略部署包和 MPC 配置复制到输出目录；若 `base.yaml` 改变，脚本仅在匹配散列的
Git 历史中寻找采集时的版本，并恢复到输出目录。`selected` 固定为辨识开发集选型，
不根据这 10 条轨迹重新选择或调参。保留各实验原有 MPC 导出参数，因此应优先
解读同一 policy 内的成对差异；跨 policy 的基础代价差异记录于报告。

`REPORT.md`、`results.json` 和 `per_trial.csv` 汇总位置／姿态 RMSE、成功、
跌倒及运行失败。RMSE 包含有 trace 的失败轨迹，早停时仅覆盖已运行区间；
`results.json/pairs` 另报告两侧共同时间前缀的成对位置 RMSE。
所有失败仍计入每个组合的任务分母。

## 多 policy 自动辨识并完成轨迹库评估

统一入口 `sysid/run_policy_identification_benchmark.py` 接收一组 policy，依次完成完整辨识，
待所有 policy 的 `collect/fit/report/export` 完成后，在同一批随机轨迹上运行每个 policy 的
`ideal`（无辨识）和 `selected`（有辨识）：

```bash
/opt/miniconda3/envs/base312/bin/python \
  sysid/run_policy_identification_benchmark.py \
  --policies I_Q NH_D_s11 robot_lab_rear_r30o_s42_11497 coord_I_26499 \
  --library benchmark/data/frozen_trajectory_library2 \
  --seed 20260915 --count 10 \
  --identification-workers 1 --evaluation-workers 4 \
  --output tmp/experiments/policy_identification_library2
```

也可以提供文本清单（每行一个 policy，`#` 开头为注释）或 JSON 字符串数组：

```bash
/opt/miniconda3/envs/base312/bin/python \
  sysid/run_policy_identification_benchmark.py \
  --policies-file policies.txt \
  --output tmp/experiments/policy_identification_library2
```

中断后使用完全相同的 policy 列表和参数追加 `--resume`。`--prepare-only` 只解析 policy、
冻结轨迹库散列和执行计划，不开展辨识或仿真。输出结构为：

```text
policy_identification_library2/
├── plan.json
├── state.json
├── logs/
├── identification/<policy>/       # 含各自 policy_bundle 和完整辨识结果
└── evaluation/
    ├── REPORT.md
    ├── results.json
    ├── per_trial.csv
    └── runs/<policy>/<ideal|selected>/<trajectory>/
```

辨识可并行，但每个 worker 都进行 MuJoCo 采集；默认使用一个辨识 worker，避免 CPU 争用。
评估默认一条 policy 一个执行通道。只有所有辨识目录验证为 `all_complete` 后才会开始评估，
任何失败都会写入 `state.json` 并保留日志和已有产物。

RC_s17 的“根据预测结果修正参数”在这里实现为离线 prediction-error refinement：候选延迟离散搜索，
`tau/omega` 连续优化，给定动态参数后以最小二乘重新估计 gain/bias，最后由 development 选型。
没有把参数在线写回正在运行的 MPC：OCS2/CppAD 模型和缓存依赖固定参数，在线改变它们既破坏可复现性，
也不能由当前离线预测证据证明更好。`quality.json` 明确只证明 held-out 预测质量；是否改善 MPC 跟踪，
仍需用冻结任务做成对闭环 benchmark。

## 当前支持范围

支持 Go2-X5、12 维腿动作和 5 ms 控制/20 ms 策略周期。命令 observation 可以是
`roboduet/dog_commands`（3/5/6 个物理命令，后续元素可为 gait metadata），也可以是
robot_lab 的 7 维 `robot_lab/velocity_pose_commands`。内部统一映射为
`vx, vy, wz, height, pitch, roll`；只采集实际存在的通道。缺失通道输出 gain=0 的
不可辨识常值模型，deployment clamp 为 0，不会让 MPC 发送 policy 从未观察过的命令。
观测维数和历史长度由配置读取；固定步频及静止时钟规则也由配置读取。
没有 `gait_frequency` 的 policy 仍可完成 F0/F1/F2 辨识，但不拟合 gait-phase residual。
当前观测适配器不认识的 term、不同机器人或周期会在采集前报错。
仅有一个没有配套配置的训练 `.pt`，或者来自其他框架且没有观测/动作适配器的网络，需先完成部署导出/适配。
gait 模型需要部署配置中有可观测的固定步频；不同步态/地形/载荷的效果需另行验证。
