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
