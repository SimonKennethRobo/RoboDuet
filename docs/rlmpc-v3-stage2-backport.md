# v3-stage2 选择性移植（2026-09-09）

目标分支 `feat/rlmpc`，起点 `8a7ea99`；来源 `v3-stage2`，检查至
`d97b56d`；共同祖先 `2a6ac7a`。开始时工作区干净。按
`project-design-rlmpc-v3.md` 和 `project-design-rlmpc-v3-coding.md` 的
Contribution 1 / 数据接口范围选择改动，不整分支合并。

## 移植决策

| 来源提交                                                        | 处理                                       | 与需求的关系                                                                                                         |
| --------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `3e7ed9e` benchmark 并行化                                    | 移植 benchmark 和说明，适配本分支 R2       | 同一策略一次批量推理多个测试点；指标按策略/测试点分片；策略在 sim device 推理；忽略隐藏候选目录                      |
| `f01e85c` benchmark 修复                                      | 只移入 adaptation 开关不参与尺寸比较的修复 | 同宽不同 adaptation 策略可比较，真实尺寸不匹配仍报错                                                                 |
| `02ba376` reset/姿态终止冲突                                  | 移入 roll/pitch 终止检查及宽限参数         | 本分支原先定义了开关却未执行检查；启用时给 reset 恢复机会，超时和低高度终止不受宽限影响                              |
| `a229e5a` 推扰课程                                            | 只移入每环境随机间隔和 reset 重排          | 保持 R8 第四阶段强度门控；不叠加从迭代 0 开始的另一套强度课程                                                        |
| `033f4ef`, `1aa5fb8` RL-SAR 导出/自动导出                   | 不移植                                     | 当前 R7 观测含参考状态及 adaptation 输入；这些通用部署变更不能单独建立本分支的部署契约，且播放不应附带写外部部署目录 |
| `4b4caab` 观测开关/紧凑布局                                   | 不移植                                     | 避免额外改变 R1 对照维度和 R7 checkpoint 布局；保留相位输入                                                          |
| `0456e8d` 无时钟奖励                                          | 不移植                                     | R3/R5 依赖固定 trot、已知相位；暂不引入不同步态训练目标                                                              |
| `5702272`, `c2905f1`, `d97b56d` Raibert/接触/冲击奖励修改 | 不移植                                     | 会同时改变鲁棒性基线和奖励形状，需要单独消融，不能归因为响应一致性方法                                               |
| `51ceb3d` 启用推扰/姿态 reset 及调参                          | 不整体移植                                 | 保持当前 R8 和 reset 课程标定；不搬用另一批训练得出的阈值、范围和奖励权重                                            |

## 本分支适配

- benchmark 的 `response_consistency_rmse` 继续读取二阶
  `response_ref.xi[:, :2]`，不退回旧 `dog_vel_ref`。它只是已有的二维速度
  诊断指标，不能替代 R9 要求的五通道、相位去趋势和跨域评测。
- benchmark 关闭训练用 grouping 和 excitation，防止内部组重采样和
  PRBS/chirp 覆盖各测试 cell 的命令；观测宽度和参考模型不变。
- 回归中补修本分支的 twin 单独 reset 相位漏洞：除 reset 成员外，同组存活成员也同步到 twin 的新相位，避免屏蔽窗口结束后仍异相比较。原始 HEAD 的定向复现得到 `[0, 0.6, 0.6, 0.6]`，修复后整组为 0。
- 推扰排除 nominal twin，修复原路径在第四阶段也推动标称锚点的问题。
  即使强度为 0，也按期更新随机时钟，避免进入第四阶段时全池同时补推。
- `domain_rand.push_interval_s_range=[1.0,8.0]`；`None` 回退到
  `push_interval_s`。水平推扰速度上限 `0.5 m/s`，角速度上限沿用 `0.6 rad/s`。
- 推扰线性课程启用：初始比例 `0`，增长 `8000` 次迭代。实际强度为
  `R8 disturbance intensity * clip(iteration / 8000, 0, 1)`。在默认 R8
  第四阶段开始时来源课程已达到 1，因此仍按 R8 从 0 到 1 开启推扰。
- `rewards.terminal_body_ori=pi/3`，`terminal_roll_pitch_grace_s=1.0`。
  当前 WTW profile 已启用 `use_terminal_roll_pitch=True`，因此默认生效。
  超时和低高度终止不受姿态宽限影响。
- reset 范围继承现有 WTW：z `0.5`、roll/pitch/yaw `3.14`，初始比例 `0.1`；
  与来源一致。课程门限改为 `0.4`，增长时长改为 `8000` 次迭代。

## 奖励最终值（用户追加授权）

| 项目 | 值 |
| --- | --- |
| `rewards.raibert_form` | `quadratic` |
| `reward_scales.raibert_heuristic` / `wbc.reward_scales.raibert_heuristic` | `-1 / -1` |
| `rewards.raibert_sigma` | `0.35` |
| `reward_scales.feet_contact_forces` | `-0.01` |
| `reward_scales.feet_impact_vel` | `+0.4` |
| `rewards.feet_impact_vel_sigma` | `0.02` |

接触/冲击项在 WBC 表没有独立覆盖，注册时继承上述值。冲击奖励为
`exp(-sum(contact * clip(previous_vz, -100, 0)^2) / 0.02)`。
`auto_train.py` 和 `unified_train.py` 支持来源的 `--raibert_exp`，启用时
Raibert 变为 `exp(-placement_error_sq / 0.35)`，两个 policy stage 的权重为
`0.4 / 0.2`。`set_raibert_form`、符号校验及 `resolve_reward_scales` 已移入。
来源把 `raibert_sigma` 误写到 `reward_scales` 的配置项改放到 `rewards`，
数值不变，避免将 sigma 注册成不存在的奖励函数。

R4 的参考跟踪和姿态交接、R1 命令裁剪、R7 观测布局保持现有实现。
旧 checkpoint 的配置恢复仍可能覆盖当前默认参数，恢复训练需检查实际配置。

## 验证方法

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m pytest go1_gym/response -q
# 独立进程保证 IsaacGym 在 torch 前导入
python -m pytest benchmark/test_stage2_backports.py -q
python scripts/check_stage2_backports.py
python scripts/check_response_runtime.py --check r5 --num_envs 32
```

`check_stage2_backports.py` 使用零动作验证 R8 域随机化/推扰门控、实际推扰写入及两个策略切片 ×
两个命令 cell 的仿真集成。不能把它作为行走能力、benchmark 加速比例或
论文实验性能证据。完整训练、三档域的 N=20 策略评测和硬件执行不在此次验证中。

## 本次验证结果

- 原有 response 数学/配置测试：237 passed（`/tmp/rlmpc-backport-pytest.log`）。
- 移植专项回归：8 passed（`/tmp/rlmpc-backport-targeted.log`），覆盖命令分片、指标池、训练命令写入隔离、推扰门控/twin 排除、reset 时钟、姿态宽限、twin reset 后存活成员相位。
- 最终 R5：32 env × 1200 步，0 command mismatches、0 phase mismatches；强制跌倒后 50 步恢复；twin 参数全程保持标称（`/tmp/rlmpc-backport-r5-fixed.log`）。
- 首轮完整 R8 + 仿真 smoke 通过；最终代码的完整 R8 零动作复跑因 `phase_variance never became non-zero` 失败（`/tmp/rlmpc-backport-runtime-final.log`）。该检查依赖 rollout 产生非零信号，此次不声称完整 R8 验收通过。专项 smoke 只检查本次相关的域随机化/推扰调度和实际仿真写入，不放宽原有完整 R8 断言。
- 最终专项 IsaacGym smoke 通过：R8 域/推扰调度、实际 indexed 推扰写入、twin 排除与 reset 相位、两策略 × 两命令 cell 和 R2 指标（`/tmp/rlmpc-backport-runtime-scoped.log`）。
- Python 编译与 `git diff --check` 通过。

原始 HEAD 的随机 R5 rollout 本次通过，但定向执行其 twin 单独 reset 路径可以复现相位分裂；因此增加定向回归，而不依赖随机 rollout 一定触发该边界情况。

改动未提交或推送。工作期间出现的 `.stignore` 外部修改未纳入移植，也未改写。


## 追加移植验证

本轮日志：`/tmp/rlmpc-reward-tests.log`、`/tmp/rlmpc-reward-targeted.log`、
`/tmp/rlmpc-reward-runtime.log`。专项测试包含冲击奖励单调性、Raibert 两种形式
的一致几何误差、两个 policy stage 的权重、符号拒绝和推扰课程端点。
本轮仿真 smoke 还检查三个奖励项均已注册，实际 dt 归一化权重等于来源值。
前一轮完整 R8 的零信号失败记录仍有效；本次未运行完整训练或重新标定。
