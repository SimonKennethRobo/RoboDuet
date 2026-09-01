# RL-MPC v3 训练执行备忘录

**状态**：2026-09-01 制定。R1–R9 的代码已全部落地（`go1_gym/response/` 227 项单测），
剩下的是**跑**。本文件是执行顺序、每步的验收点、以及可以并行发的实验。

设计依据在 [project-design-rlmpc-v3-coding.md](project-design-rlmpc-v3-coding.md)（需求）
和 [rlmpc-v3-r0-plan.md](rlmpc-v3-r0-plan.md)（实现记录与已做的决定）。本文件不重复它们，
只写"接下来做什么、怎么判断做成了"。

---

## 0. 名词：两套"阶段"，不要混

| 说法 | 指什么 | 由谁控制 |
| --- | --- | --- |
| `--train_stage stage1 / stage2` | RoboDuet 原有的双策略调度。stage1 = 只训四足、手臂是扰动源；stage2 = 接上臂策略做 WBC | `StageSchedule` / `global_switch` |
| **R8 阶段 1–4** | response-consistency 四个奖励项的开启时间表 | `cfg.response.curriculum.stage_boundaries` |

**本备忘录里所有的跑都是 `--train_stage stage1`。** R8 的四阶段发生在它内部。
Contribution 1 的产物是一个 locomotion policy；上臂策略（`stage2`）是后续文档的事。

---

## 1. 当前配置（已改，2026-09-01）

| 键 | 值 | 为什么是这个值 |
| --- | --- | --- |
| `response.curriculum.stage_boundaries` | `[6000, 9000, 15000]` | 实测 locomotion 在 ~2000 iter 收敛（见 §7），6000 留足余量；原 `[3000,…]` 会在 arm 扰动只有 7% 时就拍标定 checkpoint |
| `env.stage1_arm_ramp_iterations` | `6000` | arm 扰动在 progress 0.1 起步、0.8 饱和 ⇒ 4800 iter 饱和，正好在阶段 1 结束前。原值 20000 是为 40000 iter 的 stage-1 跑调的 |
| `response.omega_n` / `rate_limit` | **起点值，待标定** | R8.2 强制项。这是切开 Run A / Run B 的唯一原因 |
| `response.reward.sigma` | 按**起点** ωₙ 标的 | 标定后必须重跑 `calibrate_reward_sigma.py` |

---

## 2. Run A — R8 阶段 1（纯鲁棒），产出标定量具

```bash
python scripts/auto_train.py --headless --sim_device $SIM_DEVICE --graphics_device_id $GRAPHICS_DEVICE_ID \
	--video --dyna_gait --num_envs 4096 --num_learning_iterations 6000 \
	--train_stage stage1 --run_name stage1_rlmpc_calib
```

约 2.4 h（1.45 s/iter @ 4096 env）。

**这一跑必须是干净的**：R8.2 要用它标定参考模型，而一个已经被参考跟踪塑形过的策略
不再是"硬件能做到什么"的中立测量——它已经被拉向那个待标定的模型本身。

### 跑的时候盯

| wandb 指标 | 期望 | 不对的话说明 |
| --- | --- | --- |
| `Curriculum/{ref_tracking,phase_variance,steady_gain,domain_consistency}` | **恒为 0** | 阶段 1 不干净，标定就是循环论证 |
| `Curriculum/randomization` | 恒 0.30 | 强度没接上采样器（这个失效模式完全无声） |
| `Train_Reward_episode/rew_tracking_lin_vel` | 平台化 | 阶段 1 的退出判据 |
| `Performance/early_termination_rate` | 低位稳定 | 同上 |
| `Performance/ref_tracking_ripple_fraction` | < 0.5 | **起点参考模型的姿态通道就不可实现**；标定时格外看 pitch/height 的分桶 |

### 产出

`runs/<date>/stage1_rlmpc_calib_*/checkpoints_dog/ac_weights_last_dog.pt`

---

## 3. 标定闸门（Run A 与 Run B 之间，~30 min）

```bash
# 1. 反推 omega_n / rate_limit。全随机化下取 20 分位；姿态通道按 (domain, 相位) 联合分桶
python scripts/calibrate_reference_model.py \
	--policy runs/<date>/stage1_rlmpc_calib_*/checkpoints_dog/ac_weights_last_dog.pt

# 2. 把结果写回 go1_gym/envs/config/wbc.py 的 response.omega_n / response.rate_limit
#    （不变量：pitch 的 omega_n 必须严格小于 height 的）

# 3. sigma 依赖 omega_n，必须重标，否则 R4.1 的区分度是按旧带宽定的
python scripts/calibrate_reward_sigma.py
```

### 读结果时看两件事

- **saturation ratio**（脚本输出）：接近 1 = 带宽受限，远小于 1 = 压摆率受限。
  **二者是不同的参考模型**，MPC 要规划的对象也不同。
- **姿态通道各相位桶的离散度**：若相位桶之间差异极大，说明"平均相位可实现、不利相位
  不可实现"的风险很高，取最差桶是对的，但也要预期 ωₙ 会明显低于起点值。

---

## 4. Run B — R8 阶段 2/3/4，Contribution 1 的本体

### ⚠️ 发车前必须处理的坑

`global_switch.count` **每个进程都从 0 开始**，没有 offset。直接续跑会让 R8 课程回到
阶段 1、arm 扰动课程也回到 0 强度重新 ramp。两条路：

- **手工**：`stage_boundaries: [0, 3000, 9000]` + `stage1_arm_ramp_iterations: 1`
- **加个开关**（未做）：`auto_train.py --curriculum_offset 6000`，两个课程自动接上

```bash
python scripts/auto_train.py --headless --sim_device $SIM_DEVICE --graphics_device_id $GRAPHICS_DEVICE_ID \
	--video --dyna_gait --num_envs 4096 --num_learning_iterations 15000 \
	--train_stage stage1 \
	--stage1_ckpt_path runs/<date>/stage1_rlmpc_calib_*/checkpoints_dog/ac_weights_last_dog.pt \
	--run_name stage1_rlmpc_consistency
```

约 6 h。**至少发 2 个 seed**：本设置 200 iter 的奖励跨种子散布 11.6 倍，
单 seed 分不清"阶段 3 崩了"和"这个 seed 运气差"。

### 阶段判据

| 阶段（Run B 内的 iter） | 开的项 | 盯什么 | 达标 |
| --- | --- | --- | --- |
| 2（0–3000） | R4.1 | `Performance/ref_mae_detrended_*` | 下降；R4.1 奖励 > 0.8 |
| 3 开始前 | — | `Performance/group_desync_fraction` | **必须先降下来并稳住**。它高的时候 R5 根本没生效，此时升权重是在给一个大部分时间不存在的项加权 |
| 3（3000–9000） | R4.2/4.3/R5 | 总 reward、`Performance/phase_variance_raw` | reward 不塌。塌了 = 权重升太快，或参考模型在多数 domain 下不可实现 |
| 4（9000–） | 冻结权重 + 推力 | `Performance/early_termination_rate`、episode length | 回到阶段 1 的 90% |
| 全程 | — | `Performance/ref_tracking_ripple_fraction` | < 0.5。持续超过就回头重标定，别继续训 |

### 必须留下的两个 checkpoint

- **阶段 3 结束**（Run B 的 iteration 9000）：一致性最强、鲁棒性最弱
- **阶段 4 结束**：一致性保持、鲁棒性补回

R8.1：这两个是"鲁棒性–可预测性帕累托前沿"上的两个点，论文主图之一。
阶段 4 的存在本身就是"一致性是软目标"这一声明的实证，必须保留并报告。
（`save_interval = 1000`，两个边界都落在保存点上。）

---

## 5. Run B 之后 — R9 评测与导出

```bash
python scripts/export_identification_data.py --policy <ckpt>   # npz + 元数据 JSON
python -m benchmark.dog_policy.cli ...                          # 三档 domain 扫描
```

主图：**模型阶数的归一化残差能量（对数轴）** × 鲁棒性指标，双轴同图。
**不要用 R²**——实测一阶模型对二阶阶跃响应的 R² = 0.995，画出来是平线，正反都证明不了。
阶数曲线用 **chirp** 数据算，不要用阶跃。

---

## 6. 并行实验（按优先级）

### P0 — 与 Run B 同时发，同一批算力

| 实验 | 配置差异 | 买到什么 | 依赖 |
| --- | --- | --- | --- |
| **基线策略**（原版 robust） | `reward_scales.{ref_tracking,phase_variance,steady_gain,domain_consistency} = 0`，**其余全同**（课程仍 enabled，所以随机化/扰动时间曲线一致） | R9 每一张图的对照轴。没有它，"更简单的模型能描述到同等精度、且鲁棒性未退化"这句话无从比较 | 标定后（保证两者观测里的 ξ 一致） |
| **Run B 第 2 个 seed** | 仅 `--seed` | 区分"阶段 3 崩了"和"seed 运气差" | 同 Run B |

**注意**：旧的 40000 iter checkpoint（`runs/2026-08-31/*`）**不能当基线**——
obs 是 90 维、命令空间也不是 R1 裁剪后的，加载会直接报错。基线必须重训。

### P1 — Run A 的 checkpoint 一到手就能做，不占训练槽

| 实验 | 命令 | 买到什么 |
| --- | --- | --- |
| R3 验收 | `check_response_runtime.py --check r3 --policy <A>` | δ̂ 的谐波结构、相位查表零延迟 vs 低通滞后 |
| R4 幅值测量 | `--check r4 --policy <A>` | R4.2/4.3/R5 的 raw 量级——阶段 3 的权重就是按它定的，策略变了要复核 |
| **基线的模型阶数曲线** | `export_identification_data.py` + `response/analysis.py` | 主图对照侧的数据**现在就能拿到**，不用等 Run B |
| roll 冻结的辩护 | 同上数据算 `δ(φ)` 幅值 ÷ roll 命令范围 | R9.1 点名要的、支撑 v1 冻结 roll 的那个比值 |

### P2 — 消融，Run B 之后

| 消融 | 怎么做 | 回答什么问题 |
| --- | --- | --- |
| **nominal twin vs 组内均值方差** | 需小改 R5 的对齐目标 | R5 的核心设计断言：组均值方差有退化解（在所有 domain 下同样迟钝），审稿人必问 |
| **去趋势的必要性** | 验收项**不需要额外跑**：`perf_ref_mae_*` 与 `perf_ref_mae_detrended_*` 两条曲线本来就是同一次跑里的对照。若要的是"不去趋势训出来的策略更差"，那需要一个小补丁（`num_phase_bins` 最小值是 2，配置改不出"关闭"） | R4 的验收项：去趋势后误差信号的方差显著更小 |
| **权重前沿** | 4–6 组 `reward_scales.{phase_variance,steady_gain,domain_consistency}` 倍数 | R8.1 要的整条帕累托前沿 |
| **body roll 解冻** | `commands.body_roll_range` 恢复 | R1 留的那个消融 |

---

## 7. 依据这次决定的实测数据

`runs/2026-08-31/stage1_rlmpc_r6_232816`（40000 iter，R6 期，全随机化）：

```text
   iter  rew_track   vx_mae   ep_len  early_term
    500    15.9761    0.125   1002.0      0.005
   2000    17.1100    0.133    992.3      0.029
   5000    15.8525    0.142    940.7      0.125
  20000    16.8143    0.161   1002.0      0.000
  39000    17.3295    0.116    982.6      0.055
```

locomotion 在 ~2000 iter 收敛，之后 37000 iter 基本没有变化。
R8 阶段 1 的退出判据是"跟踪奖励收敛、平地稳定"，不是 iteration 预算。

---

## 8. 已知的坑与未决项

- `global_switch.count` 无 offset（§4）——续跑必踩。
- **`dog.add_obs_noise = False`**：策略在"速度估计完美"的假设下训练。仿真内不影响本文
  结论，但**真机前**必须打开噪声，并给 `base_lin_vel` 加慢漂移偏置随机化（白噪声不够，
  EMA 类统计量对偏置才敏感）。
- R7.2 的 (g, l) 观测只覆盖 `wyaw` / `pitch`，vx/vy/height 的域辨识完全靠 1.0 s 历史。
  R9 的分通道离散度会暴露这个代价；速度通道不达标时优先换 TCN（drop-in 已就位），
  而不是重开 teacher-student。
- pitch/height 的课程 bin 从第 0 iter 起全区间激活——**已决定接受**，不是欠账
  （见 r0-plan 风险表 R11 行）。
