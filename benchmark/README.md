# Benchmark Configs

这个目录存放 benchmark 运行配置和长期保留的 candidate checkpoints。

更长期的设计记录和后续计划见 [docs/benchmark_roadmap.md](../docs/benchmark_roadmap.md)。

## Candidate Checkpoints

Candidate 使用 run-like logdir 结构。目录名可以来自训练 run 名，但 benchmark mode 不使用 `stage1` 这类训练阶段语义：

```text
benchmark/candidates/
  stage1_0525_110431/
    parameters.pkl
    params.txt
    checkpoints_dog/
      ac_weights_last_dog.pt
    checkpoints_arm/
      ac_weights_last_arm.pt
```

加入 candidate 的推荐方式是把需要保留的 run-like 目录复制到 `benchmark/candidates/<run_name>`。该目录下有局部 `.gitignore`，默认忽略复制进来的无关训练产物，只 track benchmark 需要的最小文件集：

- `parameters.pkl`
- `params.txt`
- `checkpoints_dog/ac_weights_*_dog.pt`
- `checkpoints_arm/ac_weights_*_arm.pt`
- 可选 `README.md` / `candidate.json`

长期保留 candidate 时推荐复制 run-like 目录，而不是只提交 symlink。`runs/` 已被仓库全局 ignore，symlink 不能可靠表达需要长期保留的 candidate 内容。不过本地 benchmark 扫描会 follow 目录 symlink，方便临时把 `benchmark/candidates/<run_name>` 指向本机已有 `runs/...` 做评估。

当前启动前会检查每个 dog-only candidate 至少包含：

- `parameters.pkl`
- `params.txt`
- `checkpoints_dog/ac_weights_<ckptid>.pt`

其中 `--ckptids last` 对应 `checkpoints_dog/ac_weights_last_dog.pt`。

注意：checkpoint 权重通常是本地 benchmark artifact，不建议在普通代码 PR 中随分支上传。需要长期共享的 candidate 可以保留 `parameters.pkl` / `params.txt` 等轻量元数据，真实 `ac_weights_*.pt` 由本机或服务器侧 artifact 目录提供。`benchmark/candidates/.gitignore` 默认忽略未 allowlist 的训练产物；如果临时复制新权重到 candidate 目录，本地 benchmark 会读取它，但发布 PR 前应确认是否真的需要提交该权重。

## Run Dog-Only Benchmark

当前已实现的是 dog-only benchmark：

```bash
python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/smoke.json \
  --sim_device cuda:0
```

常用 smoke 验证命令：

```bash
python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/smoke.json \
  --sim_device cuda:0 \
  --num_eval_steps 20 \
  --seed 7 \
  --skip_b
```

完整 dog-only 对比可以直接扫描 candidate pool。layout 兼容的 policy 会放进同一个
IsaacGym simulation 并行评估，不兼容的 layout 会自动拆组、依次运行并汇总到同一报告：

```bash
conda run -n roboduet python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/dog_policy_standard.json \
  --sim_device cuda:0
```

更慢但更完整的 velocity grid 可以使用：

```bash
conda run -n roboduet python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/dog_policy_full.json \
  --sim_device cuda:0
```

旧入口 `python scripts/benchmark_policy.py ...` 仍然保留为兼容 wrapper。

统一入口 `benchmark.cli` 负责选择 benchmark 模式。当前 `--dog_only` 已实现，`--candidate_dir` 会递归扫描所有包含 `parameters.pkl` 的 run-like 目录，支持多级目录和本地目录 symlink，并只选择包含 `checkpoints_dog/` 的 candidate。

未来模式：

- `--arm_only`: 只评估 arm policy，要求 candidate 有 `checkpoints_arm/`。
- `--wbc`: 评估 dog + arm pair，要求 candidate 同时有 `checkpoints_dog/` 和 `checkpoints_arm/`。

这两个模式的统一入口参数已预留，但当前尚未实现。

## Compatibility

dog-only benchmark 会根据关键 observation/action/command 配置自动计算兼容签名：

- 签名相同的 candidate 共用一个 simulation，并在各自连续的 env slice 上并行推理。
- 签名不同的 candidate 使用独立 simulation，layout group 之间依次创建、运行和释放。
- 所有 group 的结果最终仍写入同一个 `results.json`、`metadata.json` 和 HTML report。

例如 `obs=86`、`obs=90` 和 `obs=99` 的 dog policy 可以放在同一个 candidate
root 中启动；benchmark 会建立三个 layout group。某个 layout 不支持的场景会只对该
group 跳过，例如 `dog_num_commands < 9` 时不运行 gait scenario。
已知历史 layout 会恢复其原始字段语义：旧 86 维 observation 会使用 pre-v3
roll/pitch 与 EE-pose 排布，99 维 trajectory layout 会在 arm joint state 前恢复
9 维 EE pose；不会用简单的补零或截断冒充兼容。

`metadata.json` 中的 `execution_mode`、`num_layout_groups`、
`peak_simultaneous_envs` 和 `layout_groups` 会记录本次实际分组。

## Inspect Policy Bundles

如果拿到一个旧 run 或未知 checkpoint，可以先做轻量检查，不创建 IsaacGym env：

```bash
python -m benchmark.cli \
  --inspect \
  --logdirs /path/to/run_like_logdir \
  --ckptid last
```

也可以扫描整个 candidate root：

```bash
python -m benchmark.cli \
  --inspect \
  --candidate_dir benchmark/candidates
```

inspect 会读取 `parameters.pkl`、`checkpoints_dog/` 和 `checkpoints_arm/`，输出 dog/arm 的 observation、history、privileged obs、action、command、adaptation module 和 checkpoint shape。它用于回答“这个 policy bundle 内部结构是否自洽”，以及“它属于哪一代 policy layout”。它不会保证该 policy 能在当前 env 中直接执行；dog-only benchmark 会在加载权重时再次校验 checkpoint 与其 layout group 环境的维度。arm-only 和 wbc benchmark 仍会执行各自的 compatibility check。

例如旧版 RoboDuet policy 可能显示：

```text
dog:
  cfg: obs=56 hist=1680 priv=2 actions=12 commands=5 adapt=True
  ckpt: adaptation_module=True adaptation_input=1680 actor_input=1682 critic_input=1682 actions=12
  internal check: ok
arm:
  cfg: obs=20 hist=600 priv=9 actions=8 commands=6 adapt=False
  ckpt: adaptation_module=True adaptation_input=600 history_encoder_input=580 actor_input=157 critic_input=157 actions=8
  internal check: mismatch
    - arm adaptation flag cfg=False checkpoint=True
```

这类结果说明 checkpoint 本身可能来自旧版 policy layout，不能直接假设和当前 loader 或当前 env layout 兼容。

## Profiles

默认 smoke profile:

```bash
benchmark/profiles/smoke.json
```

smoke profile 刻意保持较小规模，用来快速验证 candidate 加载、核心 scenario 和 metric 输出是否正常。完整 benchmark 后续应放到 nightly/full profile 中。

Dog-policy profiles:

```text
benchmark/profiles/dog_policy_standard.json
benchmark/profiles/dog_policy_full.json
```

profile 会把 `benchmark_protocol` 记录为 `dog_only`，并在 `metadata.json` 中保存 scenario grid 和固定 command。当前 scenario：

- A Velocity Grid：standard 为 `vx/vy/yaw = [-1, 0, 1]` 的 27 点 grid；full 为 `[-1, -0.5, 0, 0.5, 1]` 的 125 点 grid。固定 gait 为 `freq=4.0, stance_width=0.30, stance_length=0.35, footswing_height=0.06, gait_duration=0.5`，pose 为 0。
- B Arm Disturbance Sweep：固定 `vx=1.0, vy=0.0, yaw=0.0`，扫描 `arm_intensity = [0, 0.25, 0.5, 0.75, 1.0]`，并记录 disturbance seed metadata。
- C Body Pose Tracking：对 `stand/forward/lateral/turn` 四个 velocity group 分别扫描 pitch、roll、height delta，standard 共 56 点。
- D Gait Command Tracking：固定 `vx=0.5`，分别扫描 gait frequency、stance width、stance length，standard 共 18 点。
- E Velocity Step Response（predictable-plant）：从静止对若干速度目标（默认前向 `0.5/1.0/1.5`、侧向 `0.5`、yaw `1.0`、前向+yaw 组合，共 6 点）施加阶跃，度量腿部速度响应对一阶参考模型的贴合度。可用 `scenario_config.vel_step` 覆盖 `targets` 和 `settle_steps`。

### 响应一致性指标（response_consistency_rmse，列名 `resp_cons`）

`response_consistency_rmse` = 实测 base 速度 `base_lin_vel[:2]` 与命令的一阶参考模型 `dog_vel_ref`（`v_ref += (v_cmd - v_ref) * dt / T`）之间的 RMSE（m/s）。它累积与训练 `response_consistency` reward 逐字相同的量，**越低表示腿越接近一个固定时间常数的可预测线性 plant** —— 这是上层解析 base 前馈（`v_ff`）所依赖的性质（见 `project-design-v3.md` §2.3）。该列出现在 A（vel_grid）、B（arm_sweep）、E（vel_step）三个场景中；A/B 在保持命令下反映稳态+瞬态混合，**E 才是纯净的阶跃响应度量**。

**场景 E 的非显然行为**：为得到干净的阶跃，`run_scenario_e` 会**仅对 E** 临时把 `terrain.{z,yaw,pitch,roll}_init_range` 置 0（关闭 reset 的随机落体/翻转，try/finally 恢复，不影响其它场景），并在每个测试点先跑 `settle_steps`（默认 40）步零速命令、不累积，用来阻尼 reset 硬编码的 ±0.5 m/s 初速度、让 `dog_vel_ref` 归零，之后才施加被测阶跃。E 默认仍在 `--arm_intensity`（默认 1.0）下运行，即测"臂扰动下 plant 是否仍可预测"；要测无扰动基线跑 `--arm_intensity 0`。

新增和强化的主要字段包括 `response_consistency_rmse`、`lin_vel_xy_rmse`、`fall_rate_height`、`cmd_stance_length`、`cmd_gait_duration` 和 `stance_length_rmse_m`。legacy/no-profile 结果仍可被 HTML report 打开，缺失字段会显示为 `-`。

包含 stance-length / gait-duration 的 profile gait scenario 要求 runtime `dog_num_commands >= 11`，因为会使用 `stance_length` index 9 和 `gait_duration` index 10。不满足时 benchmark 会 fail fast，而不是静默跳过 gait 子项。`fall_rate_height` 表示由 height terminal 条件触发的 event rate，分母与 `fall_rate` 一样是该测试点累计 env step 数。

Profile 中的 `seed` 是 benchmark 评估 seed，只用于控制评估时的随机采样和环境 reset。它独立于训练 seed，不要求和 candidate checkpoint 的训练 seed 一致。

默认输出目录是：

```text
benchmark/results/<timestamp>/
```

该目录用于本地查看 benchmark 结果，已在仓库 `.gitignore` 中忽略。

## HTML Report

可以从任意已有 `results.json` 生成一个独立 HTML report：

```bash
python -m benchmark.reports.html \
  --results benchmark/results/<timestamp>/results.json
```

默认输出到同目录下的 `index.html`。HTML report 只读取已有结果文件，不改变 benchmark 计算逻辑。

通过 `python -m benchmark.cli --dog_only ...` 正常跑 benchmark 时，会自动生成：

```text
benchmark/results/<timestamp>/
  index.html
  results.json
  metadata.json
```

`metadata.json` 会记录复现实验所需的上下文，包括 benchmark protocol、命令行、candidate logdir、ckpt id、robot、seed、env 数量、git branch/commit/dirty 状态、Python/PyTorch/CUDA 运行时信息等。这样即使未来代码结构变化，也可以根据 metadata 快速 checkout 到对应 commit 复现或追踪结果。

HTML report 直接基于 `results.json` 渲染交互式 SVG charts，不再默认生成 `report.md` 或 `plots/*.png`。生成 report 时会同时维护结果根目录的 `index.html`，它会直接渲染最新 result 的完整 Report 内容：

```text
http://127.0.0.1:8765/
```

历史 result 切换和多 result 对比已经整合到每个 report 的左侧 `Results` 导航栏中。Report 默认从 Home 页面开始，左侧 `Results` 保持 0 勾选状态；点击单个 result 会进入单 report，勾选多个 result 会在原 Report 布局内进入 compare 视图。`Select all` 会选择当前 report 中真实存在的 candidate，`Clear all` 会回到 Home 页面。

多 result 模式下，Summary 和 scenario detail 会合并为按 metric 分组的宽表，每个 result 是 metric 下的 sub-column，并支持 sub-column 排序。表格行头和 metric 分组列头固定在左侧，横向滚动只移动 result metric 区域；metric chart 和 table 支持双向 hover/click 高亮，切换 chart metric 时会自动把对应 metric column 滚动到可见区域。

也可以为整个结果目录批量生成 HTML：

```bash
python -m benchmark.reports.html \
  --results_root benchmark/results
```

HTML report 当前包含：

- summary cards: points、fall rate、xy/yaw RMSE、lin/yaw reward、base height、max torque 的 overall mean
- summary metric-mean table: 按 scenario 汇总主要 metric 均值，overall mean 放在 candidate card 中
- metadata panel: seed、profile、candidate path、ckpt id、env count、git/runtime 信息等
- left navigation: Report sections 和历史 result 切换入口，包含 Summary、Metadata、各 scenario 和 Results
- scenario detail tables: 展示每个测试点的关键指标，并对主要列加 heatmap。新增稳定性 RMS（pitch_deg_rms、roll_deg_rms）、速度 XY RMSE、height fall rate 和步态追踪 RMSE（gait_freq_rmse_hz、stance_width_rmse_m、stance_length_rmse_m）
- interactive metric charts: 每个 scenario 内直接切换 metric，并基于 `results.json` 渲染 SVG 图表；structured Velocity Grid 使用 `yaw` facet heatmap（x=`vx`, y=`vy`），Body Pose Tracking 使用 velocity-group pose summary heatmap 和 pitch/roll/height grouped sweeps，Gait Tracking 使用 gait frequency/stance width/stance length grouped sweeps；完整测试点 label 保留在 hover tooltip 和下方表格中

JSON 结果中的未计算指标序列化为 `null`（不是 NaN），符合 RFC 8259 标准。

## Result Comparison

优先使用任意 result report 左侧 `Results` 导航栏中的多选 compare。它支持一次选择一个或多个 result，并直接在原 Report 布局内查看 summary、metadata、table 和 chart diff。

如果需要生成独立的离线 compare artifact，也可以直接比较两个已经落盘的 benchmark result，不需要重新加载 ckpt 或启动 IsaacGym：

```bash
python -m benchmark.cli --compare_results \
  --baseline benchmark/results/<old_result> \
  --target benchmark/results/<new_result>
```

默认会在 target result 目录下生成：

```text
compare_<old_result>_to_<new_result>.html
```

也可以显式指定输出位置：

```bash
python -m benchmark.cli --compare_results \
  --baseline benchmark/results/<old_result>/results.json \
  --target benchmark/results/<new_result>/results.json \
  --output benchmark/results/compare_old_to_new.html
```

对比报告基于两个 result 的 `results.json` 和 `metadata.json`，当前比较 summary-level primary metrics，包括 vx RMSE、yaw RMSE、fall rate、tracking reward、base height 和 max torque。它不会尝试重新运行旧 ckpt，因此适合长期保留历史 benchmark 结果并做跨时间对比。

## CI Integration

Benchmark CI 分成两层：

- GitHub Actions 静态检查：不依赖 GPU、IsaacGym、torch 或 checkpoint，只检查 benchmark Python 文件可编译、profile JSON 合法、HTML report 和 result comparison 能从离线 fixture 正常生成。
- GPU smoke benchmark：在有 IsaacGym/NVIDIA 环境的服务器上运行，验证 candidate loading、IsaacGym env step、metric accumulation 和 HTML report 全链路。

GitHub Actions workflow 位于：

```text
.github/workflows/benchmark-static.yml
```

它会在 PR 或 `develop` push 涉及 `benchmark/**` / benchmark wrapper 时运行：

```bash
python -m compileall -q benchmark scripts/benchmark_policy.py scripts/benchmark_env_fps.py
python benchmark/ci/static_check.py
```

GPU smoke benchmark 使用服务器侧脚本：

```bash
benchmark/ci/run_gpu_smoke.sh
```

默认命令等价于：

```bash
python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/smoke.json \
  --sim_device cuda:0 \
  --ckptids last \
  --output_dir benchmark/results \
  --headless
```

常用环境变量：

```bash
CANDIDATE_DIR=/data/roboduet/benchmark/candidates \
PROFILE=benchmark/profiles/smoke.json \
SIM_DEVICE=cuda:0 \
OUTPUT_DIR=/data/roboduet/benchmark/results \
NUM_EVAL_STEPS=100 \
SEED=1 \
benchmark/ci/run_gpu_smoke.sh
```

Jenkins 或 GitHub self-hosted runner 可以直接调用这个脚本，并把 `OUTPUT_DIR` 下的 `index.html`、`results.json`、`metadata.json` 和 compare report 作为 artifact 保存。不要在普通 GitHub-hosted runner 上跑 GPU smoke；它没有 IsaacGym、NVIDIA driver 和本地 checkpoint/candidate 环境。
