# Benchmark Configs

这个目录存放 benchmark 运行配置和长期保留的 candidate checkpoints。

更长期的设计记录和后续计划见 [ROADMAP.md](ROADMAP.md)。

## Candidate Checkpoints

Candidate 使用 run-like logdir 结构。目录名可以来自训练 run 名，但 benchmark mode 不使用 `stage1` 这类训练阶段语义：

```text
benchmark/candidates/
  2026-05-25/
    stage1_0525_110431/
      parameters.pkl
      params.txt
      checkpoints_dog/
        ac_weights_last_dog.pt
      checkpoints_arm/
        ac_weights_last_arm.pt
```

加入 candidate 的推荐方式是把需要保留的 run-like 目录复制到 `benchmark/candidates/<date>/<run_name>`。该目录下有局部 `.gitignore`，默认忽略复制进来的无关训练产物，只 track benchmark 需要的最小文件集：

- `parameters.pkl`
- `params.txt`
- `checkpoints_dog/ac_weights_*_dog.pt`
- `checkpoints_arm/ac_weights_*_arm.pt`
- 可选 `README.md` / `candidate.json`

长期保留 candidate 时推荐复制 run-like 目录，而不是只提交 symlink。`runs/` 已被仓库全局 ignore，symlink 不能可靠表达需要长期保留的 candidate 内容。不过本地 benchmark 扫描会 follow 目录 symlink，方便临时把 `benchmark/candidates/<date>/<run_name>` 指向本机已有 `runs/...` 做评估。

当前启动前会检查每个 dog-only candidate 至少包含：

- `parameters.pkl`
- `params.txt`
- `checkpoints_dog/ac_weights_<ckptid>.pt`

其中 `--ckptids last` 对应 `checkpoints_dog/ac_weights_last_dog.pt`。

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

旧入口 `python scripts/benchmark_policy.py ...` 仍然保留为兼容 wrapper。

统一入口 `benchmark.cli` 负责选择 benchmark 模式。当前 `--dog_only` 已实现，`--candidate_dir` 会递归扫描所有包含 `parameters.pkl` 的 run-like 目录，支持多级目录和本地目录 symlink，并只选择包含 `checkpoints_dog/` 的 candidate。

未来模式：

- `--arm_only`: 只评估 arm policy，要求 candidate 有 `checkpoints_arm/`。
- `--hybrid`: 评估 dog + arm pair，要求 candidate 同时有 `checkpoints_dog/` 和 `checkpoints_arm/`。

这两个模式的统一入口参数已预留，但当前尚未实现。

## Compatibility

当前 dog-only benchmark 会把所有 candidate 放进同一个 IsaacGym simulation 里并行运行。因此，所有被扫描到的 active candidates 必须共享相同的 observation/action/command layout。

如果两个 checkpoint 在关键配置上不同，例如 dog observation 维度、arm command 维度或 `use_rot6d`，它们就不能同时参与同一个 shared benchmark run。短期做法是把不兼容 candidate 放到不同 candidate root，分别运行；长期可以实现自动兼容性分组。

例如历史 `runs/...` 里的 dog policy 可能是 `adapt=on`、`obs=83`，而当前 `benchmark/candidates/...` 中的新结构是 `adapt=off`、`obs=86`。这两类 checkpoint 都可以单独 benchmark，但不能混在同一次多 policy shared simulation 中比较。

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

inspect 会读取 `parameters.pkl`、`checkpoints_dog/` 和 `checkpoints_arm/`，输出 dog/arm 的 observation、history、privileged obs、action、command、adaptation module 和 checkpoint shape。它用于回答“这个 policy bundle 内部结构是否自洽”，以及“它属于哪一代 policy layout”。它不会保证该 policy 能在当前 env 中直接执行；真正执行仍然需要通过 dog-only、arm-only 或 hybrid benchmark 的 shared-env compatibility check。

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

历史 result 切换已经整合到每个 report 的左侧 `Results` 导航栏中。

也可以为整个结果目录批量生成 HTML：

```bash
python -m benchmark.reports.html \
  --results_root benchmark/results
```

HTML report 当前包含：

- summary cards: points、fall rate、vx/yaw RMSE、lin/yaw reward、base height、max torque 的 overall mean
- summary metric-mean table: 按 scenario 汇总主要 metric 均值，overall mean 放在 candidate card 中
- metadata panel: seed、profile、candidate path、ckpt id、env count、git/runtime 信息等
- left navigation: Report sections 和历史 result 切换入口，包含 Summary、Metadata、各 scenario 和 Results
- scenario detail tables: 展示每个测试点的关键指标，并对主要列加 heatmap
- interactive metric charts: 每个 scenario 内直接切换 metric，并基于 `results.json` 渲染 SVG 图表；x 轴使用短语义标签，完整测试点 label 保留在 hover tooltip 和下方表格中

## Result Comparison

可以直接比较两个已经落盘的 benchmark result，不需要重新加载 ckpt 或启动 IsaacGym：

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
