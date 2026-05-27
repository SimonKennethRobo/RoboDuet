# RoboDuet Benchmark Roadmap

这个文档记录 benchmark 工具当前状态、近期已经完成的设计决策，以及后续开发方向。

## Current Status

当前 benchmark 以 `benchmark.cli` 作为统一入口，已实现 `--dog_only` 模式。Candidate 使用 run-like logdir 结构存放在 `benchmark/candidates/`，保留 `parameters.pkl`、`params.txt`、`checkpoints_dog/` 和未来需要的 `checkpoints_arm/`。

Dog-only benchmark 当前支持：

- 多 candidate 在同一个 IsaacGym simulation 中并行评估。
- Velocity Grid、Arm Disturbance Sweep、Body Pose Tracking、Gait Tracking 四组 scenario。
- 固定 benchmark seed，和训练 seed 解耦。
- 输出 `results.json`、`metadata.json` 和 `index.html`。
- HTML report 中展示 summary cards、scenario mean table、metadata、scenario metric tables、heatmap 和交互式 SVG charts。
- 左侧 Results 导航可以在历史 result report 之间切换，并保持页面滚动位置。
- `--inspect` 可轻量检查 dog/arm checkpoint shape 与配置是否自洽。
- `--compare_results` 可直接比较两个已落盘 result，不重新加载 ckpt 或启动 IsaacGym。

## Design Decisions

### Candidate Layout

Candidate 目录保持接近训练 run 的结构，而不是引入 dog-only 专用目录。这样未来同一个 candidate 可以同时包含 dog policy 和 arm policy：

```text
benchmark/candidates/<date>/<run_name>/
  parameters.pkl
  params.txt
  checkpoints_dog/
  checkpoints_arm/
```

Benchmark mode 决定使用哪些 checkpoint：

- `--dog_only`: 只要求 `checkpoints_dog/`。
- `--arm_only`: 未来只要求 `checkpoints_arm/`。
- `--hybrid`: 未来要求 dog 和 arm checkpoint 都存在。

### Report Artifacts

HTML report 已经覆盖旧 `report.md` 的信息，并额外提供 metadata panel、交互图表、表格排序、heatmap 和历史 result 切换。因此默认不再生成 `report.md` 和 `plots/*.png`。

当前默认产物为：

```text
benchmark/results/<timestamp>/
  index.html
  results.json
  metadata.json
```

`results.json` 和 `metadata.json` 作为机器可读产物保留；HTML report 作为主要人工查看入口。

### Metadata

`metadata.json` 用于长期追踪和复现实验。它记录 benchmark protocol、命令行、candidate、ckpt id、seed、env 数量、robot、sim device、git commit/branch/dirty 状态以及 Python/PyTorch/CUDA runtime 信息。

未来如果 policy 结构发生不兼容变化，应优先通过 metadata 找到对应 commit，再 checkout 历史代码复现实验，而不是让当前 benchmark loader 强行兼容所有旧结构。

## Next Steps

### Result Comparison

当前 `--compare_results` 已能比较两个 result 的 summary-level primary metrics。后续可以继续扩展：

- 在主 report 中加入 result-to-result 对比入口。
- 支持 scenario-level 和 test-point-level diff。
- 对 metric direction 做更细的配置，例如 reward 越高越好、RMSE 越低越好。
- 在 HTML 中标注重要 result，例如 baseline、candidate、promoted、deprecated。

### Arm-Only And Hybrid Benchmark

项目最终是 dog policy + arm policy 的双 policy 设计。后续 benchmark 需要支持：

- `--arm_only`: 在已有 dog behavior 固定或受控的前提下评估 arm policy。
- `--hybrid`: 同时加载 dog policy 和 arm policy，评估协同表现。
- Candidate 扫描时按 mode 自动过滤 checkpoint 完整性。
- 对 dog/arm policy 分别记录模型结构、输入输出维度和 ckpt id。

在 arm policy 还不稳定之前，dog-only benchmark 仍然是主路径。

### Parallel Benchmark Efficiency

当前 dog-only benchmark 已经支持把多个 candidate 放进一个 shared simulation 并行推理。后续可以进一步优化：

- 按兼容 layout 自动分组，避免不兼容 candidate 阻塞整批 benchmark。
- 把 `N candidates x M scenarios` 更系统地映射为 batched env。
- 增加 full/nightly profile，用更大 env 数和更长 eval steps 生成稳定结果。

### CI And Server Integration

GitHub-hosted CI 很难直接跑 IsaacGym。更现实的方向是：

- GitHub PR 或 Jenkins job 触发内部 GPU 服务器执行 benchmark。
- 服务器保存 `benchmark/results/`，并暴露静态 HTML report。
- CI 至少检查 benchmark CLI、HTML generation、metadata/result schema 和 inspect 工具是否能运行。

### Report UX

当前 report 已能满足 dog-only 日常查看。后续可以优化：

- 多 result 对比视图。
- Result annotation 和备注。
- 统一 metric schema，减少 report 侧对字段名的硬编码。
- 更丰富的 scenario visualizations，例如 heatmap、scatter、distribution。
