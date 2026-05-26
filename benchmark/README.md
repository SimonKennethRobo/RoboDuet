# Benchmark Configs

这个目录存放 benchmark 运行配置和长期保留的 candidate runs。

## Candidate Runs

Candidate 使用和 `runs/<date>/<run_name>` 相同的目录结构：

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

加入 candidate 的推荐方式是把需要保留的 run 复制到 `benchmark/candidates/<date>/<run_name>`。该目录下有局部 `.gitignore`，默认忽略复制进来的无关训练产物，只 track benchmark 需要的最小文件集：

- `parameters.pkl`
- `params.txt`
- `checkpoints_dog/ac_weights_*_dog.pt`
- `checkpoints_arm/ac_weights_*_arm.pt`
- 可选 `README.md` / `candidate.json`

不要使用 symlink 指向 `runs/`。`runs/` 已被仓库全局 ignore，symlink 不能可靠表达需要长期保留的 candidate 内容。

## Run Dog-Only Benchmark

当前已实现的是 dog-only benchmark：

```bash
python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/smoke.json \
  --sim_device cuda:0
```

旧入口 `python scripts/benchmark_policy.py ...` 仍然保留为兼容 wrapper。

统一入口 `benchmark.cli` 负责选择 benchmark 模式。当前 `--dog_only` 已实现，`--candidate_dir` 会递归扫描所有包含 `parameters.pkl` 的 run-like 目录，并只选择包含 `checkpoints_dog/` 的 candidate。

未来模式：

- `--arm_only`: 只评估 arm policy，要求 candidate 有 `checkpoints_arm/`。
- `--hybrid`: 评估 dog + arm pair，要求 candidate 同时有 `checkpoints_dog/` 和 `checkpoints_arm/`。

这两个模式的统一入口参数已预留，但当前尚未实现。

## Compatibility

当前 dog-only benchmark 会把所有 candidate 放进同一个 IsaacGym simulation 里并行运行。因此，所有被扫描到的 active candidates 必须共享相同的 observation/action/command layout。

如果两个 checkpoint 在关键配置上不同，例如 dog observation 维度、arm command 维度或 `use_rot6d`，它们就不能同时参与同一个 shared benchmark run。短期做法是把不兼容 candidate 放到不同 candidate root，分别运行；长期可以实现自动兼容性分组。

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
