# Benchmark Configs

这个目录存放 `scripts/benchmark_policy.py` 使用的 benchmark 配置。

## Candidate Manifest

默认 candidate manifest:

```bash
benchmark/candidates.json
```

运行当前激活的 dog-only candidates:

```bash
python scripts/benchmark_policy.py \
  --candidates benchmark/candidates.json \
  --profile benchmark/profiles/smoke.json \
  --sim_device cuda:0
```

manifest 现在只实际用于 dog-only benchmark，但 schema 已经按 policy bundle 设计：

```json
{
  "benchmark_type": "dog_only",
  "policies": {
    "dog": {
      "logdir": "ckpts/stage1_0525_110431",
      "ckptid": "last"
    },
    "arm": null
  }
}
```

这样未来扩展到 `arm_only` 和 `hybrid` benchmark 时，不需要推翻 candidate schema。

## Compatibility

当前 policy benchmark 会把所有 active dog candidates 放进同一个 IsaacGym simulation 里并行运行。因此，所有 active candidates 必须共享相同的 observation/action/command layout。

如果两个 checkpoint 在关键配置上不同，例如 dog observation 维度、arm command 维度或 `use_rot6d`，它们就不能同时作为 active candidates 参与同一个 shared benchmark run。可以先保留在 manifest 里并设置 `"active": false`，等后续实现按兼容性自动分组后再一起管理。

## Profiles

默认 smoke profile:

```bash
benchmark/profiles/smoke.json
```

smoke profile 刻意保持较小规模，用来快速验证 candidate 加载、核心 scenario 和 metric 输出是否正常。完整 benchmark 后续应放到 nightly/full profile 中。
