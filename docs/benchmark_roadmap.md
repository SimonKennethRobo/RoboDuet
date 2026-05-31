# RoboDuet Benchmark Roadmap

本文记录当前 `feat/benchmark` 分支中 benchmark 工具的现状、优缺点，以及后续建设方向。目标不是只增加几个 metric，而是把 benchmark 做成一个长期可维护、可比较、可视化、可自动化的评估系统。

## 目标

更好的 benchmark 应该同时满足以下要求：

- metric 足够全面，能覆盖 tracking、稳定性、鲁棒性、步态、身体姿态、资源占用等关键维度。
- 运行足够高效，充分利用 IsaacGym 的多 env 并行能力。
- 输出结果直观，能快速比较不同实验、不同 checkpoint、不同 scenario 的性能差异。
- benchmark 本身可维护，scenario、candidate checkpoint、report、runner 之间职责清晰。
- benchmark seed 独立于训练 seed，用于保证评估采样和环境 reset 尽可能可复现。
- candidate 使用 run-like logdir 结构保存，benchmark mode 按 dog-only、arm-only、hybrid 命名，不沿用训练阶段名。
- 未来可以接入服务器上的 CI/CD 或 Jenkins，作为持续回归测试和 candidate promotion 流程的一部分。

## 当前实现概览

当前 `scripts/` 下和 benchmark 直接相关的脚本主要有三个：

| 文件 | 作用 |
| --- | --- |
| `benchmark/cli.py` | 统一 benchmark 入口，负责选择 `dog_only`、`arm_only` 或 `hybrid` 模式。 |
| `benchmark/dog_policy/cli.py` | dog-only policy 质量评估入口，支持多个 policy 在同一个 IsaacGym simulation 中并行评估。 |
| `benchmark/dog_policy/evaluation.py` | dog-only evaluation 实现，包括 policy/env 加载、配置兼容性检查、metric 累积和结构化结果保存。 |
| `benchmark/env_fps.py` | 环境吞吐 benchmark，用不同 env count 测量 FPS、耗时和显存。 |

`benchmark/dog_policy/cli.py` 当前定位是 dog-only policy benchmark。它把多个候选 policy 放到同一个共享 simulation 中运行：

```text
total_envs = num_envs_per_policy * num_policies
```

每个 policy 拥有一段连续的 env slice。例如 3 个 policy、每个 policy 32 个 env，则总共创建 96 个 env：

```text
policy A -> env [0:32)
policy B -> env [32:64)
policy C -> env [64:96)
```

每一步中，脚本分别对每个 env slice 调对应 policy，然后一次 `env.step()` 推进全部 env。这个设计已经利用了 IsaacGym 的 GPU 并行能力，比逐个 policy 串行启动 simulation 更高效。

需要提前注意的是，RoboDuet 本身是双 policy 设计：

```text
dog_policy + arm_policy
```

RoboDuet 的训练流程会分阶段：先训练 dog policy，再训练 arm policy，最后训练或评估两者协同。但 benchmark 分类不应该沿用 `stage1` 这类训练阶段名，否则语义容易混淆。benchmark 应该按被评估对象命名，例如 dog-only、arm-only、hybrid。

当前 benchmark 只覆盖了 dog-only policy。它还没有真正评估 arm policy，也没有评估 dog policy 和 arm policy 在 hybrid 设置下共同工作的效果。虽然现阶段 arm policy 仍在实验中，还没有稳定可用的 candidate，但 benchmark 的数据结构和代码结构不应该被设计成“永远只有一个 dog checkpoint”。未来 candidate 需要能表达：

- dog-only candidate
- arm-only candidate
- dog + arm pair candidate
- dog 固定、只比较 arm
- arm 固定、只比较 dog
- dog 和 arm 同时变化的 hybrid candidate

当前 `benchmark/candidates/` 中的 candidate 采用和训练输出相同的 run-like logdir 结构：

```text
benchmark/candidates/<date>/<run_name>/
  parameters.pkl
  params.txt
  checkpoints_dog/ac_weights_*.pt
  checkpoints_arm/ac_weights_*.pt  # 未来 hybrid/arm-only 使用
```

目录名里的 `stage1` 只表示原始训练 run 名，不表示 benchmark mode。benchmark mode 应该使用 dog-only、arm-only、hybrid。

同一次 dog-only shared simulation 只能比较 dog policy layout 兼容的 candidate。旧 checkpoint 和新 checkpoint 可以分别跑 benchmark，但如果 observation/action/history/adaptation 相关维度不同，就应该放到不同 benchmark group 中，不能混跑。

## 当前 Scenario

`benchmark/dog_policy/cli.py` 当前包含四类 scenario：

| Scenario | 内容 | 主要关注点 |
| --- | --- | --- |
| A: Velocity Grid | 扫描 x velocity 和 yaw velocity command | 速度跟踪、yaw 跟踪、fall rate |
| B: Arm-Disturbance Sweep | 固定前进速度，扫描 arm disturbance intensity | 抗 arm 干扰能力、稳定性 |
| C: Body-Pose Tracking | 扫描 body pitch、roll、height delta | 身体姿态跟踪和姿态稳定性 |
| D: Gait-Parameter Tracking | 扫描 gait frequency、footswing height、stance width | 步态接触、摆腿高度、Raibert foot placement |

当前 scenario 是写死在 Python 代码中的，例如 velocity grid、arm intensity sweep、body pose sweep、gait sweep 都直接定义在 `benchmark/dog_policy/cli.py` 顶层常量里。

## 当前 Metric

当前 policy benchmark 已经统计了较多 metric，包括：

- velocity tracking:
  - `lin_vel_x_rmse`
  - `lin_vel_y_rmse`
  - `ang_vel_yaw_rmse`
  - `tracking_lin_vel_reward`
  - `tracking_ang_vel_reward`
- stability:
  - `fall_rate`
  - `base_height_mean`
  - `base_height_std`
  - `roll_deg_rms`
  - `pitch_deg_rms`
  - `max_torque_mean`
- body pose:
  - `pitch_rmse_deg`
  - `roll_rmse_deg`
  - `height_rmse_m`
  - `orientation_control_rmse`
- gait:
  - `gait_contact_force_cost`
  - `gait_contact_vel_cost`
  - `foot_clearance_rmse_m`
  - `raibert_rmse_m`

这些 metric 大多在 GPU tensor 上累积，只在 scenario 结束时汇总成 Python float，整体方向是正确的。

## 当前输出

一次 policy benchmark 会输出到：

```text
benchmark/results/<timestamp>/
  index.html
  results.json
  metadata.json
```

当前输出形式包括：

- terminal comparison table
- 结构化 `results.json`
- 结构化 `metadata.json`，记录 benchmark protocol、命令行、candidate path、ckpt id、robot、seed、env count、git branch/commit/dirty 状态、Python/PyTorch/CUDA runtime 等运行上下文
- 单文件 HTML report：`index.html`
- `benchmark/results/index.html`，直接渲染最新 result，并在左侧 `Results` 导航中切换或勾选历史 result 做原地对比

当前默认不再生成 `report.md` 和 `plots/*.png`。HTML report 已经覆盖旧 markdown 信息，并额外提供 summary cards、metadata panel、scenario mean table、scenario detail tables、heatmap、表格排序、交互式 SVG charts 和原地多 result 对比。`results.json` 和 `metadata.json` 作为机器可读 artifact 保留，HTML report 作为主要人工查看入口。

当前每个 HTML report 已经支持浏览器内多 result 对比，不需要单独运行 compare 命令；Summary 和 scenario detail 在多选时使用 metric-grouped 宽表，result 作为 sub-column，支持 sub-column 排序。`benchmark.cli --compare_results` 仍保留为生成独立两两 compare artifact 的离线入口。后续仍值得继续做 regression verdict、result annotation 和更稳定的 schema/direction 配置。

这些结果已经覆盖 dog-only benchmark 的日常使用。当前主要缺口是更深入的跨 result 交互式对比，例如两个不同 benchmark run 之间的 side-by-side scenario table、delta chart、test-point-level regression verdict 和 candidate promotion / retirement 记录。

## 当前优点

### 1. 多 policy 共享一个 IsaacGym simulation

这是当前实现最重要的优点。多个 policy 并不是逐个启动 simulation，而是在一个 env pool 中并行执行。这样 expensive GPU step 只需要为每个 scenario point 执行一次。

这个设计应该保留，并作为未来 benchmark runner 的核心能力。

### 2. 有配置兼容性检查

`benchmark/dog_policy/evaluation.py` 中的 `validate_shared_env_compatibility()` 会检查多个 checkpoint 是否共享关键 observation/control layout，例如：

- dog command 维度
- dog observation history 维度
- action 维度
- trajectory / dynamic gait / rot6d 等关键配置

这很重要，因为只有 observation/control layout 兼容的 checkpoint 才能安全放入同一个 shared env 中运行。

### 3. Metric 与训练目标有一定一致性

当前 gait metric 并不是只看简单命令误差，而是复用了训练中接触、clearance、Raibert foot placement 相关的 cost 思路。这使得 benchmark 更贴近 policy 实际优化目标。

### 4. 已有 JSON 结果

`results.json` 是未来 HTML dashboard、历史对比、CI regression check 的基础。后续应该继续把 JSON schema 稳定下来，而不是只依赖 terminal 输出。

### 5. Env FPS benchmark 单独存在

`benchmark/env_fps.py` 把 simulation throughput benchmark 和 policy quality benchmark 分开，是合理的。policy benchmark 回答“哪个 checkpoint 更好”，env FPS benchmark 回答“当前环境配置能跑多快、显存占用如何”。

## 当前缺点

### 1. Candidate checkpoint 没有一等公民设计

当前使用方式依赖手写命令：

```bash
python -m benchmark.cli \
  --dog_only \
  --logdirs runs/run_A runs/run_B \
  --names A B \
  --ckptids last 15000
```

这不适合长期维护一个 candidate pool。未来我们会有一些 checkpoint 在某些 metric 上表现好，需要保留一段时间继续参与 benchmark；如果新 checkpoint 全面优于旧 checkpoint，旧 checkpoint 应该退出 candidate 集合。

当前代码没有记录：

- 为什么某个 ckpt 被选为 candidate。
- 它在哪些 scenario 上强。
- 它是否已经被新 ckpt dominate。
- 它的保留期限或淘汰条件。
- 它对应的 robot、stage、command layout、训练 commit。
- 它是 dog-only、arm-only，还是 dog+arm policy pair。
- 如果是 policy pair，dog checkpoint 和 arm checkpoint 分别来自哪里、是否被一起训练、是否只是临时组合。

### 2. Scenario 配置写死在代码中

scenario grid 和 metric selection 都写在 Python 文件里。这样导致：

- 想跑 smoke benchmark、nightly benchmark、full benchmark 时不方便切换。
- 想临时加一个 stress case 必须改代码。
- CI/Jenkins 很难根据配置触发不同 benchmark profile。
- scenario 本身不容易版本化和 review。

未来更适合把 scenario 定义放进 YAML/TOML/JSON 配置中，Python runner 只负责执行。当前已经有 `benchmark/profiles/smoke.json` 这类 profile，用于控制 seed、env 数、step 数和启用哪些 scenario；后续可以继续把 scenario grid 本身也配置化。

### 3. 输出不够适合快速比较

当前 HTML report 已经可以覆盖单次 result 的浏览和检查。单个 result 内可以查看 summary、metadata、scenario mean、scenario detail table 和交互式 SVG chart，并支持表格排序、heatmap、左侧 result 切换和滚动位置保持。

目前已经有两层 compare 能力：

- HTML report 左侧 `Results` 导航：交互式多 result 对比，支持多选 result、metadata diff、metric-grouped 宽表、sub-column 排序和 multi-series chart，并保留原 Report 层级。
- `benchmark.cli --compare_results`：独立两两 compare artifact，适合 CI 或需要保存单独 HTML 文件的场景。

CLI compare 示例：

```bash
python -m benchmark.cli --compare_results \
  --baseline benchmark/results/A \
  --target benchmark/results/B
```

后续 compare/report 应该继续补强：

- 不同 result 的 metadata 对齐。
- candidate / scenario / metric summary 对齐。
- baseline vs candidate 的 delta table。
- 关键 metric bar chart。
- regression verdict。

### 4. 缺少综合评分和 regression 判断

当前结果只是列出原始 metric，没有明确回答：

- 哪个 checkpoint overall 最好。
- 哪个 checkpoint 相比 baseline 明显退化。
- 某个 checkpoint 是否可以 promotion 到 candidate pool。
- 某个旧 candidate 是否被新 candidate 全面击败。

未来需要引入 benchmark score 或至少 benchmark verdict。

示例：

```text
candidate_A vs baseline:
  velocity tracking: +8.2%
  fall rate: no regression
  gait cost: -3.1%
  torque: +5.4% worse
  verdict: keep, needs review
```

### 5. 代码组织仍然偏脚本化

当前 `benchmark/dog_policy/evaluation.py` 已经承担很多职责：

- 读取配置
- 加载 env
- 加载 policy
- 设置 commands
- 运行 eval loop
- 累积 metric
- 转换结果
- 打印 terminal table
- 保存 markdown
- 保存 plot
- 保存 JSON

短期可接受，但如果继续加入 candidate 管理、HTML report、CI profiles、scenario config、历史比较，这个文件会变得难维护。

### 6. 并行化还有进一步空间

当前已经做到多个 policy 并行，但 scenario point 仍然是外层循环逐个跑。

未来如果有：

```text
N candidates
M scenario points
K seeds / env repeats
```

理论上可以规划成更明确的 batch：

```text
total_envs = N * M * K
```

这样一个 simulation run 可以同时覆盖多个 checkpoint 和多个 scenario point。限制是：这些 candidate 必须共享 observation/control layout，且 scenario command 能在 env slice 级别设置。

### 7. Arm / Hybrid benchmark 尚未实现

当前 `--stage2` CLI 参数是 reserved，还没有真正实现 hybrid evaluation。未来 RoboDuet 的核心能力不仅是 dog-only locomotion，还需要评估 arm + dog coordination。

Arm / hybrid benchmark 需要考虑：

- arm trajectory tracking。
- end-effector pose error。
- loco stability under arm motion。
- dog-arm coordination。
- task success rate。
- hybrid switch / global_switch 相关状态。

这部分需要特别注意双 policy 组合带来的 benchmark 设计问题。一个 hybrid 结果不只属于某个单独 checkpoint，而是属于一个 policy bundle：

```text
benchmark unit = dog_policy + arm_policy + env/scenario config
```

因此未来 benchmark 不能只把 `logdir + ckptid` 当作唯一 candidate。更合理的抽象是 `PolicyBundle`：

```text
PolicyBundle:
  dog_policy: optional
  arm_policy: optional
  benchmark_mode: dog_only | arm_only | hybrid
  compatibility: observation/action/command layout
```

这样可以支持几类不同实验：

| Benchmark 类型 | dog_policy | arm_policy | 目的 |
| --- | --- | --- | --- |
| dog-only | candidate | none/fake arm | 比较狗本体 locomotion 能力 |
| arm-only | fixed or scripted | candidate | 比较机械臂 trajectory/task 能力 |
| fixed-dog arm comparison | fixed baseline | candidate set | 在相同 dog 下比较 arm policy |
| fixed-arm dog comparison | candidate set | fixed baseline | 在相同 arm 下比较 dog policy |
| hybrid pair comparison | candidate pair | candidate pair | 比较完整 RoboDuet 协同能力 |

当前 benchmark 可以继续保持 dog-only，但未来代码结构需要允许从 dog-only 平滑扩展到 policy bundle。

## Candidate Checkpoint 管理建议

建议使用 run-like candidate 目录，而不是维护全局 candidate manifest。这样可以直接复用训练产物的目录结构，加入 candidate 的成本最低。

```text
benchmark/
  candidates/
    .gitignore
    2026-05-25/
      stage1_0525_110431/
        parameters.pkl
        params.txt
        checkpoints_dog/
          ac_weights_last_dog.pt
        checkpoints_arm/
          ac_weights_last_arm.pt
```

`benchmark/candidates/.gitignore` 应该使用 allowlist，默认忽略复制进来的训练产物，只保留 benchmark 需要的最小文件集。这样可以直接把 `runs/<date>/<run_name>` 复制进 `benchmark/candidates/`，但不会把 logs、wandb、视频、完整脚本快照等无关文件都提交进 Git。

Candidate 本身不需要声明为 `dog_only`、`arm_only` 或 `hybrid`。这些是“本次 benchmark 的模式”，应该通过 CLI 控制：

```bash
python -m benchmark.cli \
  --dog_only \
  --candidate_dir benchmark/candidates \
  --profile benchmark/profiles/smoke.json
```

`--dog_only` 只要求 candidate 目录中存在 `checkpoints_dog/`；`--arm_only` 未来只要求 `checkpoints_arm/`；`--hybrid` 未来要求两者都存在。

如果将来需要记录 `baseline`、`tags`、保留原因、淘汰原因等信息，可以在每个 candidate 目录下放可选 `candidate.json`。这比全局 manifest 更不容易和真实文件结构脱节。

未来 candidate pool 应该支持：

- `active`: 默认参与 benchmark。
- `baseline`: 作为 regression 对比基准。
- `archived`: 不默认参与，但保留记录。
- `rejected`: 曾参与但被淘汰。

候选淘汰不应该只看单个 metric，而应该基于多维 criteria：

- 如果新 ckpt 在所有核心 metric 上不差于旧 ckpt，并且至少一个关键 metric 明显更好，则旧 ckpt 可以 retire。
- 如果旧 ckpt 在某个特殊 scenario 上仍然强，例如极端 yaw 或 arm disturbance，则应该保留并标记 specialty。
- 如果新 ckpt 表现更强但 torque 或 fall rate regression 明显，需要人工 review。

## HTML Dashboard 方向

建议把 HTML report 作为下一阶段最优先的用户体验改进。

理想输出：

```text
benchmark/results/<timestamp>/
  results.json
  metadata.json
  index.html
benchmark/results/index.html
```

HTML dashboard 应该包含：

- Summary
  - candidate cards
  - scenario-level metric mean table
  - baseline comparison
  - pass/fail/regression verdict
- Scenario tabs
  - velocity grid
  - arm disturbance
  - body pose
  - gait
- Metric heatmap
  - 行是 candidate
  - 列是 scenario / metric
  - 颜色表示 better/worse
- Interactive table
  - 排序
  - 过滤
  - 只看 regression
  - 只看 fall case
- Per-candidate detail
  - metadata
  - ckpt path
  - command layout
  - metric summary
- Baseline diff
  - absolute value
  - relative change
  - threshold status

当前实现已经使用静态 HTML + embedded JSON / data attributes，不需要引入服务器；图表直接基于 `results.json` 渲染 SVG，不再依赖 PNG plot。后续如果 dashboard 复杂，再考虑 React/Vite 或轻量前端框架。

## 代码结构重构建议

当前 benchmark 入口已经从 `scripts/benchmark_*.py` 迁移到 `benchmark/` package，`scripts/` 下只保留兼容 wrapper。后续可以继续把大文件拆成更细模块：

```text
benchmark/
  __init__.py
  cli.py
  candidates.py
  config.py
  schemas.py
  compatibility.py
  dog_policy/
    __init__.py
    cli.py
    evaluation.py
    scenarios.py
    metrics.py
    reports.py
  reports/
    __init__.py
    json_report.py
    markdown_report.py
    html_report.py
  profiles/
    smoke.json
    nightly.json
    full.json
```

职责划分：

| 模块 | 职责 |
| --- | --- |
| `cli.py` | 统一 benchmark 模式选择和顶层调度。 |
| `candidates.py` | 扫描 run-like candidate 目录，解析 candidate logdir、checkpoint 和可选 metadata。 |
| `dog_policy/cli.py` | dog-only benchmark CLI 和 profile/candidate 参数解析。 |
| `dog_policy/evaluation.py` | 当前 dog-only benchmark 的运行、加载、metric、report 实现。 |
| `dog_policy/scenarios.py` | 未来拆出 scenario point 和 command setter。 |
| `dog_policy/metrics.py` | 未来拆出 metric accumulation 和 summary。 |
| `dog_policy/reports.py` | 未来拆出 dog-policy 专用 report 生成逻辑。 |
| `schemas.py` | 统一结果数据结构和 JSON schema。 |
| `reports/html_report.py` | 生成 HTML dashboard。 |
| `profiles/*.json` | 定义 smoke/nightly/full benchmark profile。 |

迁移不需要一次性重写；当前已经保留 `scripts/benchmark_policy.py` 作为兼容 wrapper，内部调用 `benchmark.cli`。

## 并行化演进方向

### 当前并行模型

当前模型是：

```text
for scenario_point in scenario:
    total_envs = num_policies * envs_per_policy
    run all policies in shared env
```

优点是简单、稳定、容易解释。

### 未来并行模型

未来可以演进为：

```text
total_envs = num_candidates * num_scenario_points * num_repeats
```

每个 env slice 映射到：

```text
(candidate, scenario_point, repeat)
```

这样一次 rollout 就能覆盖多个 scenario point。

需要注意的限制：

- 所有 candidate 必须共享 observation/action/command layout。
- 不同 scenario point 的 command 必须能按 env slice 设置，而不是全局设置。
- reset、fall、metric accumulation 必须保留 `(candidate, scenario, repeat)` 维度。
- 如果某些 scenario 需要不同 env config，则仍然必须分批跑。

更现实的中间形态：

```text
for scenario_group in compatible_scenario_groups:
    batch scenario points inside one env pool
```

也就是先按配置兼容性分组，再在组内最大化并行。

## CI/CD 与服务器 Benchmark

GitHub-hosted runner 通常不适合跑 IsaacGym benchmark，因为它需要：

- NVIDIA GPU
- IsaacGym 安装
- 正确的 NVIDIA driver / CUDA / PyTorch 环境
- 大量显存和较长运行时间

更现实的方案是：

```text
GitHub PR / comment / label
  -> webhook or Jenkins trigger
  -> benchmark server pulls branch
  -> conda env runs benchmark
  -> uploads results
  -> comments summary back to PR
```

建议拆成不同 profile：

| Profile | 触发方式 | 用途 |
| --- | --- | --- |
| smoke | PR 或手动 | 快速确认 benchmark 能跑、核心 ckpt 不崩。 |
| nightly | 定时 | 跑 active candidates 的完整 scenario。 |
| full | 手动 | 发布前或重要模型比较。 |
| fps | 手动/定时 | 检查 env throughput 和显存回归。 |

Smoke benchmark 可以作为近似 UT：

- benchmark 脚本能 import。
- env 能创建。
- policy 能加载。
- 少量 env、少量 steps 能跑完。
- 输出 JSON schema 正常。

Full benchmark 不应该阻塞所有开发 PR，但可以作为模型相关 PR 的必要检查。

## 已发现的 Bug

经过实际验证（`feat/benchmark-stabilization` 分支），以下 bug 已确认存在：

### Bug 1: ScenarioResult NaN 默认值产生非法 JSON

`ScenarioResult` dataclass 中所有 metric 字段默认为 `float("nan")`。`_acc_to_result()` 根据 `CommandLayout` 只赋值一个 variant（如 `pitch_rmse_deg`），另一个 variant（如 `pitch_deg_rms`）保留 NaN 默认值。`save_results()` 直接 `dataclasses.asdict()` + `json.dump()`，不做 NaN 过滤，输出包含 `NaN` literal — 这是非法 JSON（违反 RFC 8259），会导致标准 JSON parser 解析失败。

当前每次 dog-only benchmark（`dog_num_commands >= 5`）的 `results.json` 都包含以下 NaN 字段：
- `pitch_deg_rms`（当 `has_body_pitch=True` 时未赋值）
- `roll_deg_rms`（当 `has_body_roll=True` 时未赋值）
- `gait_freq_rmse_hz`（死字段，从未被任何代码赋值）

修复方向：在 `save_results()` 中过滤 NaN 字段，或让 `_acc_to_result()` 对所有 variant 都赋值（stability RMS 和 tracking RMSE 同时计算），或移除死字段 `gait_freq_rmse_hz`。

### Bug 2: Accumulator key collision（pitch_deg / roll_deg）

`_eval_loop_parallel()` 中，`add_sq_err("pitch_deg", ...)` 和 `add_val("pitch_deg", ...)` 写同一个 `pitch_deg_sq` tensor。前者存 `(actual - cmd)^2`（tracking error），后者存 `actual^2`（stability RMS 的平方分量）。两者混入同一个 `_sq` tensor，使得 `acc.rmse("pitch_deg")` 既不是 tracking RMSE 也不是 stability RMS，而是两者的混合 — 数值错误但不产生 NaN。

`roll_deg` 同样存在此 collision。

修复方向：使用不同的 accumulator key，例如 `"pitch_deg_track"` 和 `"pitch_deg_raw"`。

### Bug 3: `--inspect` 模式硬依赖 torch

`benchmark/cli.py` 在 `--inspect` 时 import `benchmark.inspect`，后者 import torch。在没有 torch/CUDA 的环境（如 GitHub-hosted runner）中直接报错。`--inspect` 应能至少部分工作（读 config 维度），不应要求 GPU。

### Bug 4: CLI `--output` vs `--output_dir` 参数名不一致

`--compare_results` 模块的 CLI 用 `--output`，dog-policy CLI 用 `--output_dir`。用户从 dog 模式切换到 compare 模式时容易混淆。

### Bug 5: 历史垃圾 result 未清理

`benchmark/results/` 下有 3 个旧 result（`20260526_*`），全部是 1 env + 1 step 的无效测试数据，应删除或标记为 `archived`。

## 推荐路线图

### Phase 1: 文档与候选目录 ✅（已完成）

- ✅ 新增 benchmark roadmap 文档。
- ✅ 引入 `benchmark/candidates/` run-like candidate 目录。
- ✅ 使用 `benchmark/candidates/.gitignore` allowlist 控制 Git 只 track 必要文件。
- ✅ CLI 支持 `--candidate_dir benchmark/candidates`。
- ✅ 保留当前 `--logdirs --ckptids --names` 作为低层接口。
- ✅ Benchmark 模式通过 `--dog_only`、未来的 `--arm_only` / `--hybrid` 控制，而不是写死在 candidate 目录名里。

Phase 1 已完全落地。`--candidate_dir` 自动发现、symlink 去重、allowlist gitignore 均已验证可用。

### Phase 1.5: Bug 修复与 JSON 稳定化（当前优先）

Phase 1 的目录管理已落地，但 benchmark 输出的质量还有结构性问题需要修复，否则后续 dashboard、CI、compare 都会受阻。

- 修复 Bug 1: `save_results()` 过滤 NaN 或让 `_acc_to_result()` 覆盖所有 variant。
- 修复 Bug 2: accumulator key collision（pitch_deg / roll_deg），拆为独立 key。
- 修复 Bug 3: `--inspect` 模式延迟 import torch，无 torch 时至少读 config 维度。
- 修复 Bug 4: 统一 CLI 参数名为 `--output_dir`。
- 清理 Bug 5: 删除 `benchmark/results/` 下 3 个无效历史 result。
- 稳定 `results.json` schema：定义字段必选/可选规则、NaN 处理策略、metric direction 标注。
- 增加 `nightly.json` 和 `full.json` profile。

### Phase 2: HTML Report ✅（已完成）

- ✅ 基于现有 `results.json` 和 `metadata.json` 生成 `index.html`。
- ✅ 自动更新 `benchmark/results/index.html`，并直接渲染最新 result 的完整 report。
- ✅ 先实现静态 HTML，不引入复杂服务。
- ✅ 支持 summary cards、metadata panel、scenario mean table、scenario detail table、heatmap 和交互式 SVG charts。
- ✅ 左侧 Results 导航支持历史 result 切换，并保持页面滚动位置。
- ✅ 默认产物收敛为 `index.html`、`results.json`、`metadata.json`，不再默认生成 markdown 和 PNG plot。
- ✅ 已有基础 `--compare_results`，后续继续扩展为 report 内交互式对比。

Phase 2 已完全落地。HTML report 功能完整，compare report 基础可用。

### Phase 2.5: candidate.json 支持

在 Phase 1 的目录结构基础上，引入可选 `candidate.json` 文件，为每个 candidate 记录元数据和状态。

- 定义 `candidate.json` schema: `status` (active/baseline/archived/rejected)、`tags` (list)、`description`、`reason` (保留/淘汰原因)、`trained_on` (训练 commit/dataset)、`specialty` (擅长的 scenario)、`benchmark_mode` (dog_only/arm_only/hybrid)、`policy_pair` (dog/arm checkpoint 来源和组合关系)。
- `discover_run_logdirs()` 读取 candidate.json（如存在），将 status/tag 信息传入 metadata。
- `_apply_candidate_dir()` 支持 `--status active` 过滤，默认只跑 active candidate。
- `--baseline` CLI 参数指定 baseline candidate name 或 path，在 metadata 中标记。
- candidate.json 为可选文件：没有时 fallback 到当前行为（所有发现的 candidate 默认 active）。

实现难度评估：低。candidate.json schema 简单，读取逻辑只需在 `candidates.py` 中加一个 `load_candidate_meta()` 函数，CLI 只需加一个 status 过滤参数。核心改动在 `candidates.py` 和 `dog_policy/cli.py`，不涉及 eval loop 或 HTML report。预估改动量 ~100 行。

### Phase 3: Benchmark 代码结构整理

- 把 `benchmark/dog_policy/evaluation.py` 拆出 candidate、scenario、metric、report 模块。
- 保留旧 CLI 入口，降低迁移风险。
- 给 JSON 结果定义稳定 schema。
- 引入 `PolicyBundle` 概念，避免 runner 和 result schema 只绑定单个 dog policy。

### Phase 4: Candidate Promotion / Retirement

- 设计 baseline 和 active candidate。
- 支持自动生成 candidate 对比 verdict。
- 支持记录某个 candidate 被保留或淘汰的原因。
- 支持历史 benchmark 结果汇总。

### Phase 5: 更高并行度

- 在保持 shared env 的基础上，支持 scenario point batching。
- 将 env slice 从 policy 维度扩展到 `(candidate, scenario, repeat)`。
- 对不兼容的 candidate 自动分组运行。

### Phase 6: CI/CD Integration

- Jenkins 在 benchmark server 上执行 smoke/nightly/full benchmark。
- PR comment 回填 benchmark summary。
- benchmark failure 或严重 regression 自动标记 PR。

### Phase 7: Arm / Hybrid Benchmark

- 引入 arm-only 和 hybrid benchmark profile。
- 支持 fixed-dog arm comparison 和 fixed-arm dog comparison。
- 支持 dog+arm pair candidate 的完整评估。
- 增加 arm trajectory tracking、end-effector pose error、task success、coordination stability 等 metric。
- 明确 dog-only benchmark 与 arm-only/hybrid benchmark 的结果不可直接混合排名，只能在各自 profile 内比较。

## 短期建议（更新）

Phase 1（候选目录）和 Phase 2（HTML Report）已完全落地。当前最优先的改动是：

1. Bug 修复与 JSON 稳定化（Phase 1.5）
   NaN 产生的非法 JSON 和 accumulator key collision 会影响所有下游（dashboard、CI、compare），必须先修。

2. candidate.json 支持（Phase 2.5）
   难度低、价值高。让 candidate pool 从纯文件发现升级为有状态管理，为后续 promotion/retirement 打基础。

3. compare 扩展到 scenario-level / test-point-level diff
   当前 compare 只看 summary-level 7 个 primary metrics，缺乏细节。

不建议马上做大规模并行重构或 arm/hybrid benchmark。先把数据质量和管理机制补上，再扩展评估能力。

## 当前已完成的增量

截至 `feat/benchmark-stabilization`，以下能力已经落地：

- `benchmark.cli --dog_only` 作为 dog-only benchmark 主入口。
- `benchmark.cli --inspect` 可轻量检查 dog/arm checkpoint shape 与配置是否自洽。
- `benchmark.cli --compare_results` 可比较两个已保存 result 的 summary-level primary metrics。
- candidate discovery 支持多级目录和本地目录 symlink，并用 resolved path 去重。
- benchmark seed 独立于训练 seed。
- 默认 result artifact 收敛为 `index.html`、`results.json`、`metadata.json`。
- `metadata.json` 记录 benchmark protocol、命令行、candidate、ckpt id、robot、seed、env 数量、git/runtime 信息；remote URL 中的 credentials 会被 redaction。
- HTML report 支持 summary cards、scenario metric mean table、metadata panel、scenario detail table、heatmap、表格排序、交互式 SVG charts 和左侧 Results 导航。
- heatmap 对 reward / `*_rew` 这类 higher-is-better metric 做方向处理。
- **Phase 1（候选目录管理）已完成**：`--candidate_dir`、allowlist gitignore、run-like 目录结构均已验证可用。
- **Phase 2（HTML Report）已完成**：单结果 report、多结果 index、compare report 均可用。

当前优先推进的方向：

- **Phase 1.5**: Bug 修复（NaN 非法 JSON、accumulator key collision、inspect torch 依赖、CLI 参数名不一致、垃圾 result 清理）+ JSON schema 稳定化 + nightly/full profile。
- **Phase 2.5**: candidate.json 支持（可选元数据文件、status/tag 过滤、baseline 标记）。
- 把 `--compare_results` 扩展到 scenario-level 和 test-point-level diff。
- 把 metric schema 和 direction 从 report 代码中抽出来，减少字段硬编码。
- 在 arm policy 稳定后实现 `--arm_only` 和 `--hybrid`。
