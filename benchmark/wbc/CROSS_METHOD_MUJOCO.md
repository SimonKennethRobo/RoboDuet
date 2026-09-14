# Cross-method MuJoCo benchmark

The benchmark owner is `RoboDuet/benchmark/wbc`. Frozen TaskSpec/reference
validation, trace-v3 normalization, scoring, and final receipts are handled by
`cross_method_cli.py`. Method-specific controller processes stay in their
original trees and every consumed source/config/model is recorded by hash.

The first adapter runs qm_control's native SQP-MPC + QP-WBC against the shared
Go2+X5 MuJoCo model. It restores the selected TaskSpec initial state, replays
the inherited time law, runs both nominal and a derived deterministic push
variant, and preserves backend traces before applying the current RoboDuet
protocol metadata and offline scorer.

From the RoboDuet checkout:

```bash
/opt/miniconda3/envs/isaacgym/bin/python -m benchmark.wbc.cross_method_cli \
  --method qm_control \
  --suite benchmark/results/ocs2_full_reference_complete_path_v1/20260914_210621/trajectory_suite.json \
  --task-id timed-trajectory-ac28cbf79041b70f \
  --scenarios nominal push \
  --output benchmark/results/cross_method_mujoco/qm_control \
  --ros-domain-id 91
```

The adapter subprocess uses `/opt/miniconda3/envs/base312/bin/python` by
default because that environment contains the ROS/MuJoCo dependencies used by
the installed qm_control prefix. The launcher gives it a clean environment so
stale ROS overlays in the interactive shell cannot change the run.
The workspace-specific baseline default can be replaced with
`--baseline-root /absolute/path/to/baselines/mpc_baseline`.

Each timestamped output contains `cross_method_manifest.json`, `results.json`,
`adapter.log`, the exported reference, and a nominal/push directory. Scenario
directories contain the TaskSpec manifest, raw backend trace, normalized
trace-v3, current scorer output, metric coverage, controller log, and receipt.

To add another method, implement one runner function with the same return
contract as `run_qm_control`: consume `--suite`/`--task-id`, execute the common
plant with method-native controller state, retain failures, and return a
timestamped directory containing per-scenario trace-v3 and receipts. Add the
method name to `METHODS`; do not add a new task generator or scorer.
