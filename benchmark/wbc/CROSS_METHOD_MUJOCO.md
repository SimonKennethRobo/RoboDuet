# Cross-method MuJoCo benchmark

The benchmark owner is `RoboDuet/benchmark/wbc`. Frozen TaskSpec/reference
validation, trace-v3 normalization, scoring, and final receipts are handled by
`cross_method_cli.py`. Method-specific controller processes stay in their
original trees and every consumed source/config/model is recorded by hash.

The registry covers all eight handoff methods. It restores the selected
TaskSpec initial state, replays the inherited time law, runs nominal and a
derived deterministic push variant, and emits trace-v3 for every connected
backend. Run the machine-readable preflight before launching work:

```bash
/opt/miniconda3/envs/isaacgym/bin/python -m benchmark.wbc.cross_method_cli \
  --list-methods
```

The current common-plant adapters are:

| method | controller boundary |
| --- | --- |
| `roboduet` | deployed RL-SAR dog policy + scripted DLS arm |
| `roboduet_raw` | original exported dog policy + scripted DLS arm |
| `ma2022` | recurrent dual-GRU student with arm-reaction prediction + scripted DLS arm |
| `deep_whole_body_control` | native history encoder + learned 18-joint position targets |
| `visual_wholebody` | native 71D/history policy for legs + its scripted DLS arm contract |
| `umi` | official 96D actor with four future EE pose observations |
| `wb_locoman` | native FATROP sidecar, direct 18-joint torque |
| `qm_control` | native SQP-MPC + QP-WBC ROS process |

All eight handoff methods have an executable common-plant adapter. The old
`cross-wbc-v1` IsaacGym traces are not used as current benchmark evidence.

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

The qm_control subprocess uses `/opt/miniconda3/envs/base312/bin/python` by
default because that environment contains its ROS/MuJoCo dependencies. The launcher gives it a clean environment so
stale ROS overlays in the interactive shell cannot change the run.
The workspace-specific baseline default can be replaced with
`--baseline-root /absolute/path/to/baselines/mpc_baseline`.

Each timestamped output contains `cross_method_manifest.json`, `results.json`,
`adapter.log`, the exported reference, and a nominal/push directory. Scenario
directories contain the TaskSpec manifest, raw backend trace, normalized
trace-v3, current scorer output, metric coverage, controller log, and receipt.

For a policy or sidecar method, replace its blocked registry entry only after
its real checkpoint/controller consumes common-plant state and passes a short
closed-loop smoke. Do not add a new task generator or scorer.

## First measured run

Commit `c64d844` produced the complete nominal/push run at
`benchmark/results/cross_method_mujoco/qm_control/20260915_035637`. Both
scenarios recorded 911 valid 20 ms samples and exited cleanly without a fall or
numerical fault. The push replay applied the requested `[8, 0, 0] N s` impulse
over 41 physics steps.

| scenario | success | end reason | EE position RMSE | EE rotation RMSE | final progress |
| --- | --- | --- | ---: | ---: | ---: |
| nominal | false | timeout | 0.12620 m | 0.73273 rad | 0.73595 |
| push | false | timeout | 0.12615 m | 0.73271 rad | 0.73582 |

This validates the method integration and failure-preserving result path. It
does not establish task success: both runs missed the endpoint, tracking-tube,
and hold criteria. qm_control does not expose policy actions, position targets,
reach-model utilisation, base feedforward, or the runner's Jacobian/IK
diagnostics, so those fields remain explicitly not-applicable or unavailable.
