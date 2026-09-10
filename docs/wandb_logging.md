# W&B logging schema v2

Both dual-policy and unified runners use `go1_gym/logging_metrics.py`.
New logs use these sections; old W&B history and run names are not rewritten.
`logging_schema_version=2` is stored in W&B config, with `Runtime/iteration`
as the explicit default X axis. `Diagnostics/*` is hidden from automatic plots.
W&B's workspace controls determine how sections are expanded/displayed.

| Section | Meaning |
| --- | --- |
| Performance/Dog/Tracking | Velocity and body-height tracking |
| Performance/Dog/Stability | Body oscillation and foot slip |
| Performance/Dog/Efficiency | Locomotion mechanical power |
| Performance/Arm/Tracking | EE pose error and waypoint offsets |
| Performance/Arm/Trajectory | Trajectory tracking, reach utilization, success |
| Performance/Arm/Termination | Arm task termination diagnostics |
| Episode/Shared | Whole-robot returns, duration, overall early termination |
| Reward/Dog, Reward/Arm | Weighted reward terms grouped by physical task |
| PPO/Dog, PPO/Arm | Actual optimizer updates, learning rate, action std |
| Curriculum/Arm, Reset, Command, BaseUnlock, Trajectory, Terrain | Curriculum state |
| Runtime | Iteration, environment transitions, session elapsed time, throughput |
| Diagnostics | Cohort/episode-age breakdowns and unmapped diagnostics |

Reward grouping describes the physical task, not exclusive PPO ownership:
the existing reward code shares many terms between dog and arm objectives.
No reward computation, training distribution, or aggregation is changed.
`jump` is displayed as `base_height`; original reward/config keys remain valid.
Unknown episode keys go to Diagnostics, never automatically to Reward.

## Statistical meanings

- `_rollout`: sample-weighted over the current rollout, including unfinished
  and failed episodes. MAE = abs-error sum / sample count; RMSE = sqrt(squared
  error sum / sample count). Available for vx/vy/yaw when robustness metrics
  are enabled. The existing `robustness.jsonl` retains all raw sums/counts.
- `_episode_mean`: existing completed-episode physical aggregation. For RMSE
  this is an average of per-episode RMSEs, not pooled sample RMSE. Existing
  weighting is unchanged, including arm metrics' existing completion weighting.
- `_return_mean_reset_batches`: existing mean of reset-batch mean weighted
  episode reward sums, not a per-second reward or a true episode-weighted mean.
- `_100ep`: rolling window of up to 100 completed episodes. Shared return sums
  dog and arm returns as before. Episode length is explicitly in policy steps.
- `base_height_bias_m`: actual minus commanded height; negative means too low.
- easy/hard refers to reset cohorts, not terrain tiers. early/late refers to
  episode age, not time since push; those curves live under Diagnostics/Dog.
- Zero-sample rollout ratios are omitted rather than filled with zero.

Whole-robot termination/duration are under Episode/Shared. No bottom-level
failure attribution is inferred. Frozen PPO branches omit losses and learning
rates; disabled adaptation modules omit adaptation loss. Action std remains
available and `trainable` makes branch state explicit.

`Runtime/env_steps` retains the runner's existing transition counter (including
its configured environments). `Runtime/wall_time_s` is elapsed wall time for
this invocation of learn(), including logging overhead. Throughput retains the
existing collection+learning timing denominator. Static stage boundaries remain
in run config instead of separate Global_Switch curves.

## Validation

```bash
PYTHONPATH=. /opt/miniconda3/envs/isaacgym/bin/python -m pytest -q go1_gym/envs/config/test_logging_metrics.py
```

An offline W&B smoke validates metric definitions and nested keys without
uploading a run. This change does not modify an existing remote workspace or
retroactively rename historical data.
