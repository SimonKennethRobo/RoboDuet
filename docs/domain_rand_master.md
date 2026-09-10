# DR training modes

The default RoboDuet training mode is `benchmark`. Edit `COMMON_OVERRIDES`
in `go1_gym/envs/config/wbc.py`:

```python
"domain_rand.mode": "benchmark",  # benchmark / sim2real / none
```

CLI mode selection overrides the source mode:

```bash
python scripts/auto_train.py --train_stage stage1 --dyna_gait --headless --domain_rand_mode benchmark --run_name stage1_benchmark
python scripts/auto_train.py --train_stage stage1 --dyna_gait --headless --domain_rand_mode sim2real --run_name stage1_sim2real
```

Stage 2 accepts the same flag alongside its checkpoint arguments. `none` is
available for nominal ablations: use `--domain_rand_mode none`. The redundant
`enabled` field and `--domain_rand` / `--no_domain_rand` flags have been removed.
Startup prints the effective training mode.

## Mode boundaries

| Feature | benchmark | sim2real | none |
| --- | --- | --- | --- |
| Friction, restitution, base mass/COM, motor strength, Kp/Kd | Configured recipe | Configured recipe | Disabled |
| Both arm stages' gains, motor strength, link mass/COM, synthetic payload | Configured recipe | Configured recipe | Disabled |
| Pushes and EE external forces | Configured recipe | Configured recipe | Disabled |
| Gravity perturbation, dog/arm motor offsets, mount position/rotation error | Disabled | Configured recipe | Disabled |
| Observation noise, sensing delay/jitter, frame drop, action delay | Disabled | Configured recipe | Disabled |
| Task commands, arm motion, reset/trajectory curricula, rewards, smoothing, response training | Preserved | Preserved | Preserved |

“Configured recipe” respects every existing individual flag and range; it does
not turn intentionally disabled features on. Both modes preserve normalization
ranges, observation/action widths, and nominal URDF physics. Legacy config-only
`randomize_lag_timesteps` is masked alongside the active action-delay flag.

Benchmark training targets robustness in perturbed simulation, not just a high
nominal training reward. It retains task-relevant dynamics variation and removes
additional hardware uncertainties. Ranges are not automatically widened or
narrowed: performance claims require validation against the chosen test protocol.
The current dog-policy evaluator focuses on command tracking and arm disturbance;
it does not consume `configs/domain_*.json`. This training mode is consequently a
conservative robustness recipe, not an exact reconstruction of those JSON tiers.

## Evaluation ownership

`benchmark/dog_policy/evaluation.py` replaces checkpoint DR/noise settings with
one common source recipe before applying its existing evaluation overrides.
Thus candidate training mode and DR ranges cannot decide the
benchmark's DR configuration. Existing scenario code still owns which
perturbations are active. A runtime-only ownership marker ensures environment
construction does not mask scenario overrides (for example, an explicit latency
test). Robot-specific mount joint identity is preserved.

This standardizes the DR configuration; it does not implement paired random
samples or a new common disturbance schedule across policy rows. Existing
benchmark sampling/scheduling behavior remains. Matching seeds alone should not
be interpreted as proof that every policy received identical disturbances.

## Configuration lifetime and checkpoints

Mode resolution runs before actor/buffer creation in `LeggedRobot.__init__`,
including the optional evaluation config. Create a new environment to switch
modes; mount and rigid-body changes cannot be switched live.

Training `parameters.pkl` stores mode and the original individual
recipe. `env.cfg` contains effective masked settings. To switch from benchmark
to sim2real, rebuild or reload that original recipe before constructing the env;
changing a previously masked `env.cfg` cannot recover discarded values.

Full snapshots override source defaults during play. Snapshots without `mode`
fall back to `sim2real`, preserving pre-mode behavior. The snapshot loader
converts old `enabled=False` to `mode="none"` (even if an old mode is present),
and removes the obsolete field. Old `enabled=True` retains an explicit mode
or falls back to `sim2real`. Input snapshots are not mutated and newly saved
configs contain no `enabled` field. Play overrides apply after snapshot loading.

## Validation

```bash
PYTHONPATH=. /opt/miniconda3/envs/isaacgym/bin/python -m pytest -q go1_gym/envs/config/test_domain_rand_master.py go1_gym/response/test_response_config.py
PATH=/opt/miniconda3/envs/isaacgym/bin:$PATH LD_LIBRARY_PATH=/opt/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH PYTHONPATH=. /opt/miniconda3/envs/isaacgym/bin/python -m pytest -q benchmark/test_stage2_backports.py
```

Tests cover mode masking, retained dynamics/task/layout, recipe isolation,
checkpoint compatibility, and the evaluator entrypoint's independence from
candidate training modes. These tests do not measure trained-policy scores.
