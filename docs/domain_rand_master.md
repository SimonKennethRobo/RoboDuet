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

| Feature                                                                    | benchmark         | sim2real          | none      |
| -------------------------------------------------------------------------- | ----------------- | ----------------- | --------- |
| Friction, restitution, base mass/COM, motor strength, Kp/Kd                | Configured recipe | Configured recipe | Disabled  |
| Both arm stages' gains, motor strength, link mass/COM, synthetic payload   | Configured recipe | Configured recipe | Disabled  |
| Pushes and EE external forces                                              | Configured recipe | Configured recipe | Disabled  |
| Gravity perturbation, dog/arm motor offsets, mount position/rotation error | Disabled          | Configured recipe | Disabled  |
| Observation noise, sensing delay/jitter, frame drop, action delay          | Disabled          | Configured recipe | Disabled  |
| Task commands, arm motion, reset/trajectory curricula, rewards, smoothing  | Preserved         | Preserved         | Preserved |

“Configured recipe” respects every existing individual flag and range; it does
not turn intentionally disabled features on. Both modes preserve normalization
ranges, observation/action widths, and nominal URDF physics. Legacy config-only
`randomize_lag_timesteps` is masked alongside the active action-delay flag.

Benchmark training targets robustness in perturbed simulation, not just a high
nominal training reward. It retains task-relevant dynamics variation and removes
additional hardware uncertainties. Ranges are not automatically widened or
narrowed: performance claims require validation against the chosen test protocol.
The current dog-policy evaluator focuses on command tracking and arm disturbance;
it owns its scenario overrides independently of the training mode.

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
configs contain no `enabled` field. Full older snapshots also keep sensing latency
disabled, point payload loads and the old terrain generator when these new fields
are absent. Play overrides apply after snapshot loading.

## Terrain and physical randomization on v3-stage2

Ported from `33fb403`, `d136c66`, and `b71ecc0` on `feat/rlmpc`.
The response model, R8 curriculum and nominal twins are not part of this port.
Terrain is sampled uniformly among `[0.0, 0.02, 0.04]` metre height-noise
amplitudes at creation and reset. `roughness_tier_weights` controls map column
shares, not the probability of selecting each tier. Terrain and reset curricula
are task settings and remain enabled with DR mode `none`; select
`terrain.mesh_type = "plane"` separately for flat terrain.

Chassis CoM is drawn once at actor creation with a +/-5 cm range. Chassis
mass resampling reaches the simulator before DOF/root reset writes when
`randomize_rigids_after_start` is enabled. Go2 assets use the `trunk` body,
because the first body named `base` is too light for the configured +/-2 kg
range. Assets without `trunk` retain the first-body fallback; nonpositive mass
fails explicitly. Arm link mass/CoM remain per-environment creation properties.

Stage-1 synthetic payload weight acts at an EE-frame offset sampled within
`[0.10, 0.05, 0.05]` m per-axis half-widths, scaled by the existing arm curriculum.
Sensing latency affects measurements before tracking errors are computed;
commands and previous actions remain current. It does not change policy widths.
Repeated reads within a policy step share the same latency jitter. Reset fills
all latency slots with the new episode's measurements. Play disables latency.

## Validation

```bash
PATH=/opt/miniconda3/envs/isaacgym/bin:$PATH LD_LIBRARY_PATH=/opt/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH PYTHONPATH=. /opt/miniconda3/envs/isaacgym/bin/python -m pytest -q go1_gym/envs/config/test_domain_rand_master.py go1_gym/envs/config/test_domain_rand_features.py
```

These tests check modes, snapshots, latency/reset and terrain generation;
they do not measure trained-policy performance.

GPU integration smoke (8 environments, 32 steps, no policy training):

```bash
PATH=/opt/miniconda3/envs/isaacgym/bin:$PATH LD_LIBRARY_PATH=/opt/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH PYTHONPATH=. /opt/miniconda3/envs/isaacgym/bin/python scripts/check_domain_rand_runtime.py --mode sim2real
```

Use `--mode benchmark` or `--mode none` for the other modes.
