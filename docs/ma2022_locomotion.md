# Ma et al. (2022) locomotion training

This implementation adds the locomotion training pipeline from Sections III-C
and III-D and Figure 4 of [the supplied paper](<Ma et al_2022_Combining Learning-Based Locomotion Policy With Model-Based Manipulation for.pdf>).
Run it through `scripts/train_ma2022.py`. It trains the checked-in **bare Go2**
asset: there is no arm in the training simulator.

## Training

From the repository root:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/train_ma2022.py --stage teacher --headless \
  --num_envs 4096 --iterations 20000 --log_dir runs/ma2022/teacher

python scripts/train_ma2022.py --stage student --headless \
  --teacher runs/ma2022/teacher/teacher_020000.pt \
  --num_envs 4096 --iterations 20000 --log_dir runs/ma2022/student
```

Iteration counts are starting points, not a convergence claim. Select a teacher
checkpoint based on tracking and stability before distillation. The student
rolls out its own actions and receives teacher targets at the states it visits.

Edit `go1_gym/envs/config/ma2022.py` for wrench limits, randomization, input
noise, rewards, network sizes, PPO and distillation settings. New teachers use
that Python recipe. Student training restores the teacher's saved environment
and recipe. Resume restores the saved model, optimizer, recipe and iteration;
simulation episodes and recurrent states restart. It is not a bitwise replay of
the original run. `--rollout_steps` and `--terrain` are explicit runtime
overrides; the resulting settings are recorded in the new checkpoint.

```bash
python scripts/train_ma2022.py --stage teacher --headless \
  --resume runs/ma2022/teacher/teacher_020000.pt \
  --iterations 30000 --log_dir runs/ma2022/teacher_resumed

python scripts/play_ma2022.py runs/ma2022/student/student_020000.pt
```

Playback opens the viewer and continues until interrupted. It evaluates the
same synthetic-wrench task, with sampled velocity commands; it does not run an
arm MPC. GUI playback has not been exercised as part of the headless checks.

Each run writes `parameters.pkl`, `metrics.log`, and periodic/final
`teacher_NNNNNN.pt` or `student_NNNNNN.pt`. Student training also exports
`student_jit.pt`. Logs report individual distillation losses, reward, planar
velocity error (m/s), base tilt (rad), roll/pitch rate magnitude (rad/s), and
termination fraction per control step. Checkpoints publish with atomic
replacement; failed checkpoint writes stop the run. Noncritical metrics-write
errors skip that record with a rate-limited warning.

W&B logging is enabled by default in project `roboduet`. Use `--run_name`,
`--wandb_project` and `--wandb_entity` to choose the run name and destination.
`--offline` records W&B data locally under `<log_dir>/wandb` for later
`wandb sync`; `--no_wandb` disables W&B and takes precedence over `--offline`.
Configuration and command-line arguments are saved to the run. Per-iteration
charts use `Loss/*`, `Train/*`, and `Performance/*`, with iteration as their
x-axis. Resuming a checkpoint creates a new W&B run starting at the restored
iteration. W&B initialization, logging and shutdown errors are noncritical and
do not interrupt training or local checkpoint saving.

Training video recording is enabled by default, including with `--headless`.
The first iteration starts a clip; subsequent clips start every 500 iterations
when no clip is already active. Each clip follows one training robot for 10
simulation seconds across episode resets, at 640×480 and 25 fps. Frames stream
to `<log_dir>/videos/<stage>_<start_iteration>.mp4`; completed clips are also
logged to W&B under `Video/training`. A shorter final clip is finalized when
training ends. This records the training rollout without extra simulator steps.

Use `--video_interval 500 --video_length 10 --video_stride 2` to change the
schedule, length and frame sampling, or `--no_video` to disable recording.
`--no_wandb` still preserves local MP4 files. Video encoding/write errors skip
the affected clip. Headless recording needs a working GPU graphics device;
`--no_video` is available for compute-only installations.

## Paper-to-code mapping

| Paper component | Implementation |
| --- | --- |
| Eq. (6): three random six-axis wrench knots at 0, 1, 2 s | `WrenchSequence.reset` |
| Eq. (7): rolling quadratic, episode-specific rate parameter | `WrenchSequence.advance` at 50 Hz |
| Eq. (8): predictions at 0, .2, .4, .6, .8 s, commands and base twist | 39-D wrench stream in `MaLocomotionEnv.observations` |
| III-C2: unobserved acceleration-dependent disturbance and Gaussian noise | Two vectors uniform in 3-D balls; elementwise multiplication with world-frame linear/angular acceleration at each physics substep |
| Sum of predictable/unobserved wrench applied to base center | `MaLocomotionEnv._arm_decimation_hook`, at every 5-ms physics step |
| Figure 4(a): privileged PPO teacher with separate MLP encoders | `Teacher` and `teacher_iteration` |
| Figure 4(b): two recurrent student streams | `Student.wrench_rnn` and `Student.belief_rnn` |
| Eq. (9): action and embedding imitation | `distillation_losses`: `action`, `embedding` |
| Eq. (10): privileged, scan, total applied wrench, load-gain decoding | `privileged`, `scan`, `w1`, `w2` losses |
| Phase and joint-residual actions with joint PD control | Four phase-rate actions, cyclic sagittal IK, twelve joint residuals |

The belief recurrent stream sees proprioception and a noisy height scan. It
never receives the wrench prediction, and **all four decoders read only this
stream**. The separate wrench recurrent stream receives proprioception and
the noisy prediction. This preserves the information separation in Figure 4.
Teacher embeddings and actions are detached supervision. Gradients flow
through both student recurrent streams over each rollout chunk, with hidden
states masked at episode boundaries and detached between chunks.

PPO bootstraps genuine timeouts from the pre-reset terminal observation and
does not bootstrap falls. GAE never crosses an episode boundary. Wrench
parameters, finite-difference velocity caches, actions and phase histories
reset per environment; command resampling does not restart the wrench process.

## Observation and action contract

All quaternions use xyzw. World-frame wrenches are ordered
`[Fx,Fy,Fz,Tx,Ty,Tz]`, in N and Nm, about the **base link origin**, used as the
Go2 equivalent of the paper's geometric base center. IsaacGym applies forces
at COM, so the adapter uses `tau_COM = tau_origin - r_COM × F` before submitting
local-frame force/torque tensors. Both force and torque predictions rotate
into the **current** base frame; future samples are not rotated into predicted
future frames.

Teacher inputs:

| Stream | Width | Contents |
| --- | ---: | --- |
| `proprio` | 76 | Projected gravity (3), body linear velocity (3), body angular velocity (3), velocity commands (3), joint position offsets (12), joint velocities × .05 (12), previous policy action (16), action before that (16), sine/cosine of four leg phases (8) |
| `wrench` | 39 | Five normalized six-axis predictions (30), velocity commands (3), body linear/angular velocity (6) |
| `scan` | 25 | 5×5 base-height-relative ground samples, centered on target base height, clipped to ±1 m and multiplied by `scan_scale` |
| `privileged` | 49 | Friction (1), four foot contact forces × .01 (12), motor-strength/Kp/Kd factors (12 each) |

The student substitutes noisy measured gravity/twist/joints, wrench prediction
and height scan. Its command, previous-action and phase entries stay exact.
Wrench scale and bias noise is shared across the prediction horizon and sampled
per episode; additive prediction noise is sampled per observation.

`w1` targets the normalized total wrench applied during the final physics
substep leading to the observation. `w2` targets the two sampled disturbance
gain-vector norms, divided by their sampling radii. Clean privileged labels,
clean scans, actual applied wrench and gain magnitudes are training targets;
they are not student inputs.

The exported TorchScript interface is:

```python
action, next_wrench_state, next_belief_state = policy(
    proprio, wrench, scan, wrench_state, belief_state, reset
)
```

The three observations have the widths above. Both states have shape
`(batch, hidden_dim)` and initially contain zeros. `reset` is a Boolean vector
of shape `(batch,)` and clears the states **before** processing that observation.
Carry both returned states into the next call.

The policy returns 16 actions:

- `0:4`: per-leg frequency offsets in FL, FR, RL, RR order. Clip to ±1 and
  integrate `phase += .02 * (gait_frequency + phase_frequency_scale * action)`
  modulo 1. Reset phases are `[0, .5, .5, 0]`.
- `4:16`: residual joint positions in asset DOF order: FL, FR, RL, RR, with
  hip/thigh/calf in each leg. Clip policy outputs to ±3, multiply residuals by
  `residual_scale`, and add to the cyclic IK target in `env._joint_targets`.
  Use the configured PD controller, not direct torque interpretation.

These are format-versioned `ma2022-locomotion-v1` checkpoints. They are not
compatible with `auto_train.py`, `load_policy.py`, `export_rl_sar`, or existing
12-action RoboDuet dog checkpoints. Deployment needs a matching phase/IK
adapter, sensors/height scan, and externally supplied wrench predictions. The
TorchScript artifact exposes those inputs but does not implement MPC/RNEA,
an RL-SAR adapter, or hardware control.

## Reproduction boundaries and numerical choices

This implements the paper's training structure on Go2/IsaacGym, not an exact
ANYmal/RaiSim reproduction. The paper delegates proprioception, action
generation and locomotion rewards to earlier work and does not give numerical
wrench limits, beta bounds, noise magnitudes, network widths or loss weights.
Here those use explicitly editable Go2 defaults. The cyclic sagittal IK
template, 5×5 height scan, short action history, GRU cells and reused RoboDuet
reward functions are implementation choices. Orientation, vertical velocity
and roll/pitch-rate penalties are increased as described in III-D1.

For Eq. (7), signed configured bounds are converted to lower/upper *magnitudes*
for the terminal-knot random-walk interval, then the terminal knot is clipped
to the signed bounds. Only the terminal knot is clipped; the intervening
quadratic can overshoot. This sign convention is explicit because the printed
equation does not disambiguate signed bounds from magnitudes. For a
downward-only force range the terminal increment is also downward-only;
adjust signed ranges if bidirectional terminal variation is desired.

The existing response-consistency rewards, grouping/excitation curricula,
arm disturbance curriculum, reset mixtures and random velocity pushes are
disabled for this task. The base simulator's state management, terrain,
actuator randomization and common locomotion rewards are reused.

## Validation

```bash
python -m pytest go1_gym/ma2022/test_ma2022.py -q
python -m go1_gym.ma2022.check_sim --terrain plane
python -m go1_gym.ma2022.check_sim --terrain trimesh
```

The bounded simulator check verifies selective wrench resets, no artificial
reset acceleration impulse, terminal/time-out handling, teacher and student
weight updates, unscaled checkpoint configuration, and TorchScript loading.
It does not test long-run convergence, manipulation tracking, or hardware.

Validation on 2026-09-11: all 9 new unit tests passed; GPU IsaacGym integration
checks passed on plane and trimesh terrain. The CLI teacher, student, student
resume and final export paths also completed bounded runs. The combined new
and response test suite reported 250 passed / 5 failed. All five failures are
existing command-layout expectation mismatches reproduced from an isolated
archive of unmodified HEAD `0f02d9d` (that test file: 13 passed / 5 failed).
Local CLI validation artifacts are under `tmp/ma2022_validation/`; those
short-run checkpoints are execution checks, not trained locomotion policies.
