# Ma et al. (2022) locomotion training

This implementation adapts the locomotion training pipeline from Sections III-C
and III-D and Figure 4 of [the supplied paper](<Ma et al_2022_Combining Learning-Based Locomotion Policy With Model-Based Manipulation for.pdf>).
It is not a verified faithful reproduction: the [fidelity audit](ma2022_fidelity_audit.md)
records the original gaps and the v2 corrections. Observations and robot-specific
parameters still contain adaptations; see the remaining limitations below.
Run it through `scripts/train_ma2022.py`. It trains the checked-in **bare Go2**
asset: there is no arm in the training simulator.

## V3 corrections

V3 fixes three issues from an independent review against the paper:

- **Eq. (7) drift.** The terminal-knot random walk is now zero-mean (see
  "Reproduction boundaries"). Under v2, Fz drifted to and stayed at -60 N.
- **Constant privileged inputs.** Actuator factors are nominal in this recipe,
  so the 36 motor-strength/Kp/Kd entries were constant. They are removed from
  the privileged observation and the `privileged` decoder target (49 → 13).
- **Phase authority.** `phase_increment_scale` drops from 1.0 to 0.1 rad per
  action. At 1.0, a clipped action moved the phase ±0.48 cycles per 20-ms step,
  12× the nominal 0.04-cycle advance, so the phase no longer acted as a gait
  clock. At 0.1 the bound is ±0.048 cycles.

Observation widths and the dynamics changed, so v2 checkpoints are rejected.
The v2 cloud teacher (below) was trained with the Eq. (7) drift and should not
be used as the baseline.

## V2 alignment changes

V2 uses an independent reward kernel in `go1_gym/ma2022/rewards.py`, based on
[Ma reference 10, Supplement S7](https://arxiv.org/html/2201.08117v1#S7).
It includes directional command tracking, orthogonal velocity, body motion,
clearance, collision, joint motion, knee constraints, first/second target
smoothness, torque and contact-foot slip. Coefficients are the S7 coefficients;
terms are per control step, without the repository's dt multiplier or positive
reward clipping. The printed direction/sign notation is interpreted using a
unit command direction and absolute commanded yaw speed; this convention is
explicitly documented in the kernel and tested for mirrored commands.

Ma III-D1 increases orientation, vertical velocity and roll/pitch-rate costs.
The multiplier 2 and squared-horizontal-gravity orientation term are explicit
local choices because the Ma-specific formula/coefficients are not provided.
The reference curriculum `c <- c**0.98` advances independently at each completed
episode, using a locally chosen initial value 0.1. It scales only the documented
penalty terms, not wrench magnitudes or reset difficulty.

All inherited dynamics randomization flags are cleared before enabling the
explicit friction recipe. Kp, Kd, strength and zero offsets are nominal, even
if the parent periodically calls its DOF sampler. Parent configuration is
still used for the simulator schema; this is not a claim of complete schema
independence. WBC latency/drop flags and push curriculum are explicitly off.

The lift shape now follows the cubic Hermite trajectory in reference 10 S5;
phase actions are per-step radians rather than frequency perturbations.
Go2 geometry, 0.06 m lift and residual amplitude remain robot adaptations.
The reward samples 52 heights around each foot for clearance. The policy's
25-point base scan and 76-D proprioception remain the earlier adaptations;
these have not been replaced with the full referenced observation history.
The actuator-factor privileged entries are now constant nominal values.

V1 checkpoints are deliberately rejected because reward and action semantics
changed. New training requires a fresh teacher and a student distilled from
that v2 teacher. The cloud run documented below was launched with v1 and is
not automatically converted by editing local source.

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

W&B logging is online by default in `simon00715/roboduet`, matching
`auto_train.py`. Runs use `group=run_name` and names of the form
`YYYY-MM-DD/<run_name>_HHMMSS`; `--notes` sets run notes. Use `--run_name`,
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
| Phase and joint-residual actions with joint PD control | Four phase increments, cubic Hermite lift and sagittal IK, twelve joint residuals |

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
| `privileged` | 13 | Friction (1), four foot contact forces × .01 (12) |

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

- `0:4`: per-leg phase increments in FL, FR, RL, RR order. Clip to ±3 and
  integrate `phase += .02 * gait_frequency + phase_increment_scale * action / (2*pi)`
  modulo 1. Reset phases are `[0, .5, .5, 0]`.
- `4:16`: residual joint positions in asset DOF order: FL, FR, RL, RR, with
  hip/thigh/calf in each leg. Clip policy outputs to ±3, multiply residuals by
  `residual_scale`, and add to the cyclic IK target in `env._joint_targets`.
  Use the configured PD controller, not direct torque interpretation.

These are format-versioned `ma2022-locomotion-v3` checkpoints. They are not
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
template, 5×5 policy height scan, short action history and GRU cells remain
implementation choices. Reward definitions now follow reference [10] S7,
with the interpretation and Ma-specific choices described above. Orientation, vertical velocity
and roll/pitch-rate penalties are increased as described in III-D1.

For Eq. (7), the terminal-knot increment is zero-mean in every dimension:
`beta * Uniform(-(wmax-wmin)/2, +(wmax-wmin)/2)`, then the terminal knot is
clipped to the signed bounds. Only the terminal knot is clipped; the
intervening quadratic can overshoot. The printed equation does not
disambiguate signed bounds from magnitudes. v2 read them as one-sided
magnitudes, which for Fz in [-60, 0] made every increment nonpositive: the
applied Fz averaged -57 N by 10 s and 95% of environments were pinned at
-60 N by 19 s, so the z prediction was nearly constant for most of each
episode. v3 uses the zero-mean walk so the wrench keeps varying across the
configured range.

The existing response-consistency rewards, grouping/excitation curricula,
arm disturbance curriculum, reset mixtures and random velocity pushes are
disabled for this task. The base simulator's state management, terrain,
friction sampling are reused. Actuators are nominal; task-owned rewards bypass
the parent reward registry, global-switch weights and response curriculum.

The rough-terrain grid uses 10 rows × 20 columns, with PhysX
`default_buffer_size_multiplier=32`. The original 3×10 grid crowded 4096
robots onto too few tiles and overflowed GPU aggregate-pair buffers on the
cloud RTX 4090. The larger grid preserves the roughness proportions while
reducing broadphase overlap; the larger buffers leave additional capacity.

## Cloud teacher run (2026-09-11)

Host: `ssh -p 30071 root@183.147.142.40`. Runtime setup follows
[CloudGPU使用手册](CloudGPU使用手册.md). The Ma entrypoint uses `--stage` and
`--iterations`; the manual's existing `train.sh` launches `auto_train.py` and
is not the launcher for this task.

The 4096-environment, 20000-iteration teacher run is managed by tmux session
`rd-ma2022-teacher`, with video enabled and W&B online (`simon00715/roboduet`). Its actual command is
saved in `/root/gpufree-data/roboduet-conda/jobs/ma2022_teacher_online-20260911-110050.sh`.
Online run: [259o56nh](https://wandb.ai/simon00715/roboduet/runs/259o56nh).

```bash
tmux attach -t rd-ma2022-teacher
# Detach with Ctrl+B, then D; training continues.

tail -f /root/gpufree-data/roboduet-conda/logs/ma2022_teacher_online-20260911-110050.log
```

Run directory:
`/root/gpufree-data/RoboDuet/runs/ma2022/teacher_online_20260911-110050`.
Checkpoints, `videos/` and local `wandb/` data live there.
The earlier offline run was stopped at the user's request; this online run
starts from scratch. The earlier logs and videos remain in
`runs/ma2022/teacher_20260911-104604`. The matching `.exit`
file beside the terminal log is written when the task exits. Code is commit
`a499222` plus the terrain-grid/PhysX-capacity adjustment above, applied to both
local and cloud copies, plus the W&B configuration aligned with
`scripts/auto_train.py`. A short cloud preflight verified checkpoint tensors
and MP4 decoding before the formal run.

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

V2 validation: 14 CPU tests passed; bounded GPU checks on plane and trimesh
cover reward independence from global-switch weights, nominal actuators,
reset histories, PPO/distillation updates, checkpoint saving and JIT export.
These are execution checks, not convergence or performance-equivalence evidence.

## V2 cloud launch (2026-09-11, port 30322)

Commit `f911a16` was deployed separately to
`/root/gpufree-data/RoboDuet-ma2022-v2-f911a16` using the existing Conda runtime.
The fresh teacher uses GPU 0, 4096 environments, 20000 iterations, online W&B
and default training video. Bounded cloud trimesh integration passed before
launch. [Online run](https://wandb.ai/simon00715/roboduet/runs/swa03b12).

```bash
ssh -p 30322 -t root@183.147.142.40 'tmux attach -t rd-ma2022-v2-teacher'
```

Run directory: `runs/ma2022/teacher_v2_20260911-112924` under that checkout.
Terminal log: `/root/gpufree-data/roboduet-conda/logs/ma2022_v2_teacher-20260911-112924.log`.
The adjacent `.exit` file is written when the job ends. The exact launcher is
`/root/gpufree-data/roboduet-conda/jobs/ma2022_v2_teacher-20260911-112924.sh`.
Old source and runs are retained in `/root/gpufree-data/RoboDuet`.
