# Ma2022 locomotion fidelity audit — 2026-09-11

Original audit verdict (before v2 edits): the implementation is a Go2 adaptation of the paper's core
training architecture, **not a verified faithful reproduction**. Disabling
RoboDuet's response objectives does not isolate the full training recipe.

Evidence checked: supplied Ma2022 PDF, Sections III-C/III-D and equations
(6)–(10); current source; resolved `build_ma_config()`; cloud run
`teacher_online_20260911-110050/parameters.pkl`. No training configuration or
running process was changed during this audit.

## Confirmed inherited training behavior

`build_ma_config()` starts with `build_roboduet_config()`. The cloud snapshot
confirms these inherited settings, which are not explicitly selected in the
Ma recipe:

| Setting | Active value | Consumer |
| --- | --- | --- |
| `domain_rand.randomize_Kp_factor` | true, range [0.8, 1.3] | `LeggedRobot._randomize_dof_props` and P controller |
| `domain_rand.randomize_Kd_factor` | true, range [0.5, 1.5] | Same |
| `domain_rand.randomize_motor_offset` | true, range [-0.02, 0.02] rad | Same |
| `domain_rand.rand_interval_s` | 4 s | `_post_physics_step_callback` resamples actuator properties within episodes, in addition to resets |

Motor-strength randomization [0.9, 1.1] and friction [0.4, 1.5] are explicit
local recipe choices. Their numerical settings are not established as Ma's
original settings. The inherited settings above likewise lack a verified
paper mapping; this is evidence of repository behavior entering the task,
not evidence that domain randomization itself contradicts privileged RL.

## Confirmed disabled or bypassed repository features

- Response reward names are absent from the replacement reward-scale table;
  response curriculum, grouping/nominal twins and excitation are disabled.
- Reset curriculum and fixed reset mixtures are disabled (`reset_mode=legacy`).
- Random robot pushes and action delay are disabled. The inherited push
  curriculum flag is inert while `push_robots=False`.
- The task subclasses `LeggedRobot`, not `WBCEnv`; no arm policy, arm
  disturbance curriculum, trajectory reward, arm-generated gait command,
  or WBC command smoothing participates.
- Dog observation latency and frame-drop settings remain in the inherited
  config, but their consumers are in `WBCEnv`. Ma builds observations itself,
  so these flags do not establish active latency or frame drops here.
- Parent response-state/metric bookkeeping still executes. Its execution
  alone does not mean response rewards enter PPO.

## Fidelity gaps beyond inherited improvements

1. **Reward definition differs.** Ma III-D1 explicitly follows reference [10]
   with increased orientation, vertical-velocity and roll/pitch-rate weights.
   [Reference 10, Supplement S7](https://arxiv.org/html/2201.08117v1#S7)
   specifies command-direction rewards, orthogonal velocity, foot clearance,
   joint motion, slip and first/second target differences. Current code instead
   registers ten generic repository rewards, uses squared velocity-error
   exponentials, and omits several of those terms. This is a formula/term-set
   difference, not just unavailable Ma-specific coefficients. The earlier
   statement that reward hyperparameters are unpublished was too broad:
   the referenced baseline does publish formulas and coefficients.
2. **Robot and action adaptation.** Training uses bare Go2 rather than ANYmal.
   `_joint_targets` supplies a locally chosen sinusoidal foot lift, two-link
   sagittal IK and bounded phase-rate modulation. Reset phases start in a
   trot pattern. These preserve the phase-plus-residual concept but have not
   been established as the original trajectory generator or reset recipe.
3. **Observation/architecture adaptation.** The 76-D proprioception, 25-point
   body-centered scan, actuator-factor privileged labels, 128-wide MLP/GRU
   modules and noise scales are local choices. Two RNN streams and decoder
   input isolation follow the paper's intent; matching that high-level
   structure does not verify the detailed observation format referred to [8].
4. **Equation (7) interpretation.** The terminal increment uses
   `Uniform(-abs(wmin), abs(wmax))`, multiplied by beta, then clips the terminal
   knot. With signed Fz bounds [-60, 0], all terminal increments are nonpositive.
   This is a documented interpretation of ambiguous signed/magnitude notation,
   not an independently verified original sampling convention.
5. **Experiment settings.** Wrench bounds, beta, noise, PPO settings, terrain
   mixture and training duration are locally selected. Successful execution
   checks do not validate equivalence or paper-level performance.

## Core mechanisms present

The three-knot rolling quadratic, 20-ms advance, five body-frame wrench
predictions, acceleration-dependent unobserved wrench plus Gaussian noise,
privileged PPO teacher, two recurrent student streams, and action/embedding/
four-decoder losses are implemented. W&B and video record the run and do not
add learning objectives.

Before labeling a new run a faithful baseline, isolate the configuration from
the RoboDuet profile, implement and document the referenced reward definitions,
and map each observation/action/randomization choice to the paper or explicitly
label it as a robot adaptation or unresolved detail. The currently running
checkpoint should be treated as the existing adapted baseline.

## Subsequent v2 corrections

The local v2 implementation now clears inherited dynamics DR flags, fixes
actuator factors to nominal, and owns its reward registration/computation.
S7 reward terms and coefficients replace the old generic rewards; first and
second target differences, contact-foot slip and foot-centered clearance
sampling are implemented. A task-owned episode penalty curriculum follows
the reference exponent. Cubic lift and per-step phase increments replace
sinusoidal lift and frequency modulation. V1 checkpoints are rejected.

The directional command normalization/sign interpretation, Ma stability
multiplier and orientation formula, initial penalty curriculum value, Go2
geometry/lift, policy observation history/scan and network settings remain
explicit adaptations or unresolved fidelity details. Wrench Eq. (7) retains
the documented signed-bound interpretation. This update does not establish
strict full-paper equivalence. See `ma2022_locomotion.md` for the v2 contract.
