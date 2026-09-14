# AGENTS.md

Guidance for coding agents working in this RoboDuet repository.

## Project Shape

RoboDuet is a dual-policy legged manipulation environment built on IsaacGym.
A quadruped (Go1 / Go2) carries a 6-DOF arm; locomotion and arm control are
learned by two separate PPO actor-critics that share a single env step.
Active environment code lives under `go1_gym/envs/roboduet/`; the learner
lives under `go1_gym_learn/ppo_cse_automatic/`.

Core files:

- `go1_gym/envs/roboduet/legged_robot.py`: base IsaacGym task, sim stepping, reset, root/DOF tensors, terrain, actor creation, common dog observations/rewards, and generic domain randomization.
- `go1_gym/envs/roboduet/wbc_env.py`: RoboDuet arm/WBC extension. Arm commands, trajectory tracking, arm observations/rewards, stage switching, arm-specific domain randomization, and the `plan()` path that translates arm-policy outputs into smoothed dog commands.
- `go1_gym/envs/config/`: flat configuration package. `legged_robot.py`, `go1.py`, `wtw.py`, and `roboduet.py` contain profiles; `core.py` contains schema materialization, profile composition, runtime build, serialization, validation, and observation/action layout derivation.
- `go1_gym/envs/config/roboduet.py`: complete editable RoboDuet profile. `ROBODUET_OVERRIDES` can override any base/Go1/WTW field without modifying another task profile.
- `go1_gym/envs/roboduet/wbc_env_wrapper.py`: interactive wrappers and `HistoryWrapper`. `HistoryWrapper.step(action_dog, action_arm)` is the canonical step API for training and play.
- `go1_gym/envs/roboduet/utils.py`: `StageSchedule`, `apply_wbc_reward_settings`, `ObservationBuilder` (asserts obs width vs. `cfg.*_num_observations`).
- `go1_gym/envs/roboduet/traj_gen/trajectory_geometry.py`: `sample_trajectory_commands` — supports `line`, `s_curve`, `circle`, `point` types and consumes an optional `orientation_scale` curriculum from the env.
- `go1_gym/envs/rewards/rewards.py`: reward function registry. Reward methods named `_reward_<name>` are auto-wired by `cfg.rewards.scales.<name>`.
- `go1_gym/utils/global_switch.py`: process-global `global_switch` singleton driving stage transitions and reward-scale ramping.
- `go1_gym_learn/ppo_cse_automatic/__init__.py`: dual-PPO `Runner` orchestrating arm and dog learners, stage transitions, and stage-2 locomotion freezing.
- `go1_gym_learn/ppo_cse_automatic/ppo.py`: PPO algorithm; supports `set_learning_rate()` and `clear_storage()` (used when dog policy is frozen).
- `scripts/auto_train.py`: main training entrypoint. Builds an independent `cfg`, configures stage schedule, loads optional checkpoints, copies dog command limits from the dog checkpoint, and runs `Runner.learn()`.
- `scripts/load_policy.py`: play/eval loader. It builds an independent `cfg`, restores the complete `parameters.pkl` snapshot, then applies play-time overrides.
- `scripts/play_by_joy.py`: interactive joystick play; see "Play Scripts" and "Joystick Commands".

## Training Pipeline

`Runner` in `go1_gym_learn/ppo_cse_automatic/__init__.py` trains two
actor-critics together inside a single rollout loop:

- `arm_model` (`ArmActorCritic`): reads `env.get_arm_observations()`; output width is `cfg.arm.num_actions_arm_cd`.
- `dog_model` (`DogActorCritic`): reads `env.get_dog_observations()`; output width is `cfg.dog.num_actions_loco`.

A process-global `global_switch` singleton (`go1_gym/utils/global_switch.py`)
gates the curriculum:

- `global_switch.count`: iteration counter, incremented per learning iteration.
- `global_switch.switch_open` (alias for `switch_flag`): once `True`, stage-2 is active — the arm policy participates in training, WBC reward scales are applied, and trajectory-tracking logic runs in `WBCEnv`.
- `global_switch.pretrained_to_wbc_start` / `_end`: iteration window over which `get_reward_scales()` linearly interpolates from stage-1 (pretrained) scales to stage-2 (WBC) scales via a sigmoid ramp (`init_sigmoid_lr()`).
- `global_switch.stage1_count` / `stage1_arm_ramp_iterations`: drive the stage-1 arm-disturbance intensity curriculum.

`StageSchedule.configure(global_switch)` (in `wbc_env/utils.py`) sets the
switch iteration based on `--train_stage`:

- `stage1`: switch never opens; arm policy outputs are ignored; arm only receives the stage-1 disturbance curriculum.
- `stage2`: switch opens at iteration 0; assumes a stage-1 dog checkpoint is loaded via `--stage1_ckpt_path`.
- `two_stage`: switch opens at `default_switch_iteration` (2000 if `--resume`, else 8000); a single run trains stage-1 then transitions to stage-2.

### Stage-2 Locomotion Freezing

In stage-2 the dog policy is *frozen by default*
(`DogRunnerArgs.stage2_freeze_loco_policy = True`). When frozen, the `Runner`:

- Sets `dog_model.requires_grad_(False)` and calls `dog_model.eval()`.
- Skips `alg_dog.compute_returns()` and `alg_dog.update()`; calls `alg_dog.clear_storage()` to keep memory bounded.
- Loads only shape-compatible actor/adaptation/std weights via `_load_matching_state_dict`, skipping `critic_body.*` (the critic head can shape-mismatch the new dog obs width). Non-critic shape mismatches still raise.

Unfreeze with `--stage2_unfreeze_loco_policy`; optionally pass
`--stage2_loco_learning_rate` to lower the dog LR when both policies update
together.

`scripts/auto_train.py::apply_dog_checkpoint_command_limits` copies the dog
checkpoint's `Cfg.commands.limit_*` fields into the active `Cfg.commands`
before training starts, so arm-policy plan outputs are clipped to the same
ranges the dog policy was trained against. If the sibling `parameters.pkl`
is missing or unparsable, a warning is printed and the current limits are
kept; check the printed banner in stage-2 startup output to confirm which
limits are actually in use.

## Running Training and Play

Activate the env first (see "Debug Checklist").

Train two-stage from scratch (stage1 → stage2 in one run):

```bash
python scripts/auto_train.py --train_stage two_stage --dyna_gait --traj_track --run_name rd_two_stage
```

Train stage-2 only from a stage-1 dog checkpoint (default freezes dog policy):

```bash
python scripts/auto_train.py \
  --train_stage stage2 --dyna_gait --traj_track \
  --stage1_ckpt_path /path/to/checkpoints_dog/iter-<N>.pt \
  --run_name rd_stage2
```

Stage-2 with both policies still trainable:

```bash
python scripts/auto_train.py --train_stage stage2 \
  --stage1_ckpt_path .../checkpoints_dog/iter-<N>.pt \
  --stage2_unfreeze_loco_policy --stage2_loco_learning_rate 1e-4 ...
```

Other useful flags: `--debug` (4 envs, video on, wandb disabled, shorter
schedule), `--resume` (treat run as resuming, shrinks two-stage switch
iteration to 2000), `--robot {go1,go2}`, `--headless`, `--num_envs`,
`--stage2_ckpt_path` (resume an arm checkpoint).

Joystick play examples — see "Play Scripts" / "Joystick Commands" for the
full contract:

```bash
python scripts/play_by_joy.py --logdir <run_dir>          # stage2 play
python scripts/play_by_joy.py --logdir <run_dir> --stage1_only
python scripts/play_by_joy.py --logdir <run_dir> --lock_arm
```

## Policy and Action Layout

RoboDuet commonly runs separate dog and arm policies:

- Dog policy reads `env.get_dog_observations()`.
- Arm policy reads `env.get_arm_observations()`.
- `HistoryWrapper.step(action_dog, action_arm)` concatenates both action tensors before calling `WBCEnv.step()`.

Action layout is config-dependent:

- Dog actions occupy the first `cfg.dog.num_actions_loco` dimensions.
- Arm actions follow after the dog slice.
- Trajectory tracking and dynamic gait modes can extend command/action layout. See `dev_log.md` before changing those paths.

### Stage-2 Trajectory + Dynamic Gait MDP

For `traj_track=True` and `dyna_gait=True`, keep the arm policy and dog command layout aligned with `config/core.py` and `WBCEnv.plan()`. The arm policy action width is `cfg.arm.num_actions_arm_cd`: first the 6 arm joint actions, then plan actions consumed by `WBCEnv.plan()`. The current plan-action layout is 9 dimensions:

- `0:3`: delta dog velocity command (`x_vel`, `y_vel`, `yaw_vel`).
- `3:6`: dog body pose command (`body_pitch`, `body_roll`, `body_height`).
- `6:9`: dynamic gait command (`gait_frequency`, `stance_width`, `stance_length`).

`footswing_height` and `gait_duration` are not arm-policy actions in this mode. They are fixed when sent to the dog policy:

- `footswing_height = 0.06`
- `gait_duration = 0.49`

All arm-policy commands sent to the dog policy should pass through the dog-command smoothing path in `WBCEnv` and be clipped/mapped using the dog policy command ranges. Stage-2 training loads command ranges from the dog policy `parameters.pkl` in `scripts/auto_train.py`; if this fails, print and inspect the runtime values. Limit violations use one `arm_control_limits` reward scale. Smoothness can be weighted separately by command group with:

- `cfg.wbc.trajectory.dog_command_smoothness_weight_delta_vel`
- `cfg.wbc.trajectory.dog_command_smoothness_weight_body_pose`
- `cfg.wbc.trajectory.dog_command_smoothness_weight_gait`

Arm observations in trajectory mode intentionally include the full previous arm-policy action width (`cfg.arm.num_actions_arm_cd`), not only the 6 joint actions. This keeps body-height and gait plan dimensions observable. Arm observations also include arm DOF velocity. Foot contact states are not normal arm observations; they are privileged-only for the arm policy.

Trajectory progress observations are split by meaning:

- `trajectory_progress_index`: trajectory-state scalar based on walked waypoints, `(traj_progress_idx + 1) / traj_num_waypoints`, guarded to 0 before any point has been visited. For `traj_type == point`, this value must be 0.
- `trajectory_completion_time_command`: command scalar equal to the sampled `traj_target_time`. It belongs with command-like arm-policy inputs.
- `remaining_time`: may still be computed from `traj_target_time - traj_elapsed_time` for timing logic, but do not use it as trajectory progress.

### Dog Command Smoothing Buffers

When the arm policy emits plan actions, raw and smoothed dog commands are
tracked in three buffers (all shape `(num_envs, dog_num_commands)`):

- `plan_actions_raw`: the un-scaled policy output for the plan slice; used by `_reward_arm_control_limits` to penalize `|raw| > 1`.
- `dog_command_plan_targets`: the per-step plan target after clipping to `cfg.commands.limit_*`.
- `dog_command_plan_smoothed`: EMA of `dog_command_plan_targets` with coefficient `cfg.wbc.trajectory.dog_command_smoothing_alpha` (default 0.2); this buffer is what gets written to `commands_dog` and consumed by the dog policy.

`_smooth_dog_command_values(indices, values)` is the single place that
performs the EMA write; never set `commands_dog[:, plan_indices]` directly
from inside `plan()` — always route through this helper so target/smoothed
stay coherent.

On reset, `_reset_dog_command_smoothing(env_ids)` neutralizes stale plan
state from the previous episode:

- velocity is preserved (already set by the caller from `user_vel_cmd`),
- `body_pitch/roll/height` are set to 0,
- `gait_frequency/stance_width/stance_length` are set to the midpoints of their `cfg.commands.limit_*` ranges (only when `cfg.commands.use_dynamic_gait`),
- `footswing_height` is set to 0.06 and `gait_duration` to 0.49,
- then `dog_command_plan_targets` and `dog_command_plan_smoothed` are snapped to the resulting `commands_dog`.

This avoids the EMA blending the new policy's first plan output with the
previous episode's final plan output. The call lives inside
`_resample_trajectory_commands` only — do not also call it from
`_arm_reset_hook`.

### Key Runtime Buffers (Stage-2 Trajectory)

Useful when reading or modifying trajectory code:

- `traj_pos_world` / `traj_quat_world`: per-env trajectory waypoints in world frame, shape `(num_envs, traj_num_waypoints, 3 or 4)`. For `point` type, all waypoints equal the sampled start point.
- `traj_progress_idx`: long tensor, current waypoint index per env, driven by `traj_elapsed_time / traj_target_time`.
- `traj_type`: long tensor encoding `{line:0, s_curve:1, circle:2, point:3}`. Reward branches off this.
- `traj_visited_mask`: bool tensor marking waypoints that have been the current target at least once. Drives `get_trajectory_error_sum` averaging and `get_trajectory_progress_index_obs` guarding.
- `traj_curriculum_level` / `_traj_curriculum_params(env_ids)`: per-env integer level in `[0, cfg.wbc.trajectory.curriculum_levels - 1]`; returns `length`, `s_curve_amplitude`, and `orientation_scale` interpolated by `level / max_level`. The orientation scale shrinks the sampled roll/pitch/yaw ranges around 0, so level 0 produces start_quat ≈ base_quat and higher levels approach the full `cfg.arm.commands.roll_ee/pitch_ee/yaw_ee` ranges.
- `traj_target_time` / `traj_elapsed_time`: per-episode sampled completion time and accumulator. `arm_time_buf` drives elapsed time and is zeroed at reset.
- `traj_complete_buf` / `traj_episode_success_buf`: termination/curriculum-advance signals; success requires reaching the final waypoint with pos and rot errors under `completion_pos_threshold` / `completion_rot_threshold`.

## Reset Rules

The critical reset path is `LeggedRobot.reset_idx(env_ids)`.

Intended order:

```python
self._resample_commands(env_ids)
self._arm_reset_hook(env_ids)
self._randomize_dof_props(env_ids, self.cfg)
self._arm_post_dof_randomization_hook(env_ids)

if self.cfg.domain_rand.randomize_rigids_after_start:
    self._randomize_rigid_body_props(env_ids, self.cfg)
    self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

self._reset_dofs(env_ids, self.cfg)
self._reset_root_states(env_ids, self.cfg)
self._arm_post_reset_refresh_hook(env_ids)
```

Do not write sim state after `_reset_root_states()` except pure bookkeeping. In particular, `_arm_post_reset_refresh_hook()` must not call:

- `set_dof_state_tensor_indexed()`
- `set_actor_root_state_tensor_indexed()`
- `set_actor_rigid_body_properties(..., recomputeInertia=True)`

Writing DOF or actor rigid-body props after root reset can desynchronize IsaacGym articulation/graphics state and make reset height/orientation look ignored.

## Rendering Gotchas

RoboDuet reset happens inside `post_physics_step()` after simulation results for the current frame have already been refreshed:

```python
self.gym.refresh_actor_root_state_tensor(self.sim)
...
self.check_termination()
self.compute_reward()
env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
self.reset_idx(env_ids)
self.compute_observations()
```

If high-z/random-orientation reset is not visible in GUI, distinguish graphics timing from physics state:

- Print/check `root_states` after `_reset_root_states()`.
- Temporarily refresh the viewer after `reset_idx()` for debugging.
- Keep extra viewer refreshes guarded by `not self.headless`.

Headless training should not render.

## Domain Randomization

There are two safe categories.

### Per-Episode Buffer/Control DR

Safe at reset because it updates tensors used by control/observations:

- Dog and arm `Kp_factors`
- Dog and arm `Kd_factors`
- Dog and arm `motor_strengths`
- Dog and arm `motor_offsets`
- DOF reset positions/velocities
- Root pose/velocities
- Command sampling

Arm Kp/Kd/strength/offset randomization is still per-episode:

```python
self._randomize_dof_props(env_ids, self.cfg)
self._arm_post_dof_randomization_hook(env_ids)
```

`WBCEnv._arm_post_dof_randomization_hook()` calls `_randomize_arm_dof_props()` and overwrites the arm slice using `cfg.domain_rand.stage1_arm` or `cfg.domain_rand.stage2_arm`.

Arm initial DOF noise belongs in `_arm_post_dof_reset_hook()`, before the single reset-time `set_dof_state_tensor_indexed()` inside `_reset_dofs()`.

### Per-Env Rigid Body Props DR

Rigid-body property randomization should be applied before simulation starts, during actor creation.

Dog/base mass and COM already follow this pattern:

- Sample per-env buffers.
- Apply values inside `_process_rigid_body_props(props, env_id)`.
- `_create_envs()` calls `set_actor_rigid_body_properties()` once during actor creation.

Arm link mass/COM should follow the same pattern:

- `WBCEnv._process_rigid_body_props(props, env_id)` calls the base method first.
- It initializes arm default masses/COMs from `props`.
- It samples per-env arm link mass scales and COM offsets.
- It edits `props[body_idx].mass` and `props[body_idx].com`.
- `_create_envs()` applies the final props before sim starts.

Do not randomize arm link mass/COM during episode reset by calling `set_actor_rigid_body_properties()` on an existing actor.

Tradeoff: arm link mass/COM is per-env, not per-episode. This is intentional for reset stability.

Arm mount translation/rotation randomization is also per-env. `_generate_arm_mount_asset_files()` creates
URDF asset buckets before actor creation and edits the fixed mount joint's `xyz` and `rpy`. The ranges are
controlled independently by `randomize_mount_position` / `mount_position_range` and
`randomize_mount_rotation` / `mount_rpy_range`; RPY values are radians. Bucket 0 always keeps the nominal
mount transform. `arm_mount_tfs` must contain the exact `[x, y, z, roll, pitch, yaw]` written to each env's
URDF because it is exposed through privileged observations. Do not try to resample a fixed-joint mount TF
during episode reset.

## Config Gotchas

For play/eval, `scripts/load_policy.py` loads `parameters.pkl` and can overwrite source defaults. If config edits do not seem to work, print runtime `Cfg` after checkpoint loading.

For stage-2 training, `scripts/auto_train.py` also needs to reason from the dog policy checkpoint config. When `DogRunnerArgs.ckpt_path` points to a dog policy checkpoint, `auto_train.py` should load the sibling `parameters.pkl`, copy the dog command `limit_*` fields into the active `Cfg.commands`, and print both the successful load path and the specific values used by stage 2.

`go1_gym_learn/ppo_cse_automatic/__init__.py` uses `ckpt_path=None` as the non-resume state for arm and dog runner args. Do not reintroduce a separate `resume` field. A non-`None` `ckpt_path` means load/resume.

When freezing the stage-1 dog policy for stage-2 training, the dog critic is not needed. If a checkpoint critic shape mismatches the current dog observation shape, load only shape-compatible actor/adaptation/std weights and skip `critic_body.*`; non-critic shape mismatches should still be treated as real errors.

Common play-time overrides include:

- `Cfg.terrain.mesh_type = "plane"`
- `Cfg.env.num_envs = 1`
- many domain randomization flags disabled
- terminal conditions disabled
- `Cfg.control.control_type = "M"`

## Play Scripts

`play` scripts are interactive operator tools, not bounded evaluation jobs.

- `scripts/play_by_joy.py` should open the IsaacGym viewer (`headless=False`) and run until interrupted. Do not reintroduce `--headless` or `--num_eval_steps` there unless the user explicitly asks for an evaluation mode.
- `--stage1_only` means pure stage-1 play: keep `global_switch` closed, do not load/call the arm policy, and use `env.env.stage1_arm_play_intensity` as the direct arm disturbance intensity. Do not synthesize training ramp iterations in play.
- `--lock_arm` is stricter than stage-1 disturbance: it disables the stage-1 arm curriculum/disturbance and sends fake/zero arm actions.
- For checkpoint-backed play, always reason from the runtime `Cfg` loaded by `scripts/load_policy.py`; source config defaults may not be active.

## Joystick Commands

`scripts/play_by_joy.py` uses `COMMAND_KEYS` for dog/arm command indices and `_COMMAND_INDEX` for reverse lookup.

- For D-pad axis combos, the dictionary key name such as `"dpad_y:left"` is only a label. The physical trigger is the JoyLink axis value sign.
- Use `axis_direction` for which axis sign triggers the edge, and `delta` for how much the command changes. Do not use one `direction` field to mean both trigger sign and command sign.
- Apply dog defaults by runtime `env.commands_dog.shape[1]`, not by only checking `cfg.commands.use_dynamic_gait`; checkpoint command width and config flags can disagree.
- Actual gait frequency is consumed in `LeggedRobot._step_contact_targets()` only when `cfg.commands.use_dynamic_gait=True`. If it is false, `commands_dog[:, 6]` may display but the gait clock uses fixed defaults.
- Even with dynamic gait enabled, when `torch.norm(commands_dog[:, :3]) < 0.1`, contact targets are forced to stand phase. Do not judge gait frequency behavior while all velocity commands are zero.

## Rerun Visualization

Reusable Rerun telemetry lives in `go1_gym/utils/viz.py`.

- Use `add_rerun_args(parser)` to add common `--rerun*` CLI flags.
- Use `make_rerun_logger(args, ...)` or `RerunLogger` directly, then call `rerun_logger.log(env)` after `env.step(...)`.
- The logger is optional: without `--rerun`, it is inert; if `rerun-sdk` is missing or an older API is unsupported, play should continue.
- Default visible panes are `vx`, `vy`, `pitch`, and `joint torque`. `height` and `roll` panes are created hidden by default. Full `base` and `command` panes are only created/logged with `--rerun_extra_panes`.
- Default torque logging is the first 12 joints. Use `--rerun_torque_joints` to change this and `--rerun_joint_state` to also log joint positions/velocities.
- `--rerun_window_seconds` controls the visible sliding time window where the installed Rerun blueprint API supports `VisibleTimeRange`; old Rerun versions may require manual viewer configuration.

## Reset Curriculum

`LeggedRobot._init_reset_curriculum()` / `_update_reset_curriculum()` / `_get_reset_curriculum_range()` implement a curriculum that scales reset randomization ranges (z, yaw, pitch, roll) from a small initial fraction up to the full configured value, gated on locomotion tracking success.

### Key parameters (`cfg.terrain`)

| Parameter                                  | Default                 | Meaning                                                                                         |
| ------------------------------------------ | ----------------------- | ----------------------------------------------------------------------------------------------- |
| `reset_curriculum`                         | `False` (`True` in WBC) | Enable/disable the curriculum                                                                   |
| `reset_curriculum_initial_fraction`        | `0.0`                   | Starting intensity as fraction of the maximum reset range                                       |
| `reset_curriculum_tracking_threshold`      | `0.5`                   | Required EMA of the normalized joint linear/angular tracking score                              |
| `reset_curriculum_tracking_ema_alpha`      | `0.05`                  | EMA smoothing factor, updated once per training iteration                                       |
| `reset_curriculum_stability_iterations`    | `100`                   | Consecutive above-threshold iterations required before reset randomization starts growing        |
| `reset_curriculum_growth_iterations`       | `5000`                  | Iterations used to linearly grow intensity from `initial_fraction` to `1.0`                     |

### Intensity formula

```
# Before curriculum starts:
intensity = initial_fraction

# After curriculum starts (elapsed = global_switch.count - start_iteration):
progress = min(1.0, elapsed / growth_iterations)
intensity = initial_fraction + (1 - initial_fraction) * progress
```

### Tracking gate

For every completed episode, linear and angular tracking rewards are divided by
their active reward scales and clipped to `[0, 1]`. The per-env score is:

```text
tracking_score = min(normalized_linear_tracking, normalized_angular_tracking)
```

Reset batches from the same `global_switch.count` are accumulated together. The
EMA is updated once when the next training iteration begins, so episode length
and reset frequency do not change the curriculum's time scale. The gate opens
only after the EMA remains above `reset_curriculum_tracking_threshold` for
`reset_curriculum_stability_iterations` consecutive iterations.

### Logged metrics

- `reset_curriculum_lin_tracking_score`
- `reset_curriculum_ang_tracking_score`
- `reset_curriculum_tracking_score`
- `reset_curriculum_tracking_score_ema`
- `reset_curriculum_stable_iterations`
- `reset_curriculum_started`
- `reset_curriculum_intensity`

### The RoboDuet profile sets `reset_curriculum = True` with `initial_fraction = 0.1`

This means WBC training starts at 10% of the configured reset-randomization ranges and grows toward the full ranges after the success gate opens.

## Performance Metrics

`LeggedRobot._update_performance_metrics()` accumulates physical task metrics
before reward computation. They do not call reward functions and are not
affected by reward scales or reward shaping. At reset they are normalized per
episode and logged under `Performance/*` by both dual-policy and unified
runners. Reset batches are weighted by their number of completed episodes.

Locomotion metrics include velocity-command MAE/RMSE, roll/pitch and body-rate
RMS, height RMSE, contact-foot slip speed, mechanical locomotion power, early
termination rate, and episode duration. Trajectory mode also logs current EE
position RMSE in meters and quaternion geodesic orientation RMSE in radians.

Metric names include their units, for example `Performance/vx_mae_mps`,
`Performance/yaw_rate_rmse_rad_s`, and `Performance/base_height_rmse_m`.

## Trajectory Tracking Notes

Trajectory type is configured in Python config, not argparse. Do not add a `--traj_type` CLI flag unless the user explicitly asks for CLI control.

Supported sampled types include `line`, `s_curve`, `circle`, and `point`. For all trajectory types, sample the trajectory start around the current end-effector or grasper position with a random 3D direction and radius, and randomize the starting orientation from the configured EE roll/pitch/yaw ranges. Avoid special-casing only point starts; keep line/circle/s-curve starts consistent with point.

Point trajectory mode means the trajectory window is filled with the same sampled point. It should use L2 tracking loss/reward semantics for point tracking. Do not remove the exponential trajectory tracking reward for line/circle/s-curve modes; those non-point modes should keep the existing exponential tracking behavior.

For point mode, progress-like trajectory observations should be 0 because there is no meaningful path progress along a point target.

Point mode is intentionally used as the **early-curriculum precursor** to full
trajectory tracking: it teaches the robot to autonomously move its base so the
EE can reach targets sampled *outside* the current arm workspace. Its reward
is the raw negative L2 error (no `exp(-·)`), which keeps the gradient magnitude
roughly constant even when the EE is far from the target — `exp(-L2)`
saturates to 0 in the far-field and starves the policy of signal during the
exact regime we want to train. Do not "normalize" the point reward to `(0, 1]`
to match the non-point branch without first confirming the curriculum still
converges from far starts. When mixing point with line/circle/s_curve in a
single batch, expect the L2 reward to dominate; if that is undesired, split
into separate reward terms with their own scales rather than rewriting the
point reward.

Trajectory start orientation follows the same curriculum as length and
S-curve amplitude (see `_traj_curriculum_params` / `traj_curriculum_level`).
`sample_trajectory_commands(..., orientation_scale=…)` multiplies the sampled
roll/pitch/yaw by that per-env scale, so a freshly-spawned env starts with
`start_quat ≈ base_quat` and progressively unlocks the full configured EE
orientation range as it succeeds. There is no separate orientation
curriculum config; reuse `cfg.wbc.trajectory.curriculum_levels`.

## Reward and Curriculum Notes

Reward scale signs matter.

- Penalty reward scales are usually negative. For `action_rate`, a more negative absolute scale means a stronger penalty; smaller early-training penalty should be closer to zero.
- The current action-rate curriculum is intended to switch from weaker to stronger penalty only after stage-1 arm disturbance intensity exceeds its threshold and the configured delay has elapsed.
- “Active reward scale” means the runtime scale after curriculum/global-switch logic, not necessarily the raw class default in config.
- Reset randomization curricula for root `z`, roll, pitch, and yaw should be justified by locomotion tracking progress. If instability is actually caused by collisions or play-time config mismatch, prefer the smaller targeted fix over adding curriculum complexity.

## WBC Benchmark (`benchmark/wbc/`)

Stage-2 SE(3) trajectory-tracking evaluation, integrated into the benchmark
pipeline (reuses the Accumulator, per-policy env slicing, HTML reports, and
comparison tool from `benchmark/`).

```bash
python -m benchmark.cli --wbc --logdirs runs/<date>/<run> --headless
python -m benchmark.cli --wbc --logdirs runs/A runs/B --names v1 v2 --headless
```

Sweeps all 36 curriculum cells (6x6 grid), accumulating per-cell metrics:
EE tracking error, d_lat, timing, rho, base utilisation, motor power, and
finite-difference acceleration/jerk (EE, base, arm joints).

Writes `results.json`, `metadata.json` and an HTML report to
`benchmark/results/<timestamp>/`.

### Key things

- **Held-out means `bank_seed`.** The trajectory bank is deterministic at env
  construction, so training's seed (0) reproduces training trajectories.
  `bank_seed` defaults to 12345.
- **`WBCAccumulator` stores velocity time series** for offline
  finite-difference smoothness computation — acceleration/jerk are
  single-step derivatives and can't be accumulated per-step.
- **`wbc_eval_loop` batches policies correctly**: arm inference for all
  handles → `plan()` once → dog inference for all handles → `step()` once.
  Per-policy stepping would silently overwrite `plan()`'s env-wide
  `commands_dog` writes.
- **`WBCEnv.load_custom_trajectories`** is the eval-side injection point
  (probe trajectories). It routes through the same `_place_and_reset_trajectories`
  the bank path uses, so probe `s` / `d_lat` / `timing_err` are comparable to
  training episodes.

## Debug Checklist

Related skills: isaac-skill

conda env for this project is: isaacgym. Agent can use:

```
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/simon/Projects/WBC/RoboDuet
```

to activate the env and debug.

When reset height/orientation looks wrong:

1. Print runtime `Cfg.init_state.pos`, `Cfg.terrain.z_init_range`, `roll_init_range`, `pitch_init_range`, and `yaw_init_range`.
2. Confirm `_reset_root_states()` runs and writes expected `root_states`.
3. Check hooks after `_reset_root_states()` for DOF/root/rigid-body writes.
4. Temporarily refresh viewer after reset to isolate graphics timing.
5. Keep rigid-body property writes out of episode reset.

When arm randomization looks missing:

1. Check `cfg.env.stage1_arm_init_dof_pos_noise`.
2. Check `cfg.domain_rand.stage1_arm` or `stage2_arm`.
3. Confirm `_arm_post_dof_randomization_hook()` runs after generic `_randomize_dof_props()`.
4. Confirm arm link mass/COM buffers vary across envs, not episodes.
