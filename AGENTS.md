# AGENTS.md

Guidance for coding agents working in this RoboDuet repository.

## Project Shape

RoboDuet is a dual-policy legged manipulation environment built on IsaacGym.
The active environment code is under `go1_gym/envs/roboduet/`.

Core files:

- `go1_gym/envs/roboduet/legged_robot.py`: base IsaacGym task, sim stepping, reset, root/DOF tensors, terrain, actor creation, common dog observations/rewards, and generic domain randomization.
- `go1_gym/envs/roboduet/wbc_env.py`: RoboDuet arm/WBC extension, arm commands, trajectory tracking, arm observations/rewards, stage switching, and arm-specific domain randomization.
- `go1_gym/envs/roboduet/wbc_env_config.py`: RoboDuet config source of truth. It materializes values from Go1/WTW/asset recipes.
- `go1_gym/envs/roboduet/wbc_env_wrapper.py`: interactive wrappers and `HistoryWrapper`.
- `scripts/load_policy.py`: play/eval loader. It loads `parameters.pkl`, overwrites `Cfg`, then applies play-time overrides.
- `scripts/auto_train.py`: main training entrypoint.
- `dev_log.md`: trajectory MDP development notes.
- `roboduet_architecture_rl_notes.md`: longer architecture/RL handoff note.

## Policy and Action Layout

RoboDuet commonly runs separate dog and arm policies:

- Dog policy reads `env.get_dog_observations()`.
- Arm policy reads `env.get_arm_observations()`.
- `HistoryWrapper.step(action_dog, action_arm)` concatenates both action tensors before calling `WBCEnv.step()`.

Action layout is config-dependent:

- Dog actions occupy the first `cfg.dog.num_actions_loco` dimensions.
- Arm actions follow after the dog slice.
- Trajectory tracking and dynamic gait modes can extend command/action layout. See `dev_log.md` before changing those paths.

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

## Config Gotchas

For play/eval, `scripts/load_policy.py` loads `parameters.pkl` and can overwrite source defaults. If config edits do not seem to work, print runtime `Cfg` after checkpoint loading.

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

## Reward and Curriculum Notes

Reward scale signs matter.

- Penalty reward scales are usually negative. For `action_rate`, a more negative absolute scale means a stronger penalty; smaller early-training penalty should be closer to zero.
- The current action-rate curriculum is intended to switch from weaker to stronger penalty only after stage-1 arm disturbance intensity exceeds its threshold and the configured delay has elapsed.
- “Active reward scale” means the runtime scale after curriculum/global-switch logic, not necessarily the raw class default in config.
- Reset randomization curricula for root `z`, roll, pitch, and yaw should be justified by locomotion tracking progress. If instability is actually caused by collisions or play-time config mismatch, prefer the smaller targeted fix over adding curriculum complexity.

## Debug Checklist

Related skills: isaac-skill

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
