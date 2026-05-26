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
