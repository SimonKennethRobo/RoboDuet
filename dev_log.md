# Dev Log: Trajectory MDP Changes

# May 20 16:37

## Goal

Extend the original dual-stage legged manipulator task from single 6D EE goal tracking to task-space trajectory tracking, while preserving the legacy goal-pose behavior behind a switch.

## Configuration

- Added `Cfg.arm.trajectory` in `go1_gym/envs/automatic/legged_robot_config.py`.
- Main trajectory config fields:
  - `enabled`: switch for trajectory tracking mode.
  - `window_offsets`: exponential step offsets, currently `[0, 1, 2, 4, 8, 16, 32, 64]`.
  - `num_waypoints`: fixed trajectory discretization length.
  - `start_radius`: random start offset around current EE pose.
  - `length`: line/S trajectory length.
  - `s_curve_amplitude`, `s_curve_frequency`: S-curve shape.
  - `circle_radius`, `circle_turns`: circular trajectory shape.
  - `completion_time_range`: randomized target completion time.
  - `completion_pos_threshold`, `completion_rot_threshold`: completion criteria.
  - `delta_vel_limit`: small residual velocity limits for arm-to-loco command.
  - `user_cmd_mode`: `"zero"` by default, with `"random"` support.
- Added hybrid reward scale keys for trajectory terms:
  - `traj_track`
  - `trajectory_current_tracking`
  - `trajectory_completion_time`
  - `arm_delta_vel_cmd`
  - `ee_smoothness`

## Trajectory Command Generation

Implemented in `go1_gym/envs/automatic/legged_robot.py`.

- Added `_resample_trajectory_commands(env_ids)`.
- Every reset/resample generates one random trajectory from the current EE pose neighborhood.
- Supported trajectory families:
  - line
  - S-curve
  - circle
- Trajectory points are stored in world frame:
  - `traj_pos_world`
  - `traj_quat_world`
- The current implementation keeps target orientation fixed at the initial EE orientation for all trajectory types.
- `user_vel_cmd` is sampled through `_resample_user_commands()`.
  - Default mode is zero user command.
  - Random mode samples from configured `user_lin_vel_x`, `user_lin_vel_y`, and `user_ang_vel_yaw`.

## Body-Frame Pose and Twist Helpers

Implemented helper functions in `legged_robot.py`.

- `_quat_xyzw_to_rot6d()`
- `_pose_world_to_body_9d()`
- `get_ee_pose_body_9d()`
- `get_ee_twist_body()`
- `_trajectory_points_body_9d()`

The trajectory command and tracking observations use body-frame `pos3 + rot6d6`.

EE twist is defined as EE velocity relative to base velocity, expressed in the base/body frame:

- linear: `ee_lin_vel_world - base_lin_vel_world`, rotated into body frame
- angular: `ee_ang_vel_world - base_ang_vel_world`, rotated into body frame

## Progress and History Tracking

Implemented in `legged_robot.py`.

- `_step_traj_track()` updates:
  - `traj_elapsed_time`
  - `traj_progress_idx`
  - `traj_complete_buf`
  - final pose errors
- For trajectory-level error, the env stores both actual and target body-frame 9D pose at the visited progress index:
  - `traj_ee_pose_body_history`
  - `traj_target_body_history`
  - `traj_visited_mask`
- `get_trajectory_error_sum()` computes the mean error over all visited waypoint indices.

## Observations

### Loco Policy

In trajectory mode, loco observations include current EE body-frame 9D pose.

### Arm Policy

In trajectory mode, arm observations include:

- arm joint position offsets and previous arm actions
- base height
- foot contact states
- base angular velocity
- current loco velocity command
- optional current gait command when dynamic gait is enabled
- current EE body-frame 9D pose
- EE relative body-frame twist
- trajectory window, with each waypoint as `pos3 + rot6d6`
- remaining time

Legacy `lpy/rpy` goal observation remains active when trajectory mode is disabled.

## Actions and Command Composition

Implemented in `go1_gym/envs/automatic/__init__.py`.

Arm policy action layout now depends on enabled modes:

- Legacy only: arm action stays at the original layout.
- Trajectory tracking only:
  - `6 arm action + 3 residual loco velocity command`
- Dynamic gait only:
  - `6 arm action + 7 body/gait plan command`
- Trajectory tracking + dynamic gait:
  - `6 arm action + 3 residual loco velocity command + 7 body/gait plan command`

For trajectory tracking:

```python
commands_dog[:, :3] = user_vel_cmd + arm_delta_vel_cmd
```

With the default `user_cmd_mode = "zero"`, stage2 does not randomly generate external loco velocity commands for trajectory tracking. Loco velocity comes from the arm policy residual command.

## Rewards

Added reward functions in `go1_gym/envs/rewards/rewards.py`.

- `_reward_traj_track()`
  - trajectory-level dense reward over all visited waypoint errors.
- `_reward_trajectory_current_tracking()`
  - current waypoint tracking reward.
- `_reward_trajectory_completion_time()`
  - rewards completion only when final time/progress and final pose thresholds are satisfied.
- `_reward_arm_delta_vel_cmd()`
  - penalizes normalized arm residual velocity command.
- `_reward_ee_smoothness()`
  - penalizes EE relative twist acceleration.

## Training Entrypoints

### `scripts/auto_train.py`

- Added trajectory tracking CLI switch.
- Current name is:

```bash
--traj_track
```

- Dynamic gait and trajectory tracking can be enabled together.
- Debug mode was changed to use fewer envs.
- Default robot in the current file is `go2`.

### `scripts/unified_train.py`

- Added trajectory tracking CLI switch:

```bash
--traj_track
```

- Dynamic gait and trajectory tracking can be enabled together.
- Updates `Unified2AC_Args.num_actions_arm` when action dimensions change.

## Visualization and Video Overlay

Implemented mostly in `legged_robot.py`.

### Recorded Video

Video frames now overlay policy/debug values through:

- `_policy_command_overlay_lines()`
- `_overlay_policy_text()`
- `_overlay_policy_trajectory()`

Displayed values include:

- final loco command
- user command
- arm-to-loco residual command
- dynamic gait/body extra command
- trajectory progress and final error
- arm action values
- base state summary

Trajectory is projected into the camera frame and drawn into the recorded video.

### Non-Headless Viewer

Viewer overlay rendering was centralized through:

- `_draw_viewer_overlays()`
- `_draw_policy_trajectory()`
- `_draw_viewer_polyline()`

Trajectory viewer visualization:

- visited trajectory: yellow
- future trajectory: cyan
- current target: larger cyan sphere and axes
- final point: magenta sphere and axes
- EE-to-target error line: red
- lookahead window points: small cyan spheres

Keyboard and joystick wrappers call `_draw_viewer_overlays()` before `draw_viewer()` so the lines are rendered in the current frame.

## Runner Compatibility

Updated `go1_gym_learn/ppo_cse_automatic/__init__.py`.

- Fixed arm action slicing when `num_plan_actions > 0`.
- The runner now sends only the first physical arm action dimensions into `env.step()` when stage2 is active and plan actions exist.

## Current Usage Examples

Dynamic gait only:

```bash
python scripts/auto_train.py --sim_device cuda:0 --dyna_gait --debug
```

Trajectory tracking only:

```bash
python scripts/auto_train.py --sim_device cuda:0 --traj_track --debug
```

Dynamic gait + trajectory tracking:

```bash
python scripts/auto_train.py --sim_device cuda:0 --dyna_gait --traj_track --debug
```

Start directly in stage2:

```bash
python scripts/auto_train.py --sim_device cuda:0 --dyna_gait --traj_track --debug --wo_two_stage
```

## Known Issues / Follow-Ups

- The recorded-video 3D-to-camera projection is implemented but still needs a short IsaacGym recording check to confirm matrix direction and line placement.
- Target orientation is fixed along the trajectory; future work can add tangent-aligned or curriculum-controlled orientation profiles.
