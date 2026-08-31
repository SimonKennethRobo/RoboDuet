# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import copy
import os
import sys
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import (
    get_axis_params,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

assert gymtorch

import numpy as np
import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.envs.base.curriculum import command_curriculum_bounds, command_curriculum_local_range
from go1_gym.envs.config import ConfigNode
from go1_gym.response import (
    DECISION_CHANNEL_UNITS,
    GAIT_FREQUENCY,
    EnvGrouping,
    ExcitationSampler,
    PhaseResidualEstimator,
    ReferenceModel,
    build_channels,
)
from go1_gym.response.reward_terms import (
    phase_variance,
    settled_mask,
    soft_gate_from_events,
    steady_gain,
)
from go1_gym.utils import global_switch, quaternion_to_rpy
from go1_gym.utils.math_utils import get_scale_shift, quat_apply_yaw
from go1_gym.utils.terrain import Terrain

#: The reward terms the adaptive command curriculum reads as its progress
#: signal.  R6 invariant 2: only the original tracking terms may appear here --
#: a consistency reward in this list would let the curriculum stall at low
#: difficulty because consistency is hard everywhere, not because the command
#: range is too wide.
CURRICULUM_PROGRESS_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "tracking_contacts_shaped_force",
    "tracking_contacts_shaped_vel",
)

#: Reward terms introduced by the response-consistency work (R4).  Asserted to
#: be disjoint from CURRICULUM_PROGRESS_REWARDS at startup, which is the whole
#: enforcement mechanism for the invariant above.
CONSISTENCY_REWARDS = ("ref_tracking", "phase_variance", "steady_gain")


class LeggedRobot(BaseTask):
    def __init__(
        self,
        cfg: ConfigNode,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        eval_cfg=None,
        initial_dynamics_dict=None,
        graphics_device_id=None,
    ):

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.ee_idx = 23
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None:
            self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)
        self.num_actions_arm = cfg.arm.num_actions_arm
        self.num_actions_loco = cfg.dog.num_actions_loco

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg, graphics_device_id)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
        self._init_reset_curriculum()
        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0
        self.fixed_cam = False

        if not self.headless:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "fixed_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "save_image")

    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            if self.fixed_cam:  # fixed camera to tracking the robot
                cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                cam_pos = cam_target + gymapi.Vec3(1, 1, 1)
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "fixed_cam" and evt.value > 0:
                    self.fixed_cam = not self.fixed_cam

                elif evt.action == "save_image" and evt.value > 0:
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                    cam_pos = cam_target + gymapi.Vec3(0.8, 0.8, 0.8)
                    self.gym.set_camera_location(self.rendering_camera, self.envs[0], cam_pos, cam_target)
                    video_frame = self.gym.get_camera_image(
                        self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR
                    )
                    video_frame = video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
                    import matplotlib.pyplot as plt

                    # Save the image as now.png
                    plt.imsave("now.png", video_frame)

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self._draw_viewer_overlays()
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self._draw_viewer_overlays()
                self.gym.poll_viewer_events(self.viewer)

    def draw_coord_pos_quat(self, x, y, z, quat, scale=0.1):
        draw_scale = scale
        pos = gymapi.Vec3(x, y, z)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
        axes_geom = gymutil.AxesGeometry(draw_scale, pose)
        axes_pose = gymapi.Transform(pos, r=None)
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], axes_pose)

    def draw_sphere_and_axes(self, position, quaternion, sphere_radius, sphere_color, scale=0.1):
        sphere_geom = gymutil.WireframeSphereGeometry(sphere_radius, 4, 4, None, color=sphere_color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(*position), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
        self.draw_coord_pos_quat(*position, quaternion, scale)

    def _draw_base_ori_coord(self):
        x, y, z = self.base_pos[0].split(1, dim=-1)
        self.draw_sphere_and_axes((x.item(), y.item(), z.item()), self.base_quat[0], 0.2, (0, 1, 1), scale=1)

    def _draw_viewer_overlays(self):
        if self.headless or self.viewer is None or not self.cfg.asset.render_sphere:
            return
        self.gym.clear_lines(self.viewer)
        self._arm_draw_overlay_hook()
        self._draw_base_ori_coord()

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, : self.num_actions] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
        actions_scaled = torch.nn.functional.pad(actions_scaled, (0, self.num_dof - self.num_actions), "constant", 0.0)

        self.joint_pos_target = actions_scaled + self.default_dof_pos
        control_type = self.cfg.control.control_type

        if control_type == "M":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos + self.motor_offsets)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )

            torques = torques * self.motor_strengths
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

            # Only the arm slice of pos_target is used (legs use torques above).
            # motor_strengths models actuator TORQUE scaling; multiplying it into
            # a POSITION target distorts the commanded joint angle by ~+-15%
            # (~0.2 rad on a ~1.5 rad joint), which the DLS-IK loop cannot
            # compensate and was the dominant EE-tracking error floor -- so it is
            # deliberately NOT applied to the arm position command. motor_offsets
            # (a small +-0.025 rad zero-point error) is kept as a mild, realistic
            # position disturbance. Arm actuator-strength domain randomization,
            # if wanted, belongs on the DOF drive stiffness, not the target.
            pos_target = self.joint_pos_target + self.motor_offsets
            pos_target = torch.clip(pos_target, -10, 10)  # max rads
            return torch.concat(
                (torques[..., : self.num_actions_loco], pos_target[..., self.num_actions_loco :]), dim=-1
            )

        elif control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos + self.motor_offsets)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )

            torques = torques * self.motor_strengths
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            return torques
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env). 18D, 12 for loco and 6 for arm.
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self._arm_pre_step_hook()
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        if not self.headless:
            self.render_gui()
        randomize_action_delay = getattr(self.cfg.domain_rand, "randomize_action_delay", False)
        self.step_locomotion_power.zero_()
        if randomize_action_delay:
            actions_start_decimation = torch.randint(
                0,
                self.cfg.control.decimation + 1,
                (self.num_envs, 1),
                device=self.device,
            )
            # R5: actuation delay is a domain parameter like any other, and it
            # is redrawn every step -- so unlike the rest it cannot be handled
            # by _nominalize_twins and has to be zeroed here.
            if self._grouping_active():
                actions_start_decimation[self.is_nominal_twin] = 0
        for i in range(self.cfg.control.decimation):
            self._arm_decimation_hook()
            if randomize_action_delay:
                input_actions = torch.where(i >= actions_start_decimation, self.actions, self.last_actions)
            else:
                input_actions = self.actions
            self.torques = self._compute_torques(input_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self._arm_post_sim_hook()

            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.step_locomotion_power += torch.sum(
                torch.abs(
                    self.torques[:, : self.num_actions_loco]
                    * self.dof_vel[:, : self.num_actions_loco]
                ),
                dim=-1,
            ) / float(self.cfg.control.decimation)
        self.post_physics_step()

        return self.rew_buf_dog, self.rew_buf_arm, self.reset_buf, self.extras

    # ---- arm / WBC hooks: no-ops on LeggedRobot, overridden in WBCEnv ----

    def _arm_pre_step_hook(self):
        """Called once per step() before decimation; for arm/stage1 curriculum."""
        pass

    def _arm_decimation_hook(self):
        """Called at the start of every decimation sub-step; for external EE force."""
        pass

    def _arm_post_sim_hook(self):
        """Called inside the decimation loop, after torques applied; for keep_arm_fixed."""
        pass

    def _arm_post_physics_hook(self):
        """Called from post_physics_step() after base state refresh."""
        pass

    def _arm_check_termination_hook(self):
        """Augment self.reset_buf with arm-aware termination conditions."""
        pass

    def _arm_reset_hook(self, env_ids):
        """Reset arm state for the given env_ids."""
        pass

    def _arm_post_reset_refresh_hook(self, env_ids):
        """Bookkeep arm state after reset without writing sim tensors."""
        pass

    def _arm_post_dof_reset_hook(self, env_ids):
        """Let arm tasks adjust reset DOF positions before the single DOF state write."""
        pass

    def _arm_resample_commands_train_hook(self, env_ids):
        """Per-episode arm command resampling alongside _resample_commands."""
        return False

    def _arm_post_dof_randomization_hook(self, env_ids):
        """Let arm tasks override generic DOF randomization for selected envs."""
        pass

    def _arm_post_callback_hook(self):
        """Extra updates inside _post_physics_step_callback (force, traj progress)."""
        pass

    def _arm_init_buffers_hook(self):
        """Allocate arm/WBC buffers; called at the end of _init_buffers()."""
        pass

    def _arm_observation_hook(self, obs_buf, roll, pitch, yaw):
        """Append arm-related entries to obs_buf in compute_observations."""
        return obs_buf

    def _arm_observation_traj_hook(self, obs_buf):
        """Append trajectory-specific entries to obs_buf (called near the end)."""
        return obs_buf

    def _arm_privileged_obs_hook(self, privileged_obs_buf):
        """Append arm-related entries to privileged observations."""
        return privileged_obs_buf

    def _arm_draw_overlay_hook(self):
        """Viewer overlay drawing for arm/EE/trajectory."""
        pass

    def _arm_render_overlay_hook(self, frame, env_id, env_handle, camera_handle):
        """Frame-buffer overlays (text + trajectory projection) for recorded video."""
        pass

    def _arm_step_end_hook(self):
        """Final bookkeeping at the end of post_physics_step (plan actions, EE twist cache)."""
        pass

    def _arm_init_performance_metrics_hook(self):
        """Register arm/WBC performance accumulators after common buffers exist."""
        pass

    def _arm_update_performance_metrics_hook(self):
        """Accumulate arm/WBC metrics from physical state, independently of rewards."""
        pass

    def _arm_log_performance_metrics_hook(self, train_env_ids, episode_steps):
        """Append arm/WBC episode metrics to ``extras['train/episode']``."""
        pass

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[: self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[: self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[: self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[: self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[
            :, self.feet_indices, 7:10
        ]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self._arm_post_physics_hook()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        # Ahead of the metrics, not just ahead of compute_reward(): both have to
        # see this step's reference, otherwise the logged tracking error is off
        # by one control step and looks like a constant lag that isn't there.
        self._update_response_state()
        self._update_performance_metrics()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if getattr(self.cfg.env, "arm_policy_enabled", True):
            self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self._arm_step_end_hook()

        if not self.headless and self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = (
                torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
                < self.cfg.rewards.terminal_body_height
            )
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        rpy = quaternion_to_rpy(self.base_quat)
        self.roll, self.pitch, self.y = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        self._arm_check_termination_hook()
        self.reset_buf |= self.reverse_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        completed_episode_steps = self.episode_length_buf[env_ids].clone()
        completed_episode_timeouts = self.time_out_buf[env_ids].clone()

        # reset robot states.
        # R5: a grouped env that reset mid-window adopts its group's command
        # instead of drawing a new one -- drawing would break the shared command
        # vector the group exists to hold constant.  Ungrouped envs (the
        # evaluation block, and the remainder that does not fill a group) keep
        # the original behaviour exactly.
        if self._grouping_active():
            grouped = self.grouping.is_grouped[env_ids]
            self._adopt_group_commands(env_ids[grouped])
            self._resample_commands(env_ids[~grouped])
        else:
            self._resample_commands(env_ids)
        self._arm_reset_hook(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        self._arm_post_dof_randomization_hook(env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
        # R5: after every per-reset sampler, dog and arm alike, and before the
        # shape props are pushed to the simulator.
        self._nominalize_twins(env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)
        self._arm_post_reset_refresh_hook(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        # R2 invariant: align the reference to the measurement on reset only,
        # never on command resampling.  Deferred by one step because the
        # measurement buffers here still hold pre-reset values -- they are
        # refreshed at the top of the next post_physics_step().
        self.response_ref.request_alignment(env_ids)
        self.response_soft_gate_timer[env_ids] = 0.0
        self.prev_base_lin_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            train_mask = env_ids < self.num_train_envs
            self._log_performance_metrics(
                train_env_ids,
                completed_episode_steps[train_mask],
                completed_episode_timeouts[train_mask],
            )
            disable_dog_rewards = bool(getattr(self, "disable_dog_policy_rewards", False))
            for key in self.episode_sums.keys():
                if (
                    disable_dog_rewards
                    and key != "total"
                    and not self._reward_enabled_when_dog_policy_frozen(key)
                ):
                    self.episode_sums[key][train_env_ids] = 0.0
                    continue
                self.extras["train/episode"]["rew_" + key] = torch.mean(self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.0
            if hasattr(self, "stage1_arm_curriculum_intensity"):
                self.extras["train/episode"]["stage1_arm_curriculum_intensity"] = torch.tensor(
                    self.stage1_arm_curriculum_intensity,
                    device=self.device,
                )
            self.extras["train/episode"]["global_switch_count"] = torch.tensor(
                float(global_switch.count),
                device=self.device,
            )
            self.extras["train/episode"]["global_switch_stage1_count"] = torch.tensor(
                float(getattr(global_switch, "stage1_count", 0)),
                device=self.device,
            )
            self.extras["train/episode"]["global_switch_stage1_ramp_iters"] = torch.tensor(
                float(getattr(global_switch, "stage1_arm_ramp_iterations", 0)),
                device=self.device,
            )
            self.extras["train/episode"]["global_switch_pretrained_start"] = torch.tensor(
                float(global_switch.pretrained_to_wbc_start),
                device=self.device,
            )
            self.extras["train/episode"]["global_switch_pretrained_end"] = torch.tensor(
                float(global_switch.pretrained_to_wbc_end),
                device=self.device,
            )
            self.extras["train/episode"]["global_switch_open"] = torch.tensor(
                float(getattr(global_switch, "switch_open", False)),
                device=self.device,
            )
            for name, value in self.curriculum_thresholds.items():
                self.extras["train/episode"]["curriculum_threshold_" + name] = torch.tensor(
                    float(value),
                    device=self.device,
                )
            if getattr(self, "curricula", None):
                curriculum = self.curricula[0]
                bins = self.env_command_bins[train_env_ids.cpu().numpy()]
                bin_weights = torch.tensor(curriculum.weights[bins], device=self.device, dtype=torch.float)
                if bin_weights.numel() > 0:
                    self.extras["train/episode"]["command_curriculum_weight"] = torch.mean(bin_weights)
            if hasattr(self, "stage2_base_unlock_success_ema"):
                self.extras["train/episode"]["stage2_base_unlock_weight"] = torch.tensor(
                    float(self._stage2_base_unlock_weight()),
                    device=self.device,
                )
                self.extras["train/episode"]["stage2_base_unlock_success_ema"] = torch.tensor(
                    float(self.stage2_base_unlock_success_ema),
                    device=self.device,
                )
                self.extras["train/episode"]["stage2_base_unlock_batch_success"] = torch.tensor(
                    float(self.stage2_base_unlock_batch_success),
                    device=self.device,
                )
                self.extras["train/episode"]["stage2_base_unlock_started"] = torch.tensor(
                    float(self.stage2_base_unlock_started),
                    device=self.device,
                )
            if getattr(self, 'reset_curriculum_enabled', False):
                self.extras["train/episode"]["reset_curriculum_intensity"] = torch.tensor(
                    float(self.reset_curriculum_intensity),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_lin_tracking_score"] = torch.tensor(
                    float(self.reset_curriculum_lin_tracking_score),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_ang_tracking_score"] = torch.tensor(
                    float(self.reset_curriculum_ang_tracking_score),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_tracking_score"] = torch.tensor(
                    float(self.reset_curriculum_tracking_score),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_tracking_score_ema"] = torch.tensor(
                    float(self.reset_curriculum_tracking_score_ema),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_stable_iterations"] = torch.tensor(
                    float(self.reset_curriculum_stable_iterations),
                    device=self.device,
                )
                self.extras["train/episode"]["reset_curriculum_started"] = torch.tensor(
                    float(self.reset_curriculum_started),
                    device=self.device,
                )

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[: self.num_train_envs].float()
            )

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[: self.num_train_envs]

        for metric_sum in self.performance_metric_sums.values():
            metric_sum[env_ids] = 0.0

        self.gait_indices[env_ids] = 0

    def compute_observations(self):

        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        obs_buf = torch.cat(
            (
                (self.dof_pos[:, : self.num_actions] - self.default_dof_pos[:, : self.num_actions])
                * self.obs_scales.dof_pos,
                self.dof_vel[:, : self.num_actions_loco] * self.obs_scales.dof_vel,
                self.actions[:, : self.num_actions],
            ),
            dim=-1,
        )

        obs_buf = self._arm_observation_hook(obs_buf, roll, pitch, yaw)

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                obs_buf = torch.cat(
                    (
                        self.root_states[: self.num_envs, 7:10] * self.obs_scales.lin_vel,
                        self.base_ang_vel * self.obs_scales.ang_vel,
                        obs_buf,
                    ),
                    dim=-1,
                )
            else:
                obs_buf = torch.cat(
                    (self.base_lin_vel * self.obs_scales.lin_vel, self.base_ang_vel * self.obs_scales.ang_vel, obs_buf),
                    dim=-1,
                )

        if self.cfg.env.observe_only_ang_vel:
            obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            obs_buf = torch.cat((obs_buf, heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            obs_buf = torch.cat(
                (obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(self.num_envs, -1) * 1.0), dim=1
            )

        obs_buf = self._arm_observation_traj_hook(obs_buf)

        self.obs_buf = obs_buf

        assert obs_buf.shape[1] == self.cfg.env.num_observations, (
            f"num_observations ({self.cfg.env.num_observations})\
                       != the number of observations ({obs_buf.shape[1]})"
        )

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale,
                ),
                dim=1,
            )

        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range
            )
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    (self.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift)
                    * ground_friction_coeffs_scale,
                ),
                dim=1,
            )

        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1
            )

        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range
            )
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.com_displacements - com_displacements_shift) * com_displacements_scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1
            )

        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1
            )

        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    ((self.root_states[: self.num_envs, 2]).view(self.num_envs, -1) - body_height_shift)
                    * body_height_scale,
                ),
                dim=1,
            )

        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.gravities - gravity_shift) / gravity_scale), dim=1
            )

        if self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                privileged_obs_buf = torch.cat(
                    (privileged_obs_buf, self.root_states[: self.num_envs, 7:10] * self.obs_scales.lin_vel), dim=-1
                )
            else:
                privileged_obs_buf = torch.cat(
                    (privileged_obs_buf, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1
                )

        privileged_obs_buf = self._arm_privileged_obs_hook(privileged_obs_buf)

        self.privileged_obs_buf = privileged_obs_buf

        assert privileged_obs_buf.shape[1] == self.cfg.env.num_privileged_obs, (
            f"num_privileged_obs ({self.cfg.env.num_privileged_obs})\
                       != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                       you will discard data from the student!"
        )

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)
        if privileged_obs_buf is not None:
            privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def _reward_enabled_when_dog_policy_frozen(self, name):
        if name.startswith("arm_"):
            return True
        return name in {
            "ee_pos_tracking",
            "ee_rot_tracking",
            "ee_smoothness",
            "goal_pos_l2",
            "reachability_barrier",
            "manipulability",
            "joint_limit_barrier",
            "rho_rate",
            "upper_action_rate",
            "delta_vel_magnitude",
            "posture_command_rate",
            "stay_still_in_reach_sector",
            "traj_progress",
            "traj_lateral_err",
            "traj_timing",
            "traj_twist_err",
        }

    def _terrain_reference_height(self):
        """Ground height under the base, or 0 on flat terrain.

        Global invariant 9: on rough terrain every height quantity must be
        measured relative to the local ground, never in the world frame.  With
        ``terrain.measure_heights`` off (the current flat-ground setup)
        ``measured_heights`` is the scalar 0 and this degenerates correctly.
        """
        heights = self.measured_heights
        if isinstance(heights, torch.Tensor):
            return torch.mean(heights, dim=1) if heights.ndim > 1 else heights
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _response_measured(self):
        """The five decision channels as actually realised, in channel order.

        Body frame throughout, and height relative to the local terrain -- the
        quantities the reference model, the consistency rewards and the MPC all
        have to agree on.  ``self.pitch`` is refreshed by check_termination(),
        which runs earlier in post_physics_step().
        """
        body_height = (
            self.base_pos[:, 2]
            - self._terrain_reference_height()
            - float(self.cfg.rewards.base_height_target)
        )
        return torch.stack(
            (
                self.base_lin_vel[:, 0],
                self.base_lin_vel[:, 1],
                self.base_ang_vel[:, 2],
                body_height,
                self.pitch,
            ),
            dim=-1,
        )

    def _update_response_soft_gate(self):
        """R4.1's soft target: drop consistency while recovering from a hit.

        Response consistency is a soft objective -- under a large disturbance the
        policy must be free to prioritise staying upright. Two triggers, both
        read from quantities the env already computes:

        * a bad slip, measured as the mean horizontal speed of the feet that are
          actually in contact;
        * a large body acceleration, which is what an external push looks like
          from the base.
        """
        reward_cfg = self.cfg.response.reward
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        slip_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        mean_slip = (slip_speed * contact).sum(dim=-1) / torch.clamp(contact.sum(dim=-1), min=1.0)

        acceleration = torch.norm(
            (self.base_lin_vel[:, :2] - self.prev_base_lin_vel[:, :2]) / self.dt, dim=-1
        )
        triggered = (mean_slip > float(reward_cfg.soft_gate_slip_speed)) | (
            acceleration > float(reward_cfg.soft_gate_accel)
        )
        self.response_soft_gate_timer = soft_gate_from_events(
            self.response_soft_gate_timer, triggered, self.soft_gate_hold_steps
        )
        self.response_soft_gate = (self.response_soft_gate_timer == 0).float()
        self.prev_base_lin_vel[:] = self.base_lin_vel

    def _residual_sample_active(self):
        """Environments whose current sample may enter the residual estimate.

        Two exclusions, both of which would otherwise corrupt it:

        * **Standing.** ``_resample_commands`` forces the gait frequency to 0
          when the velocity command is under 0.1, so the phase clock stops
          (~17-24% of samples in practice). Every sample would then pile into
          whichever phase bin the env happened to freeze in.
        * **Just reset.** The low-pass still holds the previous episode's level
          and the robot is settling, so the residual is meaningless.
        """
        past_warmup = self.episode_length_buf > int(self.cfg.response.residual.warmup_steps)
        if self.commands_dog.shape[1] > GAIT_FREQUENCY:
            clock_running = self.commands_dog[:, GAIT_FREQUENCY] > 0.1
        else:
            # Without dynamic gait the clock runs at a fixed 3 Hz, always.
            clock_running = torch.ones_like(past_warmup)
        return past_warmup & clock_running

    def _update_response_state(self):
        """Advance the R2 reference model by one control step.

        Ordering: after check_termination() (which refreshes self.pitch) and
        before BOTH _update_performance_metrics() and compute_reward() -- R2
        invariant: they must see this step's reference, not the previous one's.
        Also caches self.response_measured for the metrics to reuse.
        """
        self.response_measured = self._response_measured()

        # Bin by COMMANDED planar speed, not measured: keeps the estimator out
        # of a feedback loop with the quantity it detrends, and it is the value
        # the planner knows.
        speed_bin = self.response_residual.speed_bin(torch.norm(self.commands_dog[:, :2], dim=-1))
        phase_bin = self.response_residual.phase_bin(self.gait_indices)
        self.response_speed_bin = speed_bin
        self.response_phase_bin = phase_bin

        # USE path first, and strictly read-only: the detrended measurement must
        # come from the estimate as it stood BEFORE this step's sample was
        # folded in, or the two R3 paths are no longer separate.
        self.response_detrended = self.response_residual.detrend(
            self.response_measured, speed_bin, phase_bin
        )
        self.response_residual_converged = self.response_residual.is_converged(speed_bin, phase_bin)

        commands = self.response_ref.gather_commands(self.commands_dog)
        # Alignment after a reset snaps to the raw measurement: the estimate has
        # nothing useful for a freshly reset env, and it is a one-off snap.
        self.response_ref.step(commands, measured=self.response_measured)

        # UPDATE path last. It returns the residual it folded in, so R4.2's
        # oscillation and the estimate it is compared against are guaranteed to
        # be built from the same low-pass state.
        self.response_oscillation = self.response_residual.update(
            self.response_measured, speed_bin, phase_bin, active=self._residual_sample_active()
        )
        self.response_delta_hat = self.response_measured - self.response_detrended

        self.steps_since_command_change += 1.0
        self._update_response_soft_gate()
        converged = self.response_residual_converged.unsqueeze(-1).float()
        self.response_phase_variance_mask = (
            settled_mask(self.steps_since_command_change, self.phase_variance_settle_steps)
            * converged
        )
        self.response_steady_gain_mask = settled_mask(
            self.steps_since_command_change, self.steady_gain_settle_steps
        )

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        reward_scales = global_switch.get_reward_scales()

        disable_dog_rewards = bool(getattr(self, "disable_dog_policy_rewards", False))

        self.rew_buf_dog[:] = 0.0
        self.rew_buf_pos_dog[:] = 0.0
        self.rew_buf_neg_dog[:] = 0.0
        self.rew_buf_arm[:] = 0.0
        self.rew_buf_pos_arm[:] = 0.0
        self.rew_buf_neg_arm[:] = 0.0
        for i in range(len(self.reward_names)):
            name = self.reward_names[i]
            if disable_dog_rewards and not self._reward_enabled_when_dog_policy_frozen(name):
                continue

            rew = self.reward_functions[i]() * reward_scales[name]

            if name in ["vis_manip_commands_tracking_lpy", "vis_manip_commands_tracking_rpy"]:
                self.episode_sums[name] += rew
                continue

            if not disable_dog_rewards:
                self.rew_buf_dog += rew
                if torch.sum(rew) >= 0:
                    self.rew_buf_pos_dog += rew
                elif torch.sum(rew) <= 0:
                    self.rew_buf_neg_dog += rew
            self.episode_sums[name] += rew

            # arm ignores pure velocity-tracking rewards; frozen-dog mode filters dog-only terms above.
            if not name in ["tracking_lin_vel", "tracking_ang_vel"]:
                self.rew_buf_arm += rew
                if torch.sum(rew) >= 0:
                    self.rew_buf_pos_arm += rew
                elif torch.sum(rew) <= 0:
                    self.rew_buf_neg_arm += rew

            if name in ["tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"]:
                self.command_sums[name] += reward_scales[name] + rew
            else:
                self.command_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf_dog[:] = torch.clip(self.rew_buf_dog[:], min=0.0)
            self.rew_buf_arm[:] = torch.clip(self.rew_buf_arm[:], min=0.0)
        elif self.cfg.rewards.only_positive_rewards_ji22_style:  # TODO: update
            self.rew_buf_dog[:] = self.rew_buf_pos_dog[:] * torch.exp(
                self.rew_buf_neg_dog[:] / self.cfg.rewards.sigma_rew_neg
            )
            self.rew_buf_arm[:] = self.rew_buf_pos_arm[:] * torch.exp(
                self.rew_buf_neg_arm[:] / self.cfg.rewards.sigma_rew_neg
            )

        # add termination reward after clipping
        if "termination" in reward_scales:
            rew = self.reward_container._reward_termination() * reward_scales["termination"]
            if not disable_dog_rewards:
                self.rew_buf_dog += rew
                self.rew_buf_arm += rew
                self.episode_sums["termination"] += rew
                self.command_sums["termination"] += rew

        self.episode_sums["total"] += self.rew_buf_dog + self.rew_buf_arm

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands_dog[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands_dog[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            if self.eval_cfg is not None:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    # ---- R5: environment grouping and the nominal twin --------------------

    def _ensure_grouping(self):
        """Build the R5 grouping.  Called from _create_envs, before any actor.

        It has to exist that early because three domain-randomisation draws --
        the arm mount-TF bucket, arm link mass/COM, and base mass -- happen once
        during actor creation and are never redrawn.  A twin chosen afterwards
        would already be non-nominal in exactly the properties that are hardest
        to notice.
        """
        if getattr(self, "grouping", None) is not None:
            return self.grouping
        cfg = self.cfg.response.grouping
        enabled = bool(cfg.enabled)
        self.grouping = EnvGrouping(
            num_envs=self.num_envs,
            group_size=int(cfg.group_size),
            # pool 0 disables grouping outright: is_grouped is all False and
            # every env keeps the original per-env resample clock, so turning
            # this off restores the pre-R5 behaviour exactly rather than
            # approximately.
            pool_envs=self.num_train_envs if enabled else 0,
            device=self.device,
        )
        self.is_nominal_twin = self.grouping.is_twin
        return self.grouping

    def _grouping_active(self):
        return getattr(self, "grouping", None) is not None and self.grouping.num_groups > 0

    def _is_nominal_twin_env(self, env_id):
        """Scalar form, for the per-env callbacks IsaacGym drives at creation."""
        grouping = self._ensure_grouping()
        return grouping.num_groups > 0 and bool(grouping.is_twin[int(env_id)])

    def _arm_nominalize_twins_hook(self, twins):
        """Reset arm-side domain randomisation for the twin rows."""
        pass

    def _nominalize_twins(self, env_ids=None):
        """Hold the nominal twins at nominal domain values.

        Written as a restore *after* each sampler rather than a mask threaded
        *through* every sampler.  There are ten randomisation sites across two
        files and three of them run only during _create_envs; a mask has to be
        added at each one and silently does nothing if a new site is added
        later.  A restore cannot miss a site the same way -- anything it does
        not cover shows up directly as the twin's parameter differing from
        nominal, which is what ``--check r5`` asserts.

        The nominal values are exactly the initialisation defaults in
        _init_custom_buffers__; this method and that one have to agree.
        """
        if not self._grouping_active():
            return
        twins = self.is_nominal_twin
        if env_ids is not None:
            selected = torch.zeros_like(twins)
            selected[env_ids] = True
            twins = twins & selected
        if not torch.any(twins):
            return
        self.friction_coeffs[twins] = self.default_friction
        self.restitutions[twins] = self.default_restitution
        self.payloads[twins] = 0.0
        self.com_displacements[twins] = 0.0
        self.motor_strengths[twins] = 1.0
        self.motor_offsets[twins] = 0.0
        self.Kp_factors[twins] = 1.0
        self.Kd_factors[twins] = 1.0
        self.dof_frictions[twins] = self.default_dof_frictions
        self.dof_dampings[twins] = self.default_dof_dampings
        self._arm_nominalize_twins_hook(twins)

    def _randomize_gravity(self, external_force=None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = (
                torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False) * (max_gravity - min_gravity)
                + min_gravity
            )

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            if self.cfg.control.control_type == "M":
                # Position-drive gains for every arm/gripper DOF, looked up by
                # the joint's actual name (URDF-specific -- see stiffness_arm in
                # wbc.py). Covers all DOFs past the 12 leg DOFs.
                for joint_idx in range(self.num_actions_loco, self.num_dof):
                    joint_name = self.dof_names[joint_idx]
                    if joint_name in self.cfg.arm.control.stiffness_arm:
                        props[joint_idx]["stiffness"] = self.cfg.arm.control.stiffness_arm[joint_name]
                        props[joint_idx]["damping"] = self.cfg.arm.control.damping_arm[joint_name]

            if env_id == 0:
                dof_frictions = props["friction"] if "friction" in props.dtype.names else np.zeros(len(props))
                self.default_dof_frictions = to_torch(dof_frictions, device=self.device, dtype=torch.float)
                self.default_dof_dampings = to_torch(props["damping"], device=self.device, dtype=torch.float)
                self.dof_frictions[:] = self.default_dof_frictions
                self.dof_dampings[:] = self.default_dof_dampings

            print(props)

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = (
                torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
                * (max_payload - min_payload)
                + min_payload
            )
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = (
                torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)
                * (max_com_displacement - min_com_displacement)
                + min_com_displacement
            )

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = (
                torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device, requires_grad=False)
                * (max_friction - min_friction)
                + min_friction
            )

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = (
                torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device, requires_grad=False)
                * (max_restitution - min_restitution)
                + min_restitution
            )

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = (
                torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
                * (max_strength - min_strength)
                + min_strength
            )
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = (
                torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                * (max_offset - min_offset)
                + min_offset
            )
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = (
                torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
                * (max_Kp_factor - min_Kp_factor)
                + min_Kp_factor
            )
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = (
                torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
                * (max_Kd_factor - min_Kd_factor)
                + min_Kd_factor
            )

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        if env_id == 0:
            assert len(props) == len(self.body_names), "props length is not equal to body_names length"
            for name, item in zip(self.body_names, props):
                print(f"{name}: {item.mass}")

        props[0].mass = self.default_body_mass + self.payloads[env_id]

        props[0].com = gymapi.Vec3(
            self.com_displacements[env_id, 0], self.com_displacements[env_id, 1], self.com_displacements[env_id, 2]
        )
        props[self.ee_idx].mass += 100.0 / 1000  # camera

        return props

    def _init_performance_metrics(self):
        """Allocate raw physical-performance accumulators, separate from reward bookkeeping."""
        metric_names = (
            "vx_abs_error",
            "vy_abs_error",
            "yaw_rate_abs_error",
            "lin_vel_sq_error",
            "yaw_rate_sq_error",
            "roll_sq",
            "pitch_sq",
            "vertical_velocity_sq",
            "horizontal_angular_velocity_sq",
            "base_height_sq_error",
            "foot_slip_speed_sum",
            "foot_contact_samples",
            "locomotion_power_sum",
            *(f"ref_abs_err_{name}" for name in self.response_ref.channel_names),
            *(f"ref_abs_err_detrended_{name}" for name in self.response_ref.channel_names),
            "phase_variance_raw",
            "steady_gain_raw",
            "excitation_jumps",
        )
        self.performance_metric_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in metric_names
        }
        self.step_locomotion_power = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._arm_init_performance_metrics_hook()

    def _update_performance_metrics(self):
        """Accumulate one simulator-step sample in physical units, without reward functions or scales."""
        lin_vel_error = self.base_lin_vel[:, :2] - self.commands_dog[:, :2]
        yaw_rate_error = self.base_ang_vel[:, 2] - self.commands_dog[:, 2]
        sums = self.performance_metric_sums
        sums["vx_abs_error"] += torch.abs(lin_vel_error[:, 0])
        sums["vy_abs_error"] += torch.abs(lin_vel_error[:, 1])
        sums["yaw_rate_abs_error"] += torch.abs(yaw_rate_error)
        sums["lin_vel_sq_error"] += torch.sum(torch.square(lin_vel_error), dim=-1)
        sums["yaw_rate_sq_error"] += torch.square(yaw_rate_error)
        sums["roll_sq"] += torch.square(self.roll)
        sums["pitch_sq"] += torch.square(self.pitch)
        sums["vertical_velocity_sq"] += torch.square(self.base_lin_vel[:, 2])
        sums["horizontal_angular_velocity_sq"] += torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

        if isinstance(self.measured_heights, torch.Tensor):
            if self.measured_heights.ndim > 1:
                reference_height = torch.mean(self.measured_heights, dim=1)
            else:
                reference_height = self.measured_heights
        else:
            reference_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        height_command = self.commands_dog[:, 5] if self.commands_dog.shape[1] > 5 else 0.0
        body_height = self.base_pos[:, 2] - reference_height
        height_target = float(self.cfg.rewards.base_height_target) + height_command
        sums["base_height_sq_error"] += torch.square(body_height - height_target)

        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_slip_speed = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        sums["foot_slip_speed_sum"] += torch.sum(foot_slip_speed * foot_contact.float(), dim=-1)
        sums["foot_contact_samples"] += torch.sum(foot_contact, dim=-1).float()
        sums["locomotion_power_sum"] += self.step_locomotion_power

        # R2 diagnostic, deliberately NOT a reward yet: how far the realised
        # response is from the prescribed reference, per channel.  Recorded from
        # the moment the reference model exists so the step that turns it into a
        # reward has a before/after baseline.
        ref_abs_err = torch.abs(self.response_measured - self.response_ref.xi)
        # Same error on the R3-detrended measurement. R4's acceptance asks for
        # the variance of the tracking reward with and without detrending; these
        # two series are that comparison, recorded before either becomes a reward.
        ref_abs_err_detrended = torch.abs(self.response_detrended - self.response_ref.xi)
        for index, name in enumerate(self.response_ref.channel_names):
            sums[f"ref_abs_err_{name}"] += ref_abs_err[:, index]
            sums[f"ref_abs_err_detrended_{name}"] += ref_abs_err_detrended[:, index]

        # R4.2 / R4.3 raw magnitudes, logged whatever their reward scale is --
        # including 0. Their scale cannot be chosen without these numbers, and
        # they are 95x larger for an untrained policy than a trained one, so
        # watching them is how the R8 ramp gets timed.
        sums["phase_variance_raw"] += phase_variance(
            self.response_oscillation,
            self.response_delta_hat,
            self.response_channel_weights,
            mask=self.response_phase_variance_mask,
        )
        sums["steady_gain_raw"] += steady_gain(
            self.response_detrended,
            self.response_ref.gather_commands(self.commands_dog),
            self.response_channel_weights,
            mask=self.response_steady_gain_mask,
        )

        self._arm_update_performance_metrics_hook()

    @staticmethod
    def _mean_valid_metric(values, valid):
        if torch.any(valid):
            return torch.mean(values[valid])
        return torch.zeros((), dtype=values.dtype, device=values.device)

    def _log_performance_metrics(self, train_env_ids, episode_steps, timeouts):
        # R6: every metric below is reported over the CONSISTENCY environments
        # only.  The identification environments are being swept on purpose, so
        # folding them in would make every tracking curve here jump the moment
        # excitation was switched on and stop being comparable with the runs
        # that came before it.  Their own metrics are emitted separately.
        finished = episode_steps > 0
        identification = self.is_identification_env[train_env_ids]
        valid = finished & ~identification
        excited = finished & identification

        steps = torch.clamp(episode_steps.float(), min=1.0)
        sums = self.performance_metric_sums
        extras = self.extras["train/episode"]

        # R6 acceptance criterion, measured rather than assumed: command jumps
        # per identification episode.  Alongside it the early-termination rate
        # for those envs on its own -- rich excitation that makes the robot fall
        # is not identification data, it is a fall, and this is the number that
        # says so.
        #
        # Both groups are written only when that group actually finished an
        # episode in this batch.  Emitting a 0 instead would be averaged in by
        # the logger as if it were a measurement: with 25% of envs in
        # identification mode, a small reset batch is all-identification often
        # enough to drag every consistency metric down by a visible margin.
        if torch.any(excited):
            extras["perf_excitation_env_count"] = excited.float().sum()
            extras["perf_excitation_jumps_per_episode"] = self._mean_valid_metric(
                sums["excitation_jumps"][train_env_ids], excited
            )
            extras["perf_excitation_early_termination_rate"] = self._mean_valid_metric(
                (~timeouts).float(), excited
            )
        if not torch.any(valid):
            return

        def episode_mean(name):
            return sums[name][train_env_ids] / steps

        def mean_valid(values):
            return self._mean_valid_metric(values, valid)

        extras["perf_episode_count"] = valid.float().sum()
        extras["perf_vx_mae_mps"] = mean_valid(episode_mean("vx_abs_error"))
        extras["perf_vy_mae_mps"] = mean_valid(episode_mean("vy_abs_error"))
        extras["perf_yaw_rate_mae_rad_s"] = mean_valid(episode_mean("yaw_rate_abs_error"))
        extras["perf_lin_vel_rmse_mps"] = mean_valid(torch.sqrt(episode_mean("lin_vel_sq_error")))
        extras["perf_yaw_rate_rmse_rad_s"] = mean_valid(torch.sqrt(episode_mean("yaw_rate_sq_error")))
        extras["perf_roll_rms_rad"] = mean_valid(torch.sqrt(episode_mean("roll_sq")))
        extras["perf_pitch_rms_rad"] = mean_valid(torch.sqrt(episode_mean("pitch_sq")))
        extras["perf_vertical_velocity_rms_mps"] = mean_valid(
            torch.sqrt(episode_mean("vertical_velocity_sq"))
        )
        extras["perf_horizontal_angular_velocity_rms_rad_s"] = mean_valid(
            torch.sqrt(episode_mean("horizontal_angular_velocity_sq"))
        )
        extras["perf_base_height_rmse_m"] = mean_valid(torch.sqrt(episode_mean("base_height_sq_error")))
        contact_samples = sums["foot_contact_samples"][train_env_ids]
        contact_valid = valid & (contact_samples > 0)
        slip_speed = sums["foot_slip_speed_sum"][train_env_ids] / torch.clamp(contact_samples, min=1.0)
        extras["perf_foot_slip_speed_mps"] = self._mean_valid_metric(slip_speed, contact_valid)
        extras["perf_locomotion_power_w"] = mean_valid(episode_mean("locomotion_power_sum"))
        for name in self.response_ref.channel_names:
            unit = DECISION_CHANNEL_UNITS.get(name, "")
            suffix = f"_{unit}" if unit else ""
            extras[f"perf_ref_mae_{name}{suffix}"] = mean_valid(episode_mean(f"ref_abs_err_{name}"))
            extras[f"perf_ref_mae_detrended_{name}{suffix}"] = mean_valid(
                episode_mean(f"ref_abs_err_detrended_{name}")
            )
        extras["perf_phase_variance_raw"] = mean_valid(episode_mean("phase_variance_raw"))
        extras["perf_steady_gain_raw"] = mean_valid(episode_mean("steady_gain_raw"))
        extras["perf_soft_gate_open_fraction"] = mean_valid(
            torch.full_like(steps, float(self.response_soft_gate.mean()))
        )
        extras["perf_early_termination_rate"] = mean_valid((~timeouts).float())
        extras["perf_episode_duration_s"] = mean_valid(steps * self.dt)

        # R5 health.  Two numbers, and the second is the one to watch.
        #
        # perf_group_desync_fraction is how much of an episode the consistency
        # term is switched off for.  It is driven by falls, so early in training
        # it is near 1 and R5 is effectively absent; it has to come down before
        # the term means anything, which makes it the gate for R8's stage 3 in
        # the same way perf_phase_variance_raw is.
        #
        # perf_twin_early_termination_rate is the twin's own fall rate.  R5
        # points out that this doubles as a training-health monitor: the twin
        # runs the easiest domain in its group, so if the twin is falling, the
        # problem is the policy, not the randomisation.
        if self._grouping_active():
            # Instantaneous population fraction, NOT an episode average.  The
            # episode-averaged version is biased by exactly what it measures: a
            # fall zeroes that env's accumulator, so the envs whose groups spend
            # the most time desynchronised are the ones contributing the
            # shortest episodes.  Measured, it read 0.03 while the policy was
            # falling almost every episode.
            scoreable = self.grouping.is_grouped & ~self.is_nominal_twin
            extras["perf_group_desync_fraction"] = (
                1.0 - self.grouping.valid[scoreable].mean()
                if bool(scoreable.any())
                else torch.zeros((), device=self.device)
            )
            # Guarded, like the R6 metrics above and for the same reason: twins
            # are one env in four, so most reset batches contain none and an
            # unguarded write puts a 0 into the average.  Measured, that made
            # the twin's fall rate read 0.0000 against a population rate of 0.6
            # -- a number that looks like great news and means nothing.
            twin = finished & self.is_nominal_twin[train_env_ids]
            if torch.any(twin):
                extras["perf_twin_episode_count"] = twin.float().sum()
                extras["perf_twin_early_termination_rate"] = self._mean_valid_metric(
                    (~timeouts).float(), twin
                )
        self._arm_log_performance_metrics_hook(train_env_ids, steps)

    def _init_reset_curriculum(self):
        self.reset_curriculum_enabled = bool(getattr(self.cfg.terrain, 'reset_curriculum', False))
        self.reset_curriculum_started = False
        self.reset_curriculum_start_iteration = -1
        self.reset_curriculum_intensity = 1.0
        self.reset_curriculum_lin_tracking_score = 0.0
        self.reset_curriculum_ang_tracking_score = 0.0
        self.reset_curriculum_tracking_score = 0.0
        self.reset_curriculum_tracking_score_ema = 0.0
        self.reset_curriculum_stable_iterations = 0
        self._reset_curriculum_accum_iteration = -1
        self._reset_curriculum_lin_score_sum = 0.0
        self._reset_curriculum_ang_score_sum = 0.0
        self._reset_curriculum_tracking_score_sum = 0.0
        self._reset_curriculum_score_count = 0
        if self.reset_curriculum_enabled:
            self.reset_curriculum_intensity = float(
                getattr(self.cfg.terrain, 'reset_curriculum_initial_fraction', 0.0)
            )

    def _update_reset_curriculum_intensity(self, iteration):
        initial_fraction = float(getattr(self.cfg.terrain, 'reset_curriculum_initial_fraction', 0.0))
        initial_fraction = float(np.clip(initial_fraction, 0.0, 1.0))
        if not self.reset_curriculum_started:
            self.reset_curriculum_intensity = initial_fraction
            return

        growth_iterations = max(1, int(getattr(self.cfg.terrain, 'reset_curriculum_growth_iterations', 2000)))
        elapsed = max(0, int(iteration) - self.reset_curriculum_start_iteration)
        progress = min(1.0, elapsed / growth_iterations)
        self.reset_curriculum_intensity = initial_fraction + (1.0 - initial_fraction) * progress

    def _finalize_reset_curriculum_iteration(self, next_iteration):
        if self._reset_curriculum_score_count == 0:
            return

        count = float(self._reset_curriculum_score_count)
        self.reset_curriculum_lin_tracking_score = self._reset_curriculum_lin_score_sum / count
        self.reset_curriculum_ang_tracking_score = self._reset_curriculum_ang_score_sum / count
        self.reset_curriculum_tracking_score = self._reset_curriculum_tracking_score_sum / count

        alpha = float(getattr(self.cfg.terrain, 'reset_curriculum_tracking_ema_alpha', 0.05))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.reset_curriculum_tracking_score_ema = (
            (1.0 - alpha) * self.reset_curriculum_tracking_score_ema
            + alpha * self.reset_curriculum_tracking_score
        )

        threshold = float(getattr(self.cfg.terrain, 'reset_curriculum_tracking_threshold', 0.7))
        if self.reset_curriculum_tracking_score_ema >= threshold:
            self.reset_curriculum_stable_iterations += 1
        else:
            self.reset_curriculum_stable_iterations = 0

        required_stability = max(
            1,
            int(getattr(self.cfg.terrain, 'reset_curriculum_stability_iterations', 1)),
        )
        if not self.reset_curriculum_started and self.reset_curriculum_stable_iterations >= required_stability:
            self.reset_curriculum_started = True
            self.reset_curriculum_start_iteration = int(next_iteration)

    def _reset_reset_curriculum_accumulator(self, iteration):
        self._reset_curriculum_accum_iteration = int(iteration)
        self._reset_curriculum_lin_score_sum = 0.0
        self._reset_curriculum_ang_score_sum = 0.0
        self._reset_curriculum_tracking_score_sum = 0.0
        self._reset_curriculum_score_count = 0

    def _update_reset_curriculum(self, tracking_task_rewards=None, tracking_reward_scales=None):
        if not getattr(self, 'reset_curriculum_enabled', False):
            return

        iteration = int(getattr(global_switch, 'count', 0))
        if self._reset_curriculum_accum_iteration < 0:
            self._reset_reset_curriculum_accumulator(iteration)
        elif iteration != self._reset_curriculum_accum_iteration:
            self._finalize_reset_curriculum_iteration(iteration)
            self._reset_reset_curriculum_accumulator(iteration)

        self._update_reset_curriculum_intensity(iteration)
        if tracking_task_rewards is None or tracking_reward_scales is None:
            return

        lin_reward = tracking_task_rewards.get("tracking_lin_vel")
        ang_reward = tracking_task_rewards.get("tracking_ang_vel")
        lin_scale = float(tracking_reward_scales.get("tracking_lin_vel", 0.0))
        ang_scale = float(tracking_reward_scales.get("tracking_ang_vel", 0.0))
        if lin_reward is None or ang_reward is None or lin_scale <= 0.0 or ang_scale <= 0.0:
            return

        lin_score = torch.clamp(lin_reward / lin_scale, 0.0, 1.0)
        ang_score = torch.clamp(ang_reward / ang_scale, 0.0, 1.0)
        tracking_score = torch.minimum(lin_score, ang_score)
        self._reset_curriculum_lin_score_sum += float(lin_score.sum().item())
        self._reset_curriculum_ang_score_sum += float(ang_score.sum().item())
        self._reset_curriculum_tracking_score_sum += float(tracking_score.sum().item())
        self._reset_curriculum_score_count += int(tracking_score.numel())

    def _get_reset_curriculum_range(self, range_name):
        max_range = float(getattr(self.cfg.terrain, range_name))
        if not getattr(self, 'reset_curriculum_enabled', False):
            return max_range
        return max_range * float(getattr(self, 'reset_curriculum_intensity', 1.0))

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        arm_controls_commands = bool(self._arm_resample_commands_train_hook(env_ids))

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        curriculum = self.curricula[0]
        # update curricula based on terminated environment bins and categories
        task_rewards, success_thresholds = [], []
        tracking_task_rewards, tracking_reward_scales = {}, {}
        for key in CURRICULUM_PROGRESS_REWARDS:
            if key in self.command_sums.keys():
                task_reward = self.command_sums[key][env_ids] / ep_len
                task_rewards.append(task_reward)
                success_thresholds.append(self.curriculum_thresholds[key] * self.pretrained_reward_scales[key])
                if key in ("tracking_lin_vel", "tracking_ang_vel"):
                    tracking_task_rewards[key] = task_reward
                    tracking_reward_scales[key] = self.pretrained_reward_scales[key]

        old_bins = self.env_command_bins[env_ids.cpu().numpy()]
        # R6 invariant: the identification environments are driven by designed
        # excitation, not by the curriculum's samples, and their tracking reward
        # is low BY CONSTRUCTION.  Feeding it to the adaptive curriculum would be
        # read as "this command bin is too hard" and would ratchet the command
        # range down for every environment, identification or not.
        scores = ~self.is_identification_env[env_ids]
        scored = scores.cpu().numpy()
        if len(success_thresholds) > 0 and bool(scored.any()):
            local_range = command_curriculum_local_range(self.cfg)
            curriculum.update(
                old_bins[scored],
                [reward[scores] for reward in task_rewards],
                success_thresholds,
                local_range=local_range,
            )
        self._update_reset_curriculum(
            {key: value[scores] for key, value in tracking_task_rewards.items()},
            tracking_reward_scales,
        )

        # sample from new category curricula
        new_commands, new_bin_inds = curriculum.sample(batch_size=len(env_ids))
        new_commands = torch.as_tensor(new_commands, dtype=self.commands_dog.dtype, device=self.device)

        # R5: one draw per GROUP, not per env.  Every random choice below has to
        # go through this same remap -- the curriculum draw, its bin, and the
        # "10% of envs stand still" coin -- or the group's command vectors differ
        # and every cross-domain comparison in the group is comparing two
        # different tasks.  The deterministic parts (the 0.07 m/s deadband, the
        # standing gait-frequency rule) follow the command and need no remap.
        take = self._group_source_rows(env_ids)
        if take is not None:
            new_commands = new_commands[take]
            new_bin_inds = new_bin_inds[take.cpu().numpy()]

        self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
        self.env_command_categories[env_ids.cpu().numpy()] = 0

        if not arm_controls_commands:
            self.commands_dog[env_ids, 0] = new_commands[:, 0]
            self.commands_dog[env_ids, 1] = new_commands[:, 1]
            self.commands_dog[env_ids, 2] = new_commands[:, 2]

        zero_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        if take is not None:
            zero_mask = zero_mask[take]
        if not arm_controls_commands and len(zero_mask.nonzero()) > 0:
            self.commands_dog[env_ids[zero_mask], :3] = 0

            self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
            self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
            self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1

        elif not arm_controls_commands:
            if not global_switch.switch_open:
                self.commands_dog[env_ids, 0] = new_commands[:, 0]
                self.commands_dog[env_ids, 1] = new_commands[:, 1]
                self.commands_dog[env_ids, 2] = new_commands[:, 2]

                # # Randomly select 10% of the environment to remain stationary
                # num_zero_envs = int(0.1 * len(env_ids))
                # zero_env_ids = torch.randperm(len(env_ids))[:num_zero_envs]
                # self.commands_dog[env_ids[zero_env_ids], :3] = 0

                zero_mask = torch.rand(len(env_ids), device=self.device) < 0.1
                if take is not None:
                    zero_mask = zero_mask[take]
                if len(zero_mask.nonzero()) > 0:
                    self.commands_dog[env_ids[zero_mask], :3] = 0

                self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
                self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
                self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1

        if not global_switch.switch_open and not arm_controls_commands:
            self.commands_dog[env_ids, 3] = new_commands[:, 3]
            self.commands_dog[env_ids, 4] = new_commands[:, 4]
            self.commands_dog[env_ids, 5] = new_commands[:, 5]  # body_height_cmd

        if self.cfg.commands.use_dynamic_gait and not global_switch.switch_open and not arm_controls_commands:
            self.commands_dog[env_ids, 6] = new_commands[:, 6]
            self.commands_dog[env_ids, 7] = new_commands[:, 7]
            self.commands_dog[env_ids, 8] = new_commands[:, 8]
            self.commands_dog[env_ids, 9] = new_commands[:, 9]
            self.commands_dog[env_ids, 10] = new_commands[:, 10]
            standing_mask = torch.norm(self.commands_dog[env_ids, :3], dim=1) < 0.1
            if len(standing_mask.nonzero()) > 0:
                self.commands_dog[env_ids[standing_mask], 6] = 0.0

        # R6: a fresh excitation plan for whichever of these are identification
        # envs.  Paired with the baseline draw above deliberately -- the four
        # passive channels must be constant for the whole identification record,
        # so a new baseline and a new plan always happen together.
        self.response_excitation.plan(env_ids)

        # R4.2 / R4.3 are masked relative to the last command step, so the
        # counter has to be zeroed wherever commands actually change.
        self.steps_since_command_change[env_ids] = 0.0

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0

    def _group_source_rows(self, env_ids):
        """Row remap that turns a per-env draw into one draw per group.

        Returns an index into ``env_ids`` such that every grouped env takes the
        row its twin drew, or ``None`` when nothing is grouped.  Envs whose twin
        is not in this batch keep their own row -- that only happens on a
        partial-group resample, which the group clock does not produce, but
        falling back to the env's own draw is the safe answer if it ever does.
        """
        if not self._grouping_active() or env_ids.numel() == 0:
            return None
        position = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        position[env_ids] = torch.arange(env_ids.numel(), device=self.device)
        source = position[self.grouping.twin_of[env_ids]]
        own = torch.arange(env_ids.numel(), device=self.device)
        return torch.where(source >= 0, source, own)

    def _sync_groups(self, group_ids):
        """Re-establish a group's shared gait phase and clear its desync flag.

        R5 invariant: "the phase clocks within a group must be synchronised --
        once the phase drifts apart the cross-domain comparison is meaningless".
        The residual model is indexed by phase bin, so two envs a half cycle
        apart are being asked to match oscillations that are, correctly,
        opposite in sign.
        """
        if group_ids.numel() == 0:
            return
        env_ids = self.grouping.envs_of_groups(group_ids)
        self.gait_indices[env_ids] = self.gait_indices[self.grouping.twin_of[env_ids]]
        self.grouping.resync(group_ids)

    def _adopt_group_commands(self, env_ids):
        """Reset path for a grouped env: take the group's command, don't draw one.

        This is the substantive change to ``_resample_commands``' semantics that
        R5 forces.  An env that fell mid-window would otherwise draw a fresh
        command and immediately break the one thing its group exists to hold
        constant.  It adopts the twin's current command instead, and the group
        is marked desynchronised so its consistency term stays off until the
        next shared resample restores the phase too.

        No curriculum interaction: the env keeps the group's bin, and scoring it
        here would credit a bin for an episode that ended in a fall partway
        through someone else's window.
        """
        if env_ids.numel() == 0:
            return
        source = self.grouping.twin_of[env_ids]
        self.commands_dog[env_ids] = self.commands_dog[source]
        self.env_command_bins[env_ids.cpu().numpy()] = self.env_command_bins[
            source.cpu().numpy()
        ]
        self.env_command_categories[env_ids.cpu().numpy()] = 0
        self.steps_since_command_change[env_ids] = 0.0
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0
        self.grouping.mark_desync(env_ids)

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ["trot"]

        # Grid definition, initial active window and expansion neighbourhood all
        # live in go1_gym/envs/base/curriculum.py so they can be exercised
        # without IsaacGym; see the R1 acceptance test.
        from go1_gym.envs.base.curriculum import build_command_curriculum

        self.curricula = [build_command_curriculum(self.cfg) for _ in self.category_names]
        self.env_command_bins = np.zeros(len(env_ids), dtype=int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=int)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        # self._teleport_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        self._arm_post_callback_hook()

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        # R5: grouped envs resample on the GROUP clock, ungrouped ones on their
        # own episode counter.  Both paths end in _resample_commands; what
        # differs is who decides when.
        #
        # R6 rides on the same mechanism.  An identification group needs its
        # four passive channels to hold still for a whole excitation plan --
        # stepping them mid-chirp is exactly the disturbance that makes the
        # record un-identifiable as SISO -- so it gets a LONGER period rather
        # than having its resample suppressed.  Suppression was the first
        # implementation and it was wrong: once R5 made a reset env adopt its
        # twin's command, an identification group whose twin was also suppressed
        # never drew a command at all and sat at zeros for the whole run.
        due = (
            (self.episode_length_buf % sample_interval == 0)
            & ~self.grouping.is_grouped
            & ~self.is_identification_env
        )
        due_groups = self.grouping.groups_due(self.group_resample_interval)
        if due_groups.numel() > 0:
            due[self.grouping.envs_of_groups(due_groups)] = True
        env_ids = due.nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # Phase sync and the desync clear come after the commands, because
        # _resample_commands is what re-establishes the shared command vector.
        # Note this runs for identification groups too, which are filtered out
        # of env_ids above: they must not have their commands stepped
        # mid-episode (R6), but they still need their phase clocks realigned
        # (R5), and the two requirements are independent.
        if due_groups.numel() > 0:
            self._sync_groups(due_groups)

        # Excitation is written after the resample and before anything reads the
        # commands, so the excited channel wins over the baseline draw.
        excitation_jumped = self.response_excitation.step(self.commands_dog)
        self.steps_since_command_change[excitation_jumped] = 0.0
        self.performance_metric_sums["excitation_jumps"] += excitation_jumped.float()

        # Check-then-advance: see EnvGrouping's clock contract.  Advancing here
        # rather than beside episode_length_buf is what makes every group due on
        # its first step instead of holding zeros for a full interval.
        self.grouping.advance()

        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        self._push_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # randomize dof properties
        env_ids = (
            (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(as_tuple=False).flatten()
        )
        self._randomize_dof_props(env_ids, self.cfg)
        self._arm_post_dof_randomization_hook(env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()

        #  without external gravity
        if (
            int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration)
            % int(self.cfg.domain_rand.gravity_rand_interval)
            == 0
        ):
            self._randomize_gravity(torch.tensor([0, 0, 0]))

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

    def _reset_dofs(self, env_ids, cfg):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0
        self._arm_post_dof_reset_hook(env_ids)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids, cfg):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        z_init_range = self._get_reset_curriculum_range("z_init_range")
        yaw_init_range = self._get_reset_curriculum_range("yaw_init_range")
        pitch_init_range = self._get_reset_curriculum_range("pitch_init_range")
        roll_init_range = self._get_reset_curriculum_range("roll_init_range")

        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(
                -cfg.terrain.x_init_range, cfg.terrain.x_init_range, (len(env_ids), 1), device=self.device
            )
            self.root_states[env_ids, 1:2] += torch_rand_float(
                -cfg.terrain.y_init_range, cfg.terrain.y_init_range, (len(env_ids), 1), device=self.device
            )
            self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
            self.root_states[env_ids, 2:3] += torch_rand_float(
                0, z_init_range, (len(env_ids), 1), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2:3] += torch_rand_float(
                0, z_init_range, (len(env_ids), 1), device=self.device
            )

        # base orientation: yaw / pitch / roll each randomized independently
        init_yaws = torch_rand_float(
            -yaw_init_range, yaw_init_range, (len(env_ids), 1), device=self.device
        )
        init_pitches = torch_rand_float(
            -pitch_init_range, pitch_init_range, (len(env_ids), 1), device=self.device
        )
        init_rolls = torch_rand_float(
            -roll_init_range, roll_init_range, (len(env_ids), 1), device=self.device
        )
        q_yaw = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        q_pitch = quat_from_angle_axis(init_pitches, torch.Tensor([0, 1, 0]).to(self.device))[:, 0, :]
        q_roll = quat_from_angle_axis(init_rolls, torch.Tensor([1, 0, 0]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat_mul(q_yaw, quat_mul(q_pitch, q_roll))
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []

    def _push_robots(self, env_ids, cfg):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if cfg.domain_rand.push_robots:
            push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]
            if len(push_env_ids) == 0:
                return

            max_vel = cfg.domain_rand.max_push_vel_xy
            max_push_ang = cfg.domain_rand.max_push_ang_vel
            n = len(push_env_ids)
            self.root_states[push_env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (n, 2), device=self.device)
            self.root_states[push_env_ids, 10:13] = torch_rand_float(
                -max_push_ang, max_push_ang, (n, 3), device=self.device
            )

            env_ids_int32 = push_env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def _teleport_robots(self, env_ids, cfg):
        """Teleports any robots that are too close to the edge to the other side"""
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset
            ]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh
            ]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat(
            (
                torch.ones(3) * noise_scales.gravity * noise_level,
                torch.ones(self.num_actions_loco) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                torch.ones(self.num_actions_loco) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                torch.zeros(self.num_actions),
            ),
            dim=0,
        )

        if self.cfg.env.observe_command:
            noise_vec = torch.cat(
                (
                    torch.ones(3) * noise_scales.gravity * noise_level,
                    torch.zeros(self.cfg.commands.num_commands),
                    torch.ones(self.num_actions_loco) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                    torch.ones(self.num_actions_loco) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                    torch.zeros(self.num_actions),
                ),
                dim=0,
            )
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec, torch.zeros(self.num_actions)), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec, torch.zeros(4)), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat(
                (
                    torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                    torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                    noise_vec,
                ),
                dim=0,
            )

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat(
                (torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel, noise_vec), dim=0
            )

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat(
                (
                    noise_vec,
                    torch.zeros(1),
                ),
                dim=0,
            )

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat(
                (
                    noise_vec,
                    torch.ones(4) * noise_scales.contact_states * noise_level,
                ),
                dim=0,
            )

        noise_vec = noise_vec.to(self.device)

        return noise_vec

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[: self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[: self.num_envs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[: self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[: self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[
            :, self.feet_indices, 7:10
        ]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[: self.num_envs * self.num_bodies, :].view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.joint_pos_target = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_joint_pos_target = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_joint_pos_target = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.desired_contact_states = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.last_contact_filt = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[: self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[: self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False

            if i >= self.num_actions:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                continue

            if i < self.num_actions_loco:
                # Legs: select a gain group by substring (leg DOF names all
                # contain the group key, e.g. "joint").
                for dof_name in self.cfg.dog.control.stiffness_leg.keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg.dog.control.stiffness_leg[dof_name]  # [N*m/rad]
                        self.d_gains[i] = self.cfg.dog.control.damping_leg[dof_name]  # [N*m*s/rad]
                        found = True
            else:
                # Arm: look up by the joint's exact name. Substring matching is
                # unsafe here because x5_joint* names contain the leg key
                # "joint". (In control_type "M" the arm slice is position-driven
                # via _process_dof_props, so these p/d gains only feed the
                # torque path for the leg slice; the arm entries stay correct
                # for any consumer that reads them.)
                if name in self.cfg.arm.control.stiffness_arm:
                    self.p_gains[i] = self.cfg.arm.control.stiffness_arm[name]  # [N*m/rad]
                    self.d_gains[i] = self.cfg.arm.control.damping_arm[name]  # [N*m*s/rad]
                    found = True

            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["M", "P"]:  # M: Mixture, P: position
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # [1,20]
        print("p gains: ", self.p_gains)
        print("d gains: ", self.d_gains)

        self.commands_dog = torch.zeros(
            self.num_envs, self.cfg.dog.dog_num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, body_pitch, body_roll, body_height, [gait...]
        self.commands_scale_dog = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.body_pitch_cmd,
                self.obs_scales.body_roll_cmd,
                self.obs_scales.body_height_cmd,
                self.obs_scales.gait_freq_cmd,
                self.obs_scales.footswing_height_cmd,
                self.obs_scales.stance_width_cmd,
                self.obs_scales.stance_length_cmd,
                self.obs_scales.gait_duration_cmd,
            ],
            device=self.device,
            requires_grad=False,
        )[: self.cfg.dog.dog_num_commands]
        # R2 prescribed reference-response model: one critically damped
        # second-order system with a rate box per MPC decision channel.  The
        # policy is trained to track its state, not the raw command, and the
        # very same linear system becomes the MPC's nominal dynamics.
        self.response_ref = ReferenceModel(
            build_channels(
                self.cfg.response.channel_order,
                self.cfg.response.omega_n,
                self.cfg.response.rate_limit,
            ),
            num_envs=self.num_envs,
            dt=self.dt,
            device=self.device,
        )
        # R3: per-environment estimate of the gait-phase-conditioned base
        # oscillation. Feeds two things that must not share a data path -- the
        # zero-delay detrend the tracking rewards need, and the target the
        # phase-variance penalty compares against.
        reward_cfg = self.cfg.response.reward
        names = self.response_ref.channel_names

        def _row(mapping):
            return torch.tensor(
                [float(mapping[name]) for name in names], device=self.device, dtype=torch.float
            ).unsqueeze(0)

        self.response_sigma = _row(reward_cfg.sigma)
        self.response_channel_weights = _row(reward_cfg.channel_weights)
        # R4.2 / R4.3 are masked for 2/omega_n and 3/omega_n after a command
        # step, per channel: omega_n differs, and pitch -- the slowest by design
        # -- therefore stays masked longest.
        omega_n = self.response_ref.omega_n
        self.phase_variance_settle_steps = (
            float(reward_cfg.phase_variance_settle_factor) / omega_n / self.dt
        )
        self.steady_gain_settle_steps = (
            float(reward_cfg.steady_gain_settle_factor) / omega_n / self.dt
        )
        self.soft_gate_hold_steps = int(round(float(reward_cfg.soft_gate_hold_s) / self.dt))

        self.steps_since_command_change = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.response_soft_gate_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.response_soft_gate = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        residual_cfg = self.cfg.response.residual
        self.response_residual = PhaseResidualEstimator(
            num_envs=self.num_envs,
            num_channels=self.response_ref.num_channels,
            dt=self.dt,
            speed_bin_edges=residual_cfg.speed_bin_edges,
            num_phase_bins=int(residual_cfg.num_phase_bins),
            lowpass_tau_s=float(residual_cfg.lowpass_tau_s),
            estimate_cycles=float(residual_cfg.estimate_cycles),
            min_cycles=float(residual_cfg.min_cycles),
            device=self.device,
        )

        # R6: designed excitation for the identification subset of environments.
        # Built from the curriculum's *initial* active window rather than the
        # hard command limits -- identification data is only informative where
        # the policy is competent, and +-1.5 m/s from iteration 0 mostly
        # produces falls.
        excitation_cfg = self.cfg.response.excitation
        curriculum_low, curriculum_high = command_curriculum_bounds(self.cfg)
        self.response_excitation = ExcitationSampler(
            self.response_ref.channels,
            num_envs=self.num_envs,
            dt=self.dt,
            low=[float(curriculum_low[c.cmd_index]) for c in self.response_ref.channels],
            high=[float(curriculum_high[c.cmd_index]) for c in self.response_ref.channels],
            # Eval envs are the tail of the full range; the identification block
            # is the tail of the TRAINING range so the two never overlap.
            pool_envs=self.num_train_envs,
            # R5 x R6: the identification block must be a whole number of
            # groups, and every env in a group must see the same excitation --
            # otherwise "one command vector per group" is false for exactly the
            # groups whose commands are most interesting.
            block_multiple=self.grouping.group_size if self._grouping_active() else 1,
            share_with=self.grouping.twin_of if self._grouping_active() else None,
            env_fraction=(
                float(excitation_cfg.env_fraction) if bool(excitation_cfg.enabled) else 0.0
            ),
            signal_weights=dict(excitation_cfg.signal_weights),
            channel_weights=dict(excitation_cfg.channel_weights),
            prbs_hold_s=tuple(excitation_cfg.prbs_hold_s),
            chirp_hz=tuple(excitation_cfg.chirp_hz),
            chirp_duration_s=float(excitation_cfg.chirp_duration_s),
            chirp_slew_fraction=float(excitation_cfg.chirp_slew_fraction),
            ramp_slope_multiple=tuple(excitation_cfg.ramp_slope_multiple),
            device=self.device,
        )
        self.is_identification_env = self.response_excitation.is_identification
        # Per-group resample period.  The identification groups get the
        # excitation plan's own period so a plan is never cut in half by a
        # command step; everyone else gets commands.resampling_time.
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        identification_interval = max(
            sample_interval,
            int(round(float(excitation_cfg.chirp_duration_s) / self.dt)),
        )
        self.group_resample_interval = torch.full(
            (self.grouping.num_groups,), sample_interval, dtype=torch.long, device=self.device
        )
        if self._grouping_active() and self.response_excitation.active:
            twins = self.grouping.is_twin.nonzero(as_tuple=False).flatten()
            identification_groups = self.grouping.group_of[
                twins[self.is_identification_env[twins]]
            ]
            self.group_resample_interval[identification_groups] = identification_interval
        self.rew_buf_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # arm-side reward buffers are kept here because compute_reward() (in this base class)
        # writes to them; their semantic content is shaped entirely by WBCEnv.
        self.rew_buf_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self._arm_init_buffers_hook()
        self._init_performance_metrics()

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitutions = self.default_restitution * torch.ones(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.motor_strengths = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.motor_offsets = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.Kp_factors = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.Kd_factors = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.dof_frictions = self.default_dof_frictions.unsqueeze(0).repeat(self.num_envs, 1)
        self.dof_dampings = self.default_dof_dampings.unsqueeze(0).repeat(self.num_envs, 1)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )

        # if custom initialization values were passed in, set them here
        dynamics_params = [
            "friction_coeffs",
            "restitutions",
            "payloads",
            "com_displacements",
            "motor_strengths",
            "Kp_factors",
            "Kd_factors",
            "dof_frictions",
            "dof_dampings",
        ]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.halftime_clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # R6 invariant 2, enforced rather than commented: the curriculum's
        # progress signal must stay the original tracking terms.  A consistency
        # reward that took one of those names would silently become a curriculum
        # criterion, and the curriculum would stop advancing.
        overlap = set(CURRICULUM_PROGRESS_REWARDS) & set(CONSISTENCY_REWARDS)
        if overlap:
            raise AssertionError(
                f"consistency rewards {sorted(overlap)} collide with the command "
                "curriculum's progress keys (R6 invariant 2)"
            )

        # reward containers
        from go1_gym.envs.rewards.rewards import Rewards

        reward_containers = {"Rewards": Rewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.pretrained_reward_scales.keys()):
            scale = self.pretrained_reward_scales[key]
            if scale == 0:
                self.pretrained_reward_scales.pop(key)
            else:
                self.pretrained_reward_scales[key] *= self.dt

        for key in list(self.wbc_reward_scales.keys()):
            self.wbc_reward_scales[key] *= self.dt

        # Complete the WBC reward table with unchanged stage-1 scales. A name
        # explicitly zeroed under wbc.reward_scales.* (e.g. to disable it for
        # WBC while stage-1 still uses it, like raibert_heuristic) must still
        # count as "present" here so this inheritance doesn't revive the
        # nonzero stage-1 value -- so zero-scale WBC entries are only dropped
        # below, after inheritance is resolved.
        for name, scale in self.pretrained_reward_scales.items():
            if name not in self.wbc_reward_scales:
                self.wbc_reward_scales[name] = scale

        # remove WBC-side zero scales (dt-scaling above turns them into
        # exactly 0 too, so this also catches those)
        for key in list(self.wbc_reward_scales.keys()):
            if self.wbc_reward_scales[key] == 0:
                self.wbc_reward_scales.pop(key)

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.wbc_reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, "_reward_" + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, "_reward_" + name))
                if name not in self.pretrained_reward_scales:
                    self.pretrained_reward_scales[name] = 0.0

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.wbc_reward_scales.keys()
        }
        self.episode_sums["total"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.wbc_reward_scales.keys()
        }
        self.episode_sums_eval["total"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in list(self.wbc_reward_scales.keys())
            + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual", "ep_timesteps"]
        }

        global_switch.set_reward_scales(self.wbc_reward_scales, self.pretrained_reward_scales)

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    @staticmethod
    def _format_vector(values):
        return " ".join(f"{float(value):.8g}" for value in values)

    @staticmethod
    def _write_xml_if_changed(tree, path):
        xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
        if os.path.exists(path):
            with open(path, "rb") as f:
                if f.read() == xml_bytes:
                    return False
        with open(path, "wb") as f:
            f.write(xml_bytes)
        return True

    def _generate_arm_mount_asset_files(self, asset_root, asset_file):
        arm_dr = self.cfg.domain_rand
        source_path = os.path.join(asset_root, asset_file)
        mount_joint_name = (
            getattr(arm_dr, "mount_joint_name", "zarx5p2_mount") if arm_dr is not None else "zarx5p2_mount"
        )

        def read_source_mount_tf():
            if not asset_file.lower().endswith(".urdf"):
                return np.zeros((1, 6), dtype=np.float32)
            source_tree_local = ET.parse(source_path)
            source_joint_local = source_tree_local.getroot().find(f"./joint[@name='{mount_joint_name}']")
            if source_joint_local is None:
                raise ValueError(f"Cannot find fixed mount joint '{mount_joint_name}' in {source_path}")
            source_origin_local = source_joint_local.find("origin")
            if source_origin_local is None:
                source_origin_local = ET.SubElement(source_joint_local, "origin")
            xyz = np.fromstring(source_origin_local.get("xyz", "0 0 0"), sep=" ", dtype=np.float32)
            rpy = np.fromstring(source_origin_local.get("rpy", "0 0 0"), sep=" ", dtype=np.float32)
            if xyz.shape != (3,):
                raise ValueError(f"Invalid xyz on joint '{mount_joint_name}': {source_origin_local.get('xyz')}")
            if rpy.shape != (3,):
                raise ValueError(f"Invalid rpy on joint '{mount_joint_name}': {source_origin_local.get('rpy')}")
            return np.concatenate((xyz, rpy)).reshape(1, 6).astype(np.float32)

        randomize_position = arm_dr is not None and getattr(arm_dr, "randomize_mount_position", False)
        randomize_rotation = arm_dr is not None and getattr(arm_dr, "randomize_mount_rotation", False)
        if not randomize_position and not randomize_rotation:
            self.arm_mount_bucket_position_offsets = np.zeros((1, 3), dtype=np.float32)
            self.arm_mount_bucket_rpy_offsets = np.zeros((1, 3), dtype=np.float32)
            self.arm_mount_bucket_tfs = read_source_mount_tf()
            return [asset_file]

        num_buckets = int(getattr(arm_dr, "mount_tf_buckets", 32))
        if num_buckets <= 1 or not asset_file.lower().endswith(".urdf"):
            self.arm_mount_bucket_position_offsets = np.zeros((1, 3), dtype=np.float32)
            self.arm_mount_bucket_rpy_offsets = np.zeros((1, 3), dtype=np.float32)
            self.arm_mount_bucket_tfs = read_source_mount_tf()
            return [asset_file]

        source_tree = ET.parse(source_path)
        source_joint = source_tree.getroot().find(f"./joint[@name='{mount_joint_name}']")
        if source_joint is None:
            raise ValueError(f"Cannot find fixed mount joint '{mount_joint_name}' in {source_path}")
        source_origin = source_joint.find("origin")
        if source_origin is None:
            source_origin = ET.SubElement(source_joint, "origin")

        base_xyz = np.fromstring(source_origin.get("xyz", "0 0 0"), sep=" ", dtype=np.float32)
        if base_xyz.shape != (3,):
            raise ValueError(f"Invalid xyz on joint '{mount_joint_name}': {source_origin.get('xyz')}")
        base_rpy = np.fromstring(source_origin.get("rpy", "0 0 0"), sep=" ", dtype=np.float32)
        if base_rpy.shape != (3,):
            raise ValueError(f"Invalid rpy on joint '{mount_joint_name}': {source_origin.get('rpy')}")

        rng = np.random.default_rng(int(getattr(arm_dr, "mount_tf_bucket_seed", 1234)))
        position_offsets = np.zeros((num_buckets, 3), dtype=np.float32)
        if randomize_position:
            position_ranges = np.asarray(arm_dr.mount_position_range, dtype=np.float32)
            if position_ranges.shape != (3, 2):
                raise ValueError(
                    f"mount_position_range must have shape (3, 2), got {position_ranges.shape}"
                )
            position_offsets = rng.uniform(
                position_ranges[:, 0], position_ranges[:, 1], size=(num_buckets, 3)
            ).astype(np.float32)

        rpy_offsets = np.zeros((num_buckets, 3), dtype=np.float32)
        if randomize_rotation:
            rpy_ranges = np.asarray(arm_dr.mount_rpy_range, dtype=np.float32)
            if rpy_ranges.shape != (3, 2):
                raise ValueError(f"mount_rpy_range must have shape (3, 2), got {rpy_ranges.shape}")
            rpy_offsets = rng.uniform(rpy_ranges[:, 0], rpy_ranges[:, 1], size=(num_buckets, 3)).astype(
                np.float32
            )

        # Always retain one nominal asset for a deterministic reference bucket.
        position_offsets[0] = 0.0
        rpy_offsets[0] = 0.0
        bucket_xyz = base_xyz.reshape(1, 3) + position_offsets
        # Mount errors are defined as component-wise offsets in the URDF RPY convention.
        bucket_rpy = base_rpy.reshape(1, 3) + rpy_offsets

        stem, ext = os.path.splitext(asset_file)
        generated_files = []
        updated_files = 0
        for bucket_id in range(num_buckets):
            tree = copy.deepcopy(source_tree)
            joint = tree.getroot().find(f"./joint[@name='{mount_joint_name}']")
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin")
            origin.set("xyz", self._format_vector(bucket_xyz[bucket_id]))
            origin.set("rpy", self._format_vector(bucket_rpy[bucket_id]))

            generated_file = f"{stem}_mount_bucket_{bucket_id:02d}{ext}"
            generated_path = os.path.join(asset_root, generated_file)
            updated_files += int(self._write_xml_if_changed(tree, generated_path))
            generated_files.append(generated_file)

        self.arm_mount_bucket_position_offsets = position_offsets
        self.arm_mount_bucket_rpy_offsets = rpy_offsets
        self.arm_mount_bucket_tfs = np.concatenate((bucket_xyz, bucket_rpy), axis=1).astype(np.float32)
        print(
            f"[RoboDuet] mount TF URDF buckets ready: {len(generated_files)} ({updated_files} updated)",
            flush=True,
        )
        return generated_files

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        asset_files = self._generate_arm_mount_asset_files(asset_root, asset_file)
        self.asset_bucket_cycle_length = int(getattr(self.cfg.env, "asset_bucket_cycle_length", 0) or 0)
        num_asset_buckets = self.asset_bucket_cycle_length if self.asset_bucket_cycle_length > 0 else self.num_envs
        asset_files = asset_files[: min(len(asset_files), num_asset_buckets)]
        self.robot_assets = []
        print(f"[RoboDuet] loading {len(asset_files)} robot asset bucket(s) from {asset_root}", flush=True)
        for bucket_id, mount_asset_file in enumerate(asset_files):
            print(
                f"[RoboDuet] loading robot asset bucket {bucket_id + 1}/{len(asset_files)}: {mount_asset_file}",
                flush=True,
            )
            robot_asset = self.gym.load_asset(self.sim, asset_root, mount_asset_file, asset_options)
            if self.gym.get_asset_rigid_shape_count(robot_asset) == 0:
                raise RuntimeError(f"Failed to load robot asset '{os.path.join(asset_root, mount_asset_file)}'")
            self.robot_assets.append(robot_asset)
        print("[RoboDuet] robot asset bucket loading complete", flush=True)
        self.robot_asset = self.robot_assets[0]
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        self.default_dof_frictions = to_torch(
            dof_props_asset["friction"] if "friction" in dof_props_asset.dtype.names else np.zeros(self.num_dof),
            device=self.device,
            dtype=torch.float,
        )
        self.default_dof_dampings = to_torch(dof_props_asset["damping"], device=self.device, dtype=torch.float)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        # Resolve the EE body index by name (body ordering is URDF-specific, so
        # the __init__ default of 23 is only a placeholder). end_effector_state
        # tracks this body plus the fixed gripper offset (see WBCEnv).
        ee_body_name = getattr(self.cfg.asset, "ee_body_name", None)
        if ee_body_name is not None:
            if ee_body_name not in body_names:
                raise ValueError(f"asset.ee_body_name '{ee_body_name}' not in body_names: {body_names}")
            self.ee_idx = body_names.index(ee_body_name)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        arm_contact_names = []
        for name in getattr(self.cfg.asset, "arm_contact_bodies", []):
            arm_contact_names.extend([s for s in body_names if name in s])
        arm_contact_names = list(dict.fromkeys(arm_contact_names))
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        hip_body_names = []
        for name in self.cfg.asset.hip_joints:
            hip_body_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.arm_mount_bucket_ids = []
        self.arm_mount_tfs = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        # R5: the twins have to be nominal BEFORE the actor loop below, because
        # _process_rigid_shape_props and _process_rigid_body_props read
        # friction/restitution/payload per env as each actor is created and
        # those values are never revisited (randomize_rigids_after_start is off).
        self._ensure_grouping()
        self._nominalize_twins()
        self.arm_mount_bucket_of_env = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._randomize_gravity()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(
                -self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1), device=self.device
            ).squeeze(1)
            pos[1:2] += torch_rand_float(
                -self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1), device=self.device
            ).squeeze(1)

            if self.cfg.terrain.mesh_type == "plane":
                pos[2:3] += self.cfg.init_state.pos[2]
            start_pose.p = gymapi.Vec3(*pos)

            bucket_index = i % self.asset_bucket_cycle_length if self.asset_bucket_cycle_length > 0 else i
            bucket_id = bucket_index % len(self.robot_assets)
            # R5: bucket 0 is the deterministic nominal mount TF (its position
            # and rpy offsets are forced to zero when the buckets are built), so
            # the twin takes it regardless of where the cycle would land.
            if self._is_nominal_twin_env(i):
                bucket_id = 0
            # Recorded because this is a create-time choice that can never be
            # revisited, which makes it the one twin property an assertion
            # cannot reconstruct after the fact.
            self.arm_mount_bucket_of_env[i] = bucket_id
            robot_asset = self.robot_assets[bucket_id]
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            self.arm_mount_bucket_ids.append(bucket_id)
            self.arm_mount_tfs[i] = to_torch(
                self.arm_mount_bucket_tfs[bucket_id], device=self.device, dtype=torch.float
            )

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.arm_contact_indices = torch.zeros(
            len(arm_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(arm_contact_names)):
            self.arm_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], arm_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.hip_body_indices = torch.zeros(
            len(hip_body_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(hip_body_names)):
            self.hip_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], hip_body_names[i]
            )

        print("self.body_names: ", body_names)
        print("self.dof_names: ", self.dof_names)
        print("self.arm_contact_indices", self.arm_contact_indices)
        print("self.termination_contact_indices", self.termination_contact_indices)
        print("self.hip_joints_indices", self.hip_body_indices)

        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = int(self.cfg.env.recording_width_px)
            self.camera_props.height = int(self.cfg.env.recording_height_px)
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(
                self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0), gymapi.Vec3(0, 0, 0)
            )
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(
                    self.envs[self.num_train_envs], self.camera_props
                )
                self.gym.set_camera_location(
                    self.rendering_camera_eval,
                    self.envs[self.num_train_envs],
                    gymapi.Vec3(1.5, 1, 3.0),
                    gymapi.Vec3(0, 0, 0),
                )
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(
            self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0), gymapi.Vec3(bx, by, bz)
        )
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        capture_train = self._should_capture_recording_frame(eval_video=False)
        capture_eval = self._should_capture_recording_frame(eval_video=True) and self.eval_cfg is not None
        if not capture_train and not capture_eval:
            return

        if capture_train:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(
                self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0), gymapi.Vec3(bx, by, bz)
            )

        if capture_eval:
            bx, by, bz = (
                self.root_states[self.num_train_envs, 0],
                self.root_states[self.num_train_envs, 1],
                self.root_states[self.num_train_envs, 2],
            )
            self.gym.set_camera_location(
                self.rendering_camera_eval,
                self.envs[self.num_train_envs],
                gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                gymapi.Vec3(bx, by, bz),
            )

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        if capture_train:
            self.video_frame = self.gym.get_camera_image(
                self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR
            )
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self._arm_render_overlay_hook(self.video_frame, 0, self.envs[0], self.rendering_camera)
            self.video_frames.append(self.video_frame)

        if capture_eval:
            self.video_frame_eval = self.gym.get_camera_image(
                self.sim, self.envs[self.num_train_envs], self.rendering_camera_eval, gymapi.IMAGE_COLOR
            )
            self.video_frame_eval = self.video_frame_eval.reshape(
                (self.camera_props.height, self.camera_props.width, 4)
            )
            self._arm_render_overlay_hook(
                self.video_frame_eval,
                self.num_train_envs,
                self.envs[self.num_train_envs],
                self.rendering_camera_eval,
            )
            self.video_frames_eval.append(self.video_frame_eval)

    def _recording_frame_due(self):
        stride = max(1, int(getattr(self.cfg.env, "recording_frame_stride", 1)))
        return self.common_step_counter % stride == 0

    def _should_capture_recording_frame(self, eval_video=False):
        if not self._recording_frame_due():
            return False
        if eval_video:
            return (
                self.record_eval_now
                and self.complete_video_frames_eval is not None
                and len(self.complete_video_frames_eval) == 0
            )
        return self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval

    def _get_env_origins(self, env_ids, cfg):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum:
                max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum:
                min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(
                    min_terrain_level, max_terrain_level + 1, (len(env_ids),), device=self.device
                )
                self.terrain_types[env_ids] = torch.randint(
                    min_terrain_type, max_terrain_type + 1, (len(env_ids),), device=self.device
                )
            else:
                self.terrain_levels[env_ids] = torch.randint(
                    min_init_level, max_init_level + 1, (len(env_ids),), device=self.device
                )
                self.terrain_types[env_ids] = torch.div(
                    torch.arange(len(env_ids), device=self.device),
                    (len(env_ids) / cfg.terrain.num_cols),
                    rounding_mode="floor",
                ).to(torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]
            ]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            xx, yy = xx.to(self.device), yy.to(self.device)
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[: len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[: len(env_ids)]
            self.env_origins[env_ids, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.pretrained_reward_scales = vars(self.cfg.reward_scales)
        print(type(self.cfg.reward_scales), type(self.cfg.wbc.reward_scales))
        self.wbc_reward_scales = vars(self.cfg.wbc.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)

        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration
        )

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self, env_ids, cfg):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.env.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.env.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if cfg.terrain.mesh_type == "plane":
            return torch.zeros(len(env_ids), cfg.env.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(
            self.base_quat[env_ids].repeat(1, cfg.env.num_height_points), self.height_points[env_ids]
        ) + (self.root_states[env_ids, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale

    def _step_contact_targets(self):
        gaits = {"pronking": [0, 0, 0], "trotting": [0.5, 0, 0], "bounding": [0, 0.5, 0], "pacing": [0, 0, 0.5]}
        phases, offsets, bounds = gaits["trotting"]

        if self.cfg.commands.use_dynamic_gait:
            frequencies = self.commands_dog[:, 6]  # (num_envs,)
            durations = self.commands_dog[:, 10]  # (num_envs,)
        else:
            frequencies = 3.0
            durations = 0.5

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if self.cfg.commands.pacing_offset:
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + bounds,
                self.gait_indices + offsets,
                self.gait_indices + phases,
            ]
        else:
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            idxs[(torch.norm(self.commands_dog[:, :3], dim=1) < 0.1)] = 0.25  # mark stand
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            if self.cfg.commands.use_dynamic_gait:
                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs])
                )
            else:
                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (0.5 / (1 - durations))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
        )

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.0

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

        # base position
        self.root_states[env_ids] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
