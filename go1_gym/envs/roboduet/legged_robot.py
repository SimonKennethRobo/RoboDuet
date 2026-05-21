# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

assert gymtorch

import numpy as np
import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils import global_switch, quaternion_to_rpy
from go1_gym.utils.math_utils import get_scale_shift, quat_apply_yaw, wrap_to_pi
from go1_gym.utils.terrain import Terrain

from .wbc_env_config import RoboDuetCfg as Cfg


class LeggedRobot(BaseTask):
    def __init__(
        self,
        cfg: Cfg,
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

            pos_target = self.joint_pos_target + self.motor_offsets
            pos_target = pos_target * self.motor_strengths
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
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
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
        for _ in range(self.cfg.control.decimation):
            self._arm_decimation_hook()
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self._arm_post_sim_hook()

            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
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
        """Refresh sim tensors so newly-reset arm state is observable."""
        pass

    def _arm_resample_commands_train_hook(self, env_ids):
        """Per-episode arm command resampling alongside _resample_commands."""
        pass

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
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

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

        # reset robot states
        self._resample_commands(env_ids)
        self._arm_reset_hook(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)
        self._arm_post_reset_refresh_hook(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]["rew_" + key] = torch.mean(self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[: self.num_train_envs].float()
            )

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[: self.num_train_envs]

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

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        reward_scales = global_switch.get_reward_scales()

        self.rew_buf_dog[:] = 0.0
        self.rew_buf_pos_dog[:] = 0.0
        self.rew_buf_neg_dog[:] = 0.0
        self.rew_buf_arm[:] = 0.0
        self.rew_buf_pos_arm[:] = 0.0
        self.rew_buf_neg_arm[:] = 0.0
        for i in range(len(reward_scales)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * reward_scales[name]

            if name in ["vis_manip_commands_tracking_lpy", "vis_manip_commands_tracking_rpy"]:
                self.episode_sums[name] += rew
                continue

            self.rew_buf_dog += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos_dog += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg_dog += rew
            self.episode_sums[name] += rew

            # arm ignore the walking reward
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

        self.episode_sums["total"] += self.rew_buf_dog

        # add termination reward after clipping
        if "termination" in reward_scales:
            rew = self.reward_container._reward_termination() * reward_scales["termination"]
            self.rew_buf_dog += rew
            self.rew_buf_arm += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

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
                for i in range(8):
                    joint_name = f"zarx_j{i + 1}"
                    joint_idx = self.num_actions_loco + i
                    props[joint_idx]["stiffness"] = self.cfg.arm.control.stiffness_arm[joint_name]
                    props[joint_idx]["damping"] = self.cfg.arm.control.damping_arm[joint_name]

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

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self._arm_resample_commands_train_hook(env_ids)

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        curriculum = self.curricula[0]
        # update curricula based on terminated environment bins and categories
        task_rewards, success_thresholds = [], []
        for key in [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "tracking_contacts_shaped_force",
            "tracking_contacts_shaped_vel",
        ]:
            if key in self.command_sums.keys():
                task_rewards.append(self.command_sums[key][env_ids] / ep_len)
                success_thresholds.append(self.curriculum_thresholds[key] * self.pretrained_reward_scales[key])

        old_bins = self.env_command_bins[env_ids.cpu().numpy()]
        if len(success_thresholds) > 0:
            local_range = np.array([0.55, 0.55, 0.55, 1.0, 1.0])
            if self.cfg.commands.use_dynamic_gait:
                local_range = np.concatenate([local_range, np.array([0.55, 0.55, 0.55, 0.55, 0.55])])
            curriculum.update(
                old_bins,
                task_rewards,
                success_thresholds,
                local_range=local_range,
            )

        # sample from new category curricula
        new_commands, new_bin_inds = curriculum.sample(batch_size=len(env_ids))
        new_commands = torch.as_tensor(new_commands, dtype=self.commands_dog.dtype, device=self.device)

        self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
        self.env_command_categories[env_ids.cpu().numpy()] = 0

        if not self.cfg.hybrid.plan_vel and not self.cfg.arm.trajectory.enabled:
            self.commands_dog[env_ids, 0] = new_commands[:, 0]
            self.commands_dog[env_ids, 1] = new_commands[:, 1]
            self.commands_dog[env_ids, 2] = new_commands[:, 2]
            # self.commands_dog[env_ids, :2] *= (torch.norm(self.commands_dog[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

            # # Randomly select 10% of the environment to remain stationary
            # num_zero_envs = int(0.1 * len(env_ids))
            # zero_env_ids = torch.randperm(len(env_ids))[:num_zero_envs]
            # self.commands_dog[env_ids[zero_env_ids], :3] = 0

            zero_mask = torch.rand(len(env_ids), device=self.device) < 0.1
            if len(zero_mask.nonzero()) > 0:
                self.commands_dog[env_ids[zero_mask], :3] = 0

            self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
            self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
            self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1

        else:
            if not global_switch.switch_open:
                self.commands_dog[env_ids, 0] = new_commands[:, 0]
                self.commands_dog[env_ids, 1] = new_commands[:, 1]
                self.commands_dog[env_ids, 2] = new_commands[:, 2]

                # # Randomly select 10% of the environment to remain stationary
                # num_zero_envs = int(0.1 * len(env_ids))
                # zero_env_ids = torch.randperm(len(env_ids))[:num_zero_envs]
                # self.commands_dog[env_ids[zero_env_ids], :3] = 0

                zero_mask = torch.rand(len(env_ids), device=self.device) < 0.1
                if len(zero_mask.nonzero()) > 0:
                    self.commands_dog[env_ids[zero_mask], :3] = 0

                self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
                self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
                self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1

        if not global_switch.switch_open:
            self.commands_dog[env_ids, 3] = new_commands[:, 3]
            self.commands_dog[env_ids, 4] = new_commands[:, 4]

        if self.cfg.commands.use_dynamic_gait and not global_switch.switch_open:
            self.commands_dog[env_ids, 5] = new_commands[:, 5]
            self.commands_dog[env_ids, 6] = new_commands[:, 6]
            self.commands_dog[env_ids, 7] = new_commands[:, 7]
            self.commands_dog[env_ids, 8] = new_commands[:, 8]
            self.commands_dog[env_ids, 9] = new_commands[:, 9]
            standing_mask = torch.norm(self.commands_dog[env_ids, :3], dim=1) < 0.1
            if len(standing_mask.nonzero()) > 0:
                self.commands_dog[env_ids[standing_mask], 5] = 0.0

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ["trot"]

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from go1_gym.envs.base.curriculum import RewardThresholdCurriculum

            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            curriculum_kwargs = dict(
                seed=self.cfg.commands.curriculum_seed,
                x_vel=(
                    self.cfg.commands.limit_vel_x[0],
                    self.cfg.commands.limit_vel_x[1],
                    self.cfg.commands.num_bins_vel_x,
                ),
                y_vel=(
                    self.cfg.commands.limit_vel_y[0],
                    self.cfg.commands.limit_vel_y[1],
                    self.cfg.commands.num_bins_vel_y,
                ),
                yaw_vel=(
                    self.cfg.commands.limit_vel_yaw[0],
                    self.cfg.commands.limit_vel_yaw[1],
                    self.cfg.commands.num_bins_vel_yaw,
                ),
                body_pitch=(
                    self.cfg.commands.limit_body_pitch[0],
                    self.cfg.commands.limit_body_pitch[1],
                    self.cfg.commands.num_bins_body_pitch,
                ),
                body_roll=(
                    self.cfg.commands.limit_body_roll[0],
                    self.cfg.commands.limit_body_roll[1],
                    self.cfg.commands.num_bins_body_roll,
                ),
            )
            if self.cfg.commands.use_dynamic_gait:
                curriculum_kwargs.update(
                    gait_frequency=(
                        self.cfg.commands.limit_gait_frequency[0],
                        self.cfg.commands.limit_gait_frequency[1],
                        self.cfg.commands.num_bins_gait_frequency,
                    ),
                    footswing_height=(
                        self.cfg.commands.limit_footswing_height[0],
                        self.cfg.commands.limit_footswing_height[1],
                        self.cfg.commands.num_bins_footswing_height,
                    ),
                    stance_width=(
                        self.cfg.commands.limit_stance_width[0],
                        self.cfg.commands.limit_stance_width[1],
                        self.cfg.commands.num_bins_stance_width,
                    ),
                    stance_length=(
                        self.cfg.commands.limit_stance_length[0],
                        self.cfg.commands.limit_stance_length[1],
                        self.cfg.commands.num_bins_stance_length,
                    ),
                    gait_duration=(
                        self.cfg.commands.limit_gait_duration[0],
                        self.cfg.commands.limit_gait_duration[1],
                        self.cfg.commands.num_bins_gait_duration,
                    ),
                )
            self.curricula += [CurriculumClass(**curriculum_kwargs)]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(
                    lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                    binary_phases=self.cfg.commands.binary_phases,
                )
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [
                self.cfg.commands.lin_vel_x[0],
                self.cfg.commands.lin_vel_y[0],
                self.cfg.commands.ang_vel_yaw[0],
                self.cfg.commands.body_pitch_range[0],
                self.cfg.commands.body_roll_range[0],
            ]
        )
        high = np.array(
            [
                self.cfg.commands.lin_vel_x[1],
                self.cfg.commands.lin_vel_y[1],
                self.cfg.commands.ang_vel_yaw[1],
                self.cfg.commands.body_pitch_range[1],
                self.cfg.commands.body_roll_range[1],
            ]
        )
        if self.cfg.commands.use_dynamic_gait:
            low = np.concatenate(
                [
                    low,
                    np.array(
                        [
                            self.cfg.commands.limit_gait_frequency[0],
                            self.cfg.commands.limit_footswing_height[0],
                            self.cfg.commands.limit_stance_width[0],
                            self.cfg.commands.limit_stance_length[0],
                            self.cfg.commands.limit_gait_duration[0],
                        ]
                    ),
                ]
            )
            high = np.concatenate(
                [
                    high,
                    np.array(
                        [
                            self.cfg.commands.limit_gait_frequency[1],
                            self.cfg.commands.limit_footswing_height[1],
                            self.cfg.commands.limit_stance_width[1],
                            self.cfg.commands.limit_stance_length[1],
                            self.cfg.commands.limit_gait_duration[1],
                        ]
                    ),
                ]
            )
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        # self._teleport_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        self._arm_post_callback_hook()

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        # self._push_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

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
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base orientation: yaw / pitch / roll each randomized independently
        init_yaws = torch_rand_float(
            -cfg.terrain.yaw_init_range, cfg.terrain.yaw_init_range, (len(env_ids), 1), device=self.device
        )
        init_pitches = torch_rand_float(
            -getattr(cfg.terrain, "pitch_init_range", 0.0),
            getattr(cfg.terrain, "pitch_init_range", 0.0),
            (len(env_ids), 1), device=self.device,
        )
        init_rolls = torch_rand_float(
            -getattr(cfg.terrain, "roll_init_range", 0.0),
            getattr(cfg.terrain, "roll_init_range", 0.0),
            (len(env_ids), 1), device=self.device,
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
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[env_ids, 7:9] = torch_rand_float(
                -max_vel, max_vel, (len(env_ids), 2), device=self.device
            )  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

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

            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = (
                        self.cfg.dog.control.stiffness_leg[dof_name]
                        if i < self.num_actions_loco
                        else self.cfg.arm.control.stiffness_arm[dof_name]
                    )  # [N*m/rad]
                    self.d_gains[i] = (
                        self.cfg.dog.control.damping_leg[dof_name]
                        if i < self.num_actions_loco
                        else self.cfg.arm.control.damping_arm[dof_name]
                    )  # [N*m*s/rad]
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
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale_dog = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.body_pitch_cmd,
                self.obs_scales.body_roll_cmd,
                self.obs_scales.gait_freq_cmd,
                self.obs_scales.footswing_height_cmd,
                self.obs_scales.stance_width_cmd,
                self.obs_scales.stance_length_cmd,
                self.obs_scales.gait_duration_cmd,
            ],
            device=self.device,
            requires_grad=False,
        )[: self.cfg.dog.dog_num_commands]
        self.rew_buf_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg_dog = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # arm-side reward buffers are kept here because compute_reward() (in this base class)
        # writes to them; their semantic content is shaped entirely by WBCEnv.
        self.rew_buf_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self._arm_init_buffers_hook()

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

        for key in list(self.hybrid_reward_scales.keys()):
            self.hybrid_reward_scales[key] *= self.dt

        # update pretrained reward scales with hybrid reward scales
        for name, scale in self.pretrained_reward_scales.items():
            if name not in self.hybrid_reward_scales:
                self.hybrid_reward_scales[name] = scale

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.hybrid_reward_scales.items():
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
            for name in self.hybrid_reward_scales.keys()
        }
        self.episode_sums["total"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.hybrid_reward_scales.keys()
        }
        self.episode_sums_eval["total"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in list(self.hybrid_reward_scales.keys())
            + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual", "ep_timesteps"]
        }

        global_switch.set_reward_scales(self.hybrid_reward_scales, self.pretrained_reward_scales)

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

    def _arm_mount_randomization_cfg(self):
        return self.cfg.domain_rand

    def _format_xyz(self, xyz):
        return " ".join(f"{float(v):.8g}" for v in xyz)

    def _write_xml_if_changed(self, tree, path):
        xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
        if os.path.exists(path):
            with open(path, "rb") as f:
                if f.read() == xml_bytes:
                    return False
        with open(path, "wb") as f:
            f.write(xml_bytes)
        return True

    def _generate_arm_mount_asset_files(self, asset_root, asset_file):
        arm_dr = self._arm_mount_randomization_cfg()
        if arm_dr is None or not getattr(arm_dr, "randomize_mount_pos", False):
            self.arm_mount_bucket_offsets = np.zeros((1, 3), dtype=np.float32)
            return [asset_file]

        num_buckets = int(getattr(self.cfg.domain_rand, "mount_pos_buckets", 32))
        if num_buckets <= 1 or not asset_file.lower().endswith(".urdf"):
            self.arm_mount_bucket_offsets = np.zeros((1, 3), dtype=np.float32)
            return [asset_file]

        source_path = os.path.join(asset_root, asset_file)
        mount_joint_name = getattr(self.cfg.domain_rand, "mount_joint_name", "zarx5p2_mount")
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

        ranges = np.asarray(arm_dr.mount_pos_range, dtype=np.float32)
        if ranges.shape != (3, 2):
            raise ValueError(f"mount_pos_range must have shape (3, 2), got {ranges.shape}")

        rng = np.random.default_rng(int(getattr(self.cfg.domain_rand, "mount_pos_bucket_seed", 1234)))
        offsets = rng.uniform(ranges[:, 0], ranges[:, 1], size=(num_buckets, 3)).astype(np.float32)
        offsets[0] = 0.0

        stem, ext = os.path.splitext(asset_file)
        generated_files = []
        updated_files = 0
        for bucket_id, offset in enumerate(offsets):
            tree = ET.parse(source_path)
            joint = tree.getroot().find(f"./joint[@name='{mount_joint_name}']")
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin")
            origin.set("xyz", self._format_xyz(base_xyz + offset))
            origin.set("rpy", source_origin.get("rpy", "0 0 0"))

            generated_file = f"{stem}_mount_bucket_{bucket_id:02d}{ext}"
            generated_path = os.path.join(asset_root, generated_file)
            updated_files += int(self._write_xml_if_changed(tree, generated_path))
            generated_files.append(generated_file)

        self.arm_mount_bucket_offsets = offsets
        print(
            f"[RoboDuet] mount position URDF buckets ready: {len(generated_files)} "
            f"({updated_files} updated)",
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
        asset_files = asset_files[: min(len(asset_files), self.num_envs)]
        self.robot_assets = []
        print(f"[RoboDuet] loading {len(asset_files)} robot asset bucket(s) from {asset_root}", flush=True)
        for bucket_id, mount_asset_file in enumerate(asset_files):
            print(f"[RoboDuet] loading robot asset bucket {bucket_id + 1}/{len(asset_files)}: {mount_asset_file}", flush=True)
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

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
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
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
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

            bucket_id = i % len(self.robot_assets)
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
        print(type(self.cfg.reward_scales), type(self.cfg.hybrid.reward_scales))
        self.hybrid_reward_scales = vars(self.cfg.hybrid.reward_scales)
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
            frequencies = self.commands_dog[:, 5]  # (num_envs,)
            durations = self.commands_dog[:, 9]  # (num_envs,)
            footswing_height_cmd = self.commands_dog[:, 6]  # (num_envs,)
        else:
            frequencies = 3.0
            durations = 0.5
            footswing_height_cmd = 0.04

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
