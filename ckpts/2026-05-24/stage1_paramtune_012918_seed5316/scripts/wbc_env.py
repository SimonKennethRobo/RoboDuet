import sys

import cv2
import gym
import isaacgym

assert isaacgym
import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from params_proto import Meta

from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import (
    ee_twist_body_6d,
    get_scale_shift,
    pose_world_to_body_9d,
    quat_apply_yaw,
    quat_to_angle,
    quat_xyzw_to_rot6d,
    wrap_to_pi,
)

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .observation_builder import ObservationBuilder, clip_observation
from .trajectory_geometry import sample_trajectory_commands
from .wbc_env_config import RoboDuetCfg as Cfg

dog_cmd_idx = {
    "x_vel": 0,
    "y_vel": 1,
    "yaw_vel": 2,
    "body_pitch": 3,
    "body_roll": 4,
    "body_height": 5,
    "gait_frequency": 6,
    "footswing_height": 7,
    "stance_width": 8,
    "stance_length": 9,
    "gait_duration": 10,
    # named slices
    "velocity": slice(0, 3),
    "body_pose": slice(3, 6),
    "gait_params": slice(6, 11),
}


class WBCEnv(LeggedRobot):
    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        cfg: Cfg = None,
        eval_cfg: Cfg = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
        graphics_device_id=None,
    ):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
            eval_cfg,
            initial_dynamics_dict,
            graphics_device_id,
        )

    # ============================================================
    # Geometry / EE helpers (formerly in LeggedRobot)
    # ============================================================

    def quat_to_angle(self, quat):
        return quat_to_angle(quat.to(self.device))

    def _pose_world_to_body_9d(self, pos_world, quat_world, env_ids=None):
        if env_ids is None:
            base_pos = self.base_pos
            base_quat = self.base_quat
        else:
            base_pos = self.base_pos[env_ids]
            base_quat = self.base_quat[env_ids]
        return pose_world_to_body_9d(pos_world, quat_world, base_pos, base_quat)

    def get_ee_pose_body_9d(self, env_ids=None):
        if env_ids is None:
            return self._pose_world_to_body_9d(
                self.end_effector_state[:, :3], self.end_effector_state[:, 3:7]
            )
        return self._pose_world_to_body_9d(
            self.end_effector_state[env_ids, :3],
            self.end_effector_state[env_ids, 3:7],
            env_ids,
        )

    def get_ee_twist_body(self):
        return ee_twist_body_6d(self.end_effector_state, self.root_states, self.base_quat, self.num_envs)

    def _trajectory_points_body_9d(self, waypoint_ids):
        env_ids = torch.arange(self.num_envs, device=self.device)
        flat_env_ids = env_ids[:, None].expand(-1, waypoint_ids.shape[1]).reshape(-1)
        flat_wp_ids = waypoint_ids.reshape(-1)
        points = self._pose_world_to_body_9d(
            self.traj_pos_world[flat_env_ids, flat_wp_ids],
            self.traj_quat_world[flat_env_ids, flat_wp_ids],
            flat_env_ids,
        )
        return points.view(self.num_envs, waypoint_ids.shape[1] * 9)

    def get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        grasper_offset = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], grasper_offset)
        grasper_in_world = self.end_effector_state[env_ids, :3] + grasper_move_in_world

        x = torch.cos(yaw) * (grasper_in_world[:, 0] - self.root_states[env_ids, 0]) + torch.sin(yaw) * (
            grasper_in_world[:, 1] - self.root_states[env_ids, 1]
        )
        y = -torch.sin(yaw) * (grasper_in_world[:, 0] - self.root_states[env_ids, 0]) + torch.cos(yaw) * (
            grasper_in_world[:, 1] - self.root_states[env_ids, 1]
        )
        z = torch.mean(grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.38

        l = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        p = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
        y_aw = torch.atan2(y, x)
        return torch.stack([l, p, y_aw], dim=-1)

    def get_alpha_beta_gamma_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), self.end_effector_state[:, 3:7])
        return self.quat_to_angle(ee_in_base_quats)

    def _lpy_to_world_xyz_env(self, env_ids):
        l = self.commands_arm[env_ids, 0]
        p = self.commands_arm[env_ids, 1]
        y = self.commands_arm[env_ids, 2]
        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[env_ids, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[env_ids, 1]
        z_ = z + self.measured_heights + 0.38
        return x_, y_, z_

    def _get_object_pose_in_ee(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        x, y, z = self._lpy_to_world_xyz_env(env_ids)
        xyz = torch.stack([x, y, z], dim=-1)
        dxyz = xyz - self.end_effector_state[:, 0:3]
        self.obj_pose_in_ee[:] = quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)
        return self.obj_pose_in_ee[:]

    def _get_object_abg_in_ee(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        rot_in_world = quat_mul(base_quats, self.obj_quats)
        rot_in_ee = quat_mul(quat_conjugate(self.end_effector_state[:, 3:7]), rot_in_world)
        self.obj_abg_in_ee[:] = self.quat_to_angle(rot_in_ee)
        return self.obj_abg_in_ee[:]

    # ============================================================
    # Arm / trajectory command resampling
    # ============================================================

    def _resample_user_commands(self, env_ids):
        if self.cfg.arm.trajectory.user_cmd_mode == "random":
            self.user_vel_cmd[env_ids, 0] = torch_rand_float(
                self.cfg.arm.trajectory.user_lin_vel_x[0],
                self.cfg.arm.trajectory.user_lin_vel_x[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
            self.user_vel_cmd[env_ids, 1] = torch_rand_float(
                self.cfg.arm.trajectory.user_lin_vel_y[0],
                self.cfg.arm.trajectory.user_lin_vel_y[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
            self.user_vel_cmd[env_ids, 2] = torch_rand_float(
                self.cfg.arm.trajectory.user_ang_vel_yaw[0],
                self.cfg.arm.trajectory.user_ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
        else:
            self.user_vel_cmd[env_ids] = 0.0

    def _traj_curriculum_params(self, env_ids):
        """Interpolate length and s_curve_amplitude from curriculum level."""
        max_level = max(1, self.cfg.arm.trajectory.curriculum_levels - 1)
        difficulty = self.traj_curriculum_level[env_ids].float() / max_level  # (n,)

        lo, hi = self.cfg.arm.trajectory.length_range
        length = lo + (hi - lo) * difficulty

        lo_a, hi_a = self.cfg.arm.trajectory.s_curve_amplitude_range
        s_amplitude = lo_a + (hi_a - lo_a) * difficulty

        return length, s_amplitude

    def _resample_trajectory_commands(self, env_ids):
        self._resample_user_commands(env_ids)
        length, s_amplitude = self._traj_curriculum_params(env_ids)
        traj_pos, traj_quat, target_time = sample_trajectory_commands(
            self.cfg,
            self.end_effector_state,
            self.base_quat,
            env_ids,
            self.traj_num_waypoints,
            self.device,
            length=length,
            s_curve_amplitude=s_amplitude,
        )
        self.traj_pos_world[env_ids] = traj_pos
        self.traj_quat_world[env_ids] = traj_quat
        self.traj_target_time[env_ids] = target_time
        self.T_trajs[env_ids] = self.traj_target_time[env_ids]
        self.arm_time_buf[env_ids] = 0
        self.traj_elapsed_time[env_ids] = 0.0
        self.traj_progress_idx[env_ids] = 0
        self.traj_complete_buf[env_ids] = False
        self.traj_final_pos_error[env_ids] = 0.0
        self.traj_final_rot_error[env_ids] = 0.0
        self.traj_ee_pose_body_history[env_ids] = 0.0
        self.traj_target_body_history[env_ids] = 0.0
        self.traj_visited_mask[env_ids] = False
        self.arm_delta_vel_cmd[env_ids] = 0.0
        self.commands_dog[env_ids, dog_cmd_idx["velocity"]] = self.user_vel_cmd[env_ids]

    def _resample_T_traj(self, env_ids):
        time_range = (self.cfg.arm.commands.T_traj[1] - self.cfg.arm.commands.T_traj[0]) / self.dt
        time_interval = torch.randint(
            0,
            int(time_range + 1),
            (len(env_ids),),
            device=self.device,
        ).to(dtype=self.T_trajs.dtype)
        self.T_trajs[env_ids] = (
            torch.ones_like(self.T_trajs[env_ids]) * self.cfg.arm.commands.T_traj[0] + time_interval * self.dt
        )
        self.arm_time_buf[env_ids] = torch.zeros_like(self.arm_time_buf[env_ids])

    def _resample_arm_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if not global_switch.switch_open:
            return
        if self.cfg.arm.trajectory.enabled:
            self._resample_trajectory_commands(env_ids)
            return

        self.commands_arm[env_ids, 0] = torch_rand_float(
            self.cfg.arm.commands.l[0], self.cfg.arm.commands.l[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()
        self.commands_arm[env_ids, 1] = torch_rand_float(
            self.cfg.arm.commands.p[0], self.cfg.arm.commands.p[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()
        self.commands_arm[env_ids, 2] = torch_rand_float(
            self.cfg.arm.commands.y[0], self.cfg.arm.commands.y[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()

        self.commands_arm_obs[env_ids, 0] = self.commands_arm[env_ids, 0]
        self.commands_arm_obs[env_ids, 1] = self.commands_arm[env_ids, 1]
        self.commands_arm_obs[env_ids, 2] = self.commands_arm[env_ids, 2]

        roll = torch_rand_float(
            self.cfg.arm.commands.roll_ee[0], self.cfg.arm.commands.roll_ee[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()
        pitch = torch_rand_float(
            self.cfg.arm.commands.pitch_ee[0], self.cfg.arm.commands.pitch_ee[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()
        yaw = torch_rand_float(
            self.cfg.arm.commands.yaw_ee[0], self.cfg.arm.commands.yaw_ee[1], (env_ids.shape[0], 1), device=self.device
        ).squeeze()

        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))
        self.obj_quats[env_ids] = quats.reshape(-1, 4)

        assert torch.allclose(
            torch.norm(self.obj_quats[env_ids], dim=1), torch.ones(len(env_ids)).to(self.device), atol=1e-5
        ), "quats is not unit vector."

        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        self.visual_rpy[env_ids] = quaternion_to_rpy(self.obj_quats[env_ids]).to(self.device)
        self.target_abg[env_ids] = self.quat_to_angle(self.obj_quats[env_ids])
        if self.cfg.use_rot6d:
            r6d = quat_xyzw_to_rot6d(quats)
            self.commands_arm_obs[env_ids, 3:9] = r6d.to(self.device)
        else:
            rpy = self.quat_to_angle(self.obj_quats[env_ids])
            self.commands_arm_obs[env_ids, 3] = rpy[:, 0]
            self.commands_arm_obs[env_ids, 4] = rpy[:, 1]
            self.commands_arm_obs[env_ids, 5] = rpy[:, 2]

        self._resample_T_traj(env_ids)

    def _step_traj_track(self):
        if not self.cfg.arm.trajectory.enabled or not global_switch.switch_open:
            return
        self.traj_elapsed_time[:] = self.arm_time_buf.float() * self.dt
        progress = self.traj_elapsed_time / torch.clamp(self.traj_target_time, min=self.dt)
        self.traj_progress_idx[:] = torch.clamp(
            (progress * (self.traj_num_waypoints - 1)).long(),
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        ee_pose = self.get_ee_pose_body_9d()
        target_pose = self._pose_world_to_body_9d(
            self.traj_pos_world[env_ids, self.traj_progress_idx],
            self.traj_quat_world[env_ids, self.traj_progress_idx],
            env_ids,
        )
        self.traj_ee_pose_body_history[env_ids, self.traj_progress_idx] = ee_pose
        self.traj_target_body_history[env_ids, self.traj_progress_idx] = target_pose
        self.traj_visited_mask[env_ids, self.traj_progress_idx] = True

        target_final = self._pose_world_to_body_9d(self.traj_pos_world[:, -1], self.traj_quat_world[:, -1])
        self.traj_final_pos_error[:] = torch.norm(ee_pose[:, :3] - target_final[:, :3], dim=-1)
        self.traj_final_rot_error[:] = torch.norm(ee_pose[:, 3:] - target_final[:, 3:], dim=-1)
        final_time_reached = self.traj_progress_idx >= self.traj_num_waypoints - 1
        self.traj_complete_buf[:] = (
            final_time_reached
            & (self.traj_final_pos_error < self.cfg.arm.trajectory.completion_pos_threshold)
            & (self.traj_final_rot_error < self.cfg.arm.trajectory.completion_rot_threshold)
        )
        self.traj_episode_success_buf |= self.traj_complete_buf

    def get_trajectory_window_obs(self):
        waypoint_ids = torch.clamp(
            self.traj_progress_idx[:, None] + self.traj_window_offsets[None, :],
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        return self._trajectory_points_body_9d(waypoint_ids)

    def get_full_trajectory_privileged_obs(self):
        waypoint_ids = torch.arange(self.traj_num_waypoints, device=self.device, dtype=torch.long)
        waypoint_ids = waypoint_ids.unsqueeze(0).expand(self.num_envs, -1)
        return self._trajectory_points_body_9d(waypoint_ids)

    def get_trajectory_error_sum(self):
        pos_error = torch.sum(
            torch.square(self.traj_ee_pose_body_history[..., :3] - self.traj_target_body_history[..., :3]), dim=-1
        )
        rot_error = torch.sum(
            torch.square(self.traj_ee_pose_body_history[..., 3:] - self.traj_target_body_history[..., 3:]), dim=-1
        )
        error = (
            self.cfg.arm.trajectory.pos_error_scale * pos_error
            + self.cfg.arm.trajectory.rot_error_scale * rot_error
        )
        error = error * self.traj_visited_mask.float()
        denom = torch.clamp(self.traj_visited_mask.float().sum(dim=-1), min=1.0)
        return torch.sum(error, dim=-1) / denom

    # ============================================================
    # EE force / arm action curriculum
    # ============================================================

    def resample_force(self, env_ids):
        self.ee_forces[env_ids, self.ee_idx] = torch_rand_float(
            -self.cfg.domain_rand.max_force, self.cfg.domain_rand.max_force, (len(env_ids), 3), device=self.device
        )
        time_range = (self.cfg.commands.T_force_range[1] - self.cfg.commands.T_force_range[0]) / self.dt
        time_interval = torch.randint(
            0, int(time_range + 1), (len(env_ids),), device=self.device,
        ).to(dtype=self.T_force.dtype)
        self.T_force[env_ids] = (
            torch.ones_like(self.T_force[env_ids]) * self.cfg.commands.T_force_range[0] + time_interval * self.dt
        )
        self.force_time_buf[env_ids] = torch.zeros_like(self.force_time_buf[env_ids])
        self.add_force_flag[env_ids] = torch.rand_like(self.add_force_flag[env_ids])
        self.ee_forces[env_ids, self.ee_idx] *= (
            self.add_force_flag[env_ids] > self.cfg.commands.add_force_thres
        ).reshape(-1, 1)

    def add_continue_force(self):
        if not self.cfg.domain_rand.randomize_end_effector_force:
            return
        self.force_positions = self.rigid_body_state[..., :3].clone().reshape(self.num_envs, -1, 3)
        self.ee_forces[:, self.ee_idx] = 10
        self.ee_forces[:, 0] = 20

        offset = torch_rand_float(
            -self.cfg.domain_rand.max_force_offset,
            self.cfg.domain_rand.max_force_offset,
            (self.num_envs, 3),
            device=self.device,
        )
        self.force_positions[:, self.ee_idx] += offset

        assert self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.ee_forces.reshape(-1, 3)),
            gymtorch.unwrap_tensor(self.force_positions.reshape(-1, 3)),
        ), "Failed to apply force at position."

    def _stage1_arm_curriculum_active(self):
        return bool(getattr(self.cfg.env, "stage1_arm_curriculum", False)) and not global_switch.switch_open

    def _get_stage1_arm_curriculum_intensity(self):
        if not self._stage1_arm_curriculum_active():
            return 0.0
        ramp_iters = max(
            1,
            int(getattr(global_switch, "stage1_arm_ramp_iterations", global_switch.pretrained_to_hybrid_start)),
        )
        stage1_iter = getattr(global_switch, "stage1_count", global_switch.count)
        progress = min(1.0, max(0.0, stage1_iter / ramp_iters))
        fixed_fraction = min(1.0, max(0.0, float(self.cfg.env.stage1_arm_fixed_fraction)))
        saturation_fraction = min(1.0, max(fixed_fraction, float(
            getattr(self.cfg.env, "stage1_arm_saturation_fraction", 1.0)
        )))
        if progress <= fixed_fraction:
            return 0.0
        if progress >= saturation_fraction:
            return 1.0
        return (progress - fixed_fraction) / max(1e-6, saturation_fraction - fixed_fraction)

    def _apply_stage1_arm_curriculum_actions(self):
        if not self._stage1_arm_curriculum_active():
            return
        intensity = self._get_stage1_arm_curriculum_intensity()
        self.stage1_arm_curriculum_intensity = intensity
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        arm_default = self.default_dof_pos[:, arm_slice]

        if intensity <= 0.0:
            self.stage1_arm_target_offset.zero_()
            self.stage1_arm_target_vel.zero_()
            self.stage1_arm_target_accel.zero_()
            # Set actions so pos_target matches stage1_arm_fixed_dof_pos.
            # pos_target = action_scale * action + default_dof_pos
            # action = (fixed_pos - default_arm) / action_scale
            self.actions[:, arm_slice] = (
                (self.stage1_arm_fixed_dof_pos - arm_default) / self.cfg.control.action_scale
            )
            return

        resample_steps = max(1, int(self.cfg.env.stage1_arm_accel_resample_time_s / self.dt))
        if self.common_step_counter % resample_steps == 0:
            accel = torch.rand_like(self.stage1_arm_target_accel) * 2.0 - 1.0
            self.stage1_arm_target_accel[:] = accel * self.cfg.env.stage1_arm_max_accel * intensity

        step_accel = self.stage1_arm_target_accel
        zero_accel_prob = min(1.0, max(0.0, float(getattr(self.cfg.env, "stage1_arm_zero_accel_probability", 0.0))))
        if zero_accel_prob > 0.0:
            use_zero_accel = torch.rand(self.num_envs, 1, device=self.device) < zero_accel_prob
            step_accel = torch.where(use_zero_accel, torch.zeros_like(step_accel), step_accel)

        self.stage1_arm_target_vel += step_accel * self.dt
        max_vel = self.cfg.env.stage1_arm_max_vel * intensity
        self.stage1_arm_target_vel[:] = torch.clamp(self.stage1_arm_target_vel, -max_vel, max_vel)

        zero_vel_prob = min(1.0, max(0.0, float(getattr(self.cfg.env, "stage1_arm_zero_vel_probability", 0.0))))
        if zero_vel_prob > 0.0:
            use_zero_vel = torch.rand(self.num_envs, 1, device=self.device) < zero_vel_prob
            self.stage1_arm_target_vel[:] = torch.where(
                use_zero_vel, torch.zeros_like(self.stage1_arm_target_vel), self.stage1_arm_target_vel
            )

        self.stage1_arm_target_offset += self.stage1_arm_target_vel * self.dt

        raw_target = arm_default + self.stage1_arm_target_offset
        lower = self.dof_pos_limits[arm_slice, 0].unsqueeze(0)
        upper = self.dof_pos_limits[arm_slice, 1].unsqueeze(0)
        hit_limit = ((raw_target < lower) & (self.stage1_arm_target_vel < 0.0)) | (
            (raw_target > upper) & (self.stage1_arm_target_vel > 0.0)
        )
        self.stage1_arm_target_vel[hit_limit] *= -1
        target = torch.clamp(raw_target, lower, upper)

        self.actions[:, arm_slice] = (target - arm_default) / self.cfg.control.action_scale

    def _keep_arm_fixed(self):
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)

        if global_switch.switch_open:
            # Stage2: arm is controlled by policy, nothing to fix.
            return
        elif self._stage1_arm_curriculum_active():
            intensity = self._get_stage1_arm_curriculum_intensity()
            if intensity > 0.0:
                # Curriculum active: arm moves via position targets from _apply_stage1_arm_curriculum_actions.
                # Fix DOFs beyond the arm (none exist, but harmless).
                idx = self.num_actions_loco + self.num_actions_arm
                self.dof_pos[:, idx:] = self.default_dof_pos[:, idx:]
                self.dof_vel[:, idx:] = 0.0
            else:
                # Fixed phase: hold the per-env randomized reset position.
                self.dof_pos[:, arm_slice] = self.stage1_arm_fixed_dof_pos
                self.dof_vel[:, arm_slice] = 0.0
        else:
            # No curriculum: fix arm at per-env randomized reset position.
            self.dof_pos[:, arm_slice] = self.stage1_arm_fixed_dof_pos
            self.dof_vel[:, arm_slice] = 0.0

        ret = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        assert ret, "[ERROR] Failed to set dof state."

    # ============================================================
    # LeggedRobot hook overrides
    # ============================================================

    def _arm_init_buffers_hook(self):
        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.stage1_arm_target_offset = torch.zeros(
            self.num_envs, self.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.stage1_arm_target_vel = torch.zeros_like(self.stage1_arm_target_offset)
        self.stage1_arm_target_accel = torch.zeros_like(self.stage1_arm_target_offset)
        self.stage1_arm_curriculum_intensity = 0.0

        self.commands_arm = torch.zeros(
            self.num_envs, self.cfg.arm.arm_num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_arm_obs = torch.zeros(
            self.num_envs, self.cfg.arm.arm_num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_abg = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.end_effector_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]
        self.x_vector = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.y_vector = to_torch([0.0, 1.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.z_vector = to_torch([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self.visual_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_arm_lpy_range = torch.tensor(
            [
                self.cfg.arm.commands.l[1] - self.cfg.arm.commands.l[0],
                self.cfg.arm.commands.p[1] - self.cfg.arm.commands.p[0],
                self.cfg.arm.commands.y[1] - self.cfg.arm.commands.y[0],
            ],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)
        self.commands_arm_rpy_range = torch.tensor(
            [
                self.cfg.arm.commands.roll_ee[1] - self.cfg.arm.commands.roll_ee[0],
                self.cfg.arm.commands.pitch_ee[1] - self.cfg.arm.commands.pitch_ee[0],
                self.cfg.arm.commands.yaw_ee[1] - self.cfg.arm.commands.yaw_ee[0],
            ],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)

        self.obj_obs_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_obs_abg_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_abg_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_quats = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)

        self.T_trajs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.T_force = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.add_force_flag = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_forces = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.force_positions = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)

        self.num_plan_actions = self.cfg.arm.num_actions_arm_cd - self.num_actions_arm
        self.last_plan_actions = torch.zeros(
            self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.plan_actions = torch.zeros(
            self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.traj_window_offsets = torch.tensor(
            self.cfg.arm.trajectory.window_offsets, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.traj_num_waypoints = int(self.cfg.arm.trajectory.num_waypoints)
        self.traj_pos_world = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_quat_world = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_quat_world[..., 3] = 1.0
        self.traj_progress_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.traj_target_time = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_elapsed_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_complete_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.traj_episode_success_buf = torch.zeros_like(self.traj_complete_buf)
        self.traj_final_pos_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_final_rot_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_ee_pose_body_history = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 9, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_target_body_history = torch.zeros_like(self.traj_ee_pose_body_history)
        self.traj_visited_mask = torch.zeros(
            self.num_envs, self.traj_num_waypoints, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.user_vel_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_delta_vel_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_ee_twist_body = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        # Per-env arm target position when arm is fixed (intensity=0). Populated at reset
        # with randomized init noise so _keep_arm_fixed holds a varied pose, not always 0.
        arm_default = self.default_dof_pos[
            :, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm
        ]
        self.stage1_arm_fixed_dof_pos = arm_default.expand(self.num_envs, -1).clone()

        # Arm rigid-body domain rand buffers
        arm_body_names = [n for n in self.body_names if "zarx" in n.lower()]
        self.arm_body_indices = [self.body_names.index(n) for n in arm_body_names]
        n_arm_bodies = len(self.arm_body_indices)
        # Default masses fetched from env 0 actor (available after _create_envs)
        props0 = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.arm_default_link_masses = torch.tensor(
            [props0[i].mass for i in self.arm_body_indices],
            dtype=torch.float, device=self.device,
        )
        self.arm_default_link_coms = torch.tensor(
            [[props0[i].com.x, props0[i].com.y, props0[i].com.z] for i in self.arm_body_indices],
            dtype=torch.float, device=self.device,
        )
        self.arm_link_mass_scales = torch.ones(
            self.num_envs, n_arm_bodies, dtype=torch.float, device=self.device
        )
        self.arm_link_com_offsets = torch.zeros(
            self.num_envs, n_arm_bodies, 3, dtype=torch.float, device=self.device
        )

        # Deferred resample: set at episode reset, cleared after first post-simulate step.
        # Avoids using stale FK state (set_dof_state_tensor_indexed does not propagate FK
        # until the next gym.simulate() call).
        self.traj_deferred_resample = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )

        self.traj_curriculum_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )

    def _arm_pre_step_hook(self):
        self._apply_stage1_arm_curriculum_actions()

    def _arm_decimation_hook(self):
        self.add_continue_force()

    def _arm_post_sim_hook(self):
        if self.cfg.env.keep_arm_fixed:
            self._keep_arm_fixed()

    def _arm_post_physics_hook(self):
        self.arm_time_buf += 1
        self.force_time_buf += 1
        self.end_effector_state[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]
        self._step_traj_track()
        if self.cfg.arm.trajectory.enabled and global_switch.switch_open:
            self.prev_ee_twist_body[:] = self.get_ee_twist_body()

    def _arm_check_termination_hook(self):
        if global_switch.switch_open and self.cfg.hybrid.rewards.use_terminal_roll:
            reverse_buf1 = torch.logical_and(
                self.roll > self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] > 0.0
            )
            reverse_buf2 = torch.logical_and(
                self.roll < -self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] < 0.0
            )
            self.reverse_buf |= reverse_buf1 | reverse_buf2

        p_align = self.commands_arm[:, 1]
        l_align = self.commands_arm[:, 0]
        self.delta_z = l_align * torch.sin(p_align) + 0.38 - self.base_pos[:, 2]

        if global_switch.switch_open and self.cfg.hybrid.rewards.use_terminal_pitch:
            reverse_buf3 = torch.logical_and(
                self.pitch < -self.cfg.hybrid.rewards.terminal_body_pitch,
                self.delta_z < -self.cfg.hybrid.rewards.headupdown_thres,
            )
            reverse_buf4 = torch.logical_and(
                self.pitch > self.cfg.hybrid.rewards.terminal_body_pitch,
                self.delta_z > self.cfg.hybrid.rewards.headupdown_thres,
            )
            self.reverse_buf |= reverse_buf3 | reverse_buf4

        if global_switch.switch_open:
            time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.6
            self.reverse_buf = self.reverse_buf & time_exceed_half

    def _arm_reset_hook(self, env_ids):
        if self.cfg.arm.trajectory.enabled and global_switch.switch_open:
            self._update_traj_curriculum(env_ids)
        elif not self.cfg.arm.trajectory.enabled:
            self._resample_arm_commands(env_ids)
        # stage1_arm_target_offset / vel / accel are re-initialised in
        # _arm_post_reset_refresh_hook (after the randomised dof_pos is known).
        self.prev_ee_twist_body[env_ids] = 0.0

    def _update_traj_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return
        max_level = self.cfg.arm.trajectory.curriculum_levels - 1
        completed = self.traj_episode_success_buf[env_ids]
        advance_ids = env_ids[completed]
        if len(advance_ids) > 0:
            self.traj_curriculum_level[advance_ids] = torch.clamp(
                self.traj_curriculum_level[advance_ids] + 1, max=max_level
            )
        self.traj_episode_success_buf[env_ids] = False

    def _arm_post_reset_refresh_hook(self, env_ids):
        if len(env_ids) == 0:
            return

        # ---- Arm DOF domain rand (Kp/Kd/strength/offset) ----
        # Must override the arm slice AFTER _randomize_dof_props has set leg-wide values.
        self._randomize_arm_dof_props(env_ids)
        # ---- Arm rigid-body domain rand (link mass / COM) ----
        self._randomize_arm_rigid_body_props(env_ids)

        # Add additive noise to arm joint positions at reset.
        # Arm default angles are 0, so the multiplicative noise in _reset_dofs has no effect;
        # we apply ±noise [rad] here instead.
        noise = getattr(self.cfg.env, "stage1_arm_init_dof_pos_noise", 0.0)
        if noise > 0.0:
            arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
            self.dof_pos[env_ids, arm_slice] += torch_rand_float(
                -noise, noise,
                (len(env_ids), self.num_actions_arm),
                device=self.device,
            )
            # Clamp to DOF limits to avoid out-of-range positions
            self.dof_pos[env_ids] = torch.clamp(
                self.dof_pos[env_ids],
                self.dof_pos_limits[:, 0],
                self.dof_pos_limits[:, 1],
            )
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

        # Unified initial arm state: record the randomized position and initialise the
        # curriculum offset so both fix (intensity=0) and disturbance (intensity>0) phases
        # start from the same randomised joint configuration.
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        self.stage1_arm_fixed_dof_pos[env_ids] = self.dof_pos[env_ids, arm_slice].clone()

        arm_default = self.default_dof_pos[:, arm_slice]  # (1, num_actions_arm)
        self.stage1_arm_target_offset[env_ids] = self.stage1_arm_fixed_dof_pos[env_ids] - arm_default
        self.stage1_arm_target_vel[env_ids] = 0.0
        self.stage1_arm_target_accel[env_ids] = 0.0

        if not self.cfg.arm.trajectory.enabled:
            return
        # Deferred trajectory resample (FK not valid until next simulate()).
        self.traj_deferred_resample[env_ids] = True

    def _randomize_arm_dof_props(self, env_ids):
        """Override arm DOF slice in Kp/Kd/strength/offset buffers with stage-specific ranges."""
        if len(env_ids) == 0:
            return
        arm_dr = (
            self.cfg.domain_rand.stage1_arm
            if not global_switch.switch_open
            else self.cfg.domain_rand.stage2_arm
        )
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        n = len(env_ids)

        if arm_dr.randomize_Kp_factor:
            lo, hi = arm_dr.Kp_factor_range
            self.Kp_factors[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_Kd_factor:
            lo, hi = arm_dr.Kd_factor_range
            self.Kd_factors[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_motor_strength:
            lo, hi = arm_dr.motor_strength_range
            self.motor_strengths[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_motor_offset:
            r = arm_dr.motor_offset_range
            self.motor_offsets[env_ids, arm_slice] = torch_rand_float(
                -r, r, (n, self.num_actions_arm), device=self.device
            )

    def _arm_post_dof_randomization_hook(self, env_ids):
        self._randomize_arm_dof_props(env_ids)

    def _randomize_arm_rigid_body_props(self, env_ids):
        """Randomize arm link masses and COM offsets; push to the physics engine."""
        arm_dr = (
            self.cfg.domain_rand.stage1_arm
            if not global_switch.switch_open
            else self.cfg.domain_rand.stage2_arm
        )
        if not self.arm_body_indices:
            return
        if not arm_dr.randomize_link_mass and not arm_dr.randomize_link_com:
            return

        n = len(env_ids)
        n_arm = len(self.arm_body_indices)

        if arm_dr.randomize_link_mass:
            lo, hi = arm_dr.link_mass_range
            self.arm_link_mass_scales[env_ids] = (
                torch.rand(n, n_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_link_com:
            r = arm_dr.link_com_range
            self.arm_link_com_offsets[env_ids] = (
                torch.rand(n, n_arm, 3, device=self.device) * 2 * r - r
            )

        for env_id in env_ids.tolist():
            props = self.gym.get_actor_rigid_body_properties(
                self.envs[env_id], self.actor_handles[env_id]
            )
            for k, body_idx in enumerate(self.arm_body_indices):
                if arm_dr.randomize_link_mass:
                    props[body_idx].mass = (
                        self.arm_default_link_masses[k].item()
                        * self.arm_link_mass_scales[env_id, k].item()
                    )
                if arm_dr.randomize_link_com:
                    com = self.arm_default_link_coms[k] + self.arm_link_com_offsets[env_id, k]
                    props[body_idx].com = gymapi.Vec3(com[0].item(), com[1].item(), com[2].item())
            self.gym.set_actor_rigid_body_properties(
                self.envs[env_id], self.actor_handles[env_id], props, recomputeInertia=True
            )

    def _arm_resample_commands_train_hook(self, env_ids):
        if self.cfg.arm.trajectory.enabled and global_switch.switch_open:
            self._resample_user_commands(env_ids)
            self.commands_dog[env_ids, dog_cmd_idx["velocity"]] = self.user_vel_cmd[env_ids] + self.arm_delta_vel_cmd[env_ids]

    def _get_privileged_dof_slice(self, policy):
        if policy == "dog":
            return slice(0, self.num_actions_loco)
        if policy == "arm":
            return slice(self.num_actions_loco, self.num_actions_loco + self.cfg.arm.num_actions_arm_cd)
        raise ValueError(f"Unknown privileged observation policy: {policy}")

    def _get_physics_privileged_observations(self, policy):
        privileged_obs_buf = torch.empty(self.num_envs, 0, device=self.device)
        dof_slice = self._get_privileged_dof_slice(policy)

        if self.cfg.env.priv_observe_friction:
            scale, shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.friction_coeffs[:, 0].unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            scale, shift = get_scale_shift(self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.ground_friction_coeffs.unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_restitution:
            scale, shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_base_mass:
            scale, shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.payloads.unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_com_displacement:
            scale, shift = get_scale_shift(self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.com_displacements - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_motor_strength:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_strengths[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_motor_offset:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_offsets[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_Kp_factor:
            scale, shift = get_scale_shift(self.cfg.normalization.Kp_factor_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.Kp_factors[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_Kd_factor:
            scale, shift = get_scale_shift(self.cfg.normalization.Kd_factor_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.Kd_factors[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_joint_friction:
            scale, shift = get_scale_shift(self.cfg.normalization.joint_friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.dof_frictions[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if getattr(self.cfg.env, "priv_observe_dof_damping", False):
            scale, shift = get_scale_shift(self.cfg.normalization.dof_damping_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.dof_dampings[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_body_height:
            scale, shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, ((self.root_states[: self.num_envs, 2]).view(self.num_envs, -1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_gravity:
            scale, shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - shift) / scale), dim=1)

        if self.cfg.env.priv_observe_body_velocity or self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                lin_vel = self.root_states[: self.num_envs, 7:10]
            else:
                lin_vel = self.base_lin_vel
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                ),
                dim=1,
            )

        if self.cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.clock_inputs), dim=1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.desired_contact_states), dim=1)

        if self.cfg.env.priv_observe_high_freq_goal:
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, self.obj_pose_in_ee.clone(), self.obj_abg_in_ee.clone()),
                dim=1,
            )

        if getattr(self.cfg.env, "priv_observe_arm_mount_tf", False):
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.arm_mount_tfs), dim=1)

        if policy == "arm" and self.cfg.arm.trajectory.enabled:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.get_full_trajectory_privileged_obs()), dim=1)

        return privileged_obs_buf

    def _arm_post_callback_hook(self):
        if global_switch.switch_open:
            # Deferred resample: FK is now valid (after gym.simulate()), sample trajectory
            # from the correct post-reset EE position.
            deferred_ids = self.traj_deferred_resample.nonzero(as_tuple=False).flatten()
            if len(deferred_ids) > 0:
                self._resample_arm_commands(deferred_ids)
                self.traj_deferred_resample[deferred_ids] = False

            # Periodic resample: arm_time_buf is reset to 0 on resample, then incremented
            # by _arm_post_physics_hook, so this triggers exactly once per T_trajs seconds.
            traj_period = torch.clamp((self.T_trajs / self.dt).long(), min=1)
            traj_ids = (self.arm_time_buf % traj_period == 0).nonzero(as_tuple=False).flatten()
            if len(deferred_ids) > 0 and len(traj_ids) > 0:
                is_deferred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                is_deferred[deferred_ids] = True
                traj_ids = traj_ids[~is_deferred[traj_ids]]
            self._resample_arm_commands(traj_ids)

        if self.cfg.domain_rand.randomize_end_effector_force:
            traj_ids = (
                self.force_time_buf % (self.T_force / self.dt).long() == 0
            ).nonzero(as_tuple=False).flatten()
            self.resample_force(traj_ids)

    def _arm_observation_hook(self, obs_buf, roll, pitch, yaw):
        # Vision branch reads from self.obj_pose_in_ee / self.obj_abg_in_ee which need refreshing
        # before compute_observations consumes them.
        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        n_cmd_dims = self.cfg.dog.dog_num_commands if self.cfg.commands.use_dynamic_gait else 3

        if self.cfg.hybrid.use_vision:
            env_ids = (
                (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
            self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :n_cmd_dims],
                    self.obj_obs_pose_in_ee[:] if global_switch.switch_open else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    self.obj_obs_abg_in_ee[:] if global_switch.switch_open else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )
        else:
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :n_cmd_dims],
                    self.commands_arm_obs[:] if global_switch.switch_open else torch.zeros_like(self.commands_arm_obs[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )

        return obs_buf

    def _arm_observation_traj_hook(self, obs_buf):
        if not self.cfg.arm.trajectory.enabled:
            return obs_buf
        contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        remaining_time = torch.clamp(
            self.traj_target_time - self.traj_elapsed_time, min=0.0
        ).unsqueeze(-1)
        return torch.cat(
            (
                obs_buf,
                self.base_pos[:, 2:3],
                contact_states,
                self.get_ee_pose_body_9d(),
                self.get_ee_twist_body(),
                self.get_trajectory_window_obs(),
                remaining_time,
            ),
            dim=-1,
        )

    def _arm_step_end_hook(self):
        self.last_plan_actions[:] = self.plan_actions[:]
        if self.cfg.arm.trajectory.enabled:
            self.prev_ee_twist_body[:] = self.get_ee_twist_body()

    def _arm_privileged_obs_hook(self, privileged_obs_buf):
        return self._get_physics_privileged_observations("dog")

    # ------------------------------------------------------------------
    # Viewer / video overlays
    # ------------------------------------------------------------------

    def _draw_ee_ori_coord(self):
        grasper_offset = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).reshape(1, -1)
        grasper_in_world = self.end_effector_state[0, :3] + quat_rotate(self.end_effector_state[0:1, 3:7], grasper_offset)[0]
        x, y, z = grasper_in_world[0], grasper_in_world[1], grasper_in_world[2]
        ee_quat = self.end_effector_state[0, 3:7]
        self.draw_sphere_and_axes((x, y, z), ee_quat, 0.02, (1, 1, 0))

    def _draw_command_ori_coord(self):
        if self.cfg.arm.trajectory.enabled:
            target = self.traj_pos_world[0, self.traj_progress_idx[0]]
            quat = self.traj_quat_world[0, self.traj_progress_idx[0]]
            self.draw_sphere_and_axes(
                (target[0].item(), target[1].item(), target[2].item()), quat, 0.02, (0, 1, 1)
            )
            return

        x, y, z = self.lpy_to_world_xyz()
        roll = self.visual_rpy[0, -3]
        pitch = self.visual_rpy[0, -2]
        yaw = self.visual_rpy[0, -1]
        quat_base = quat_from_euler_xyz(roll, pitch, yaw)
        quat_world = quat_mul(self.base_quat[0], quat_base)
        self.draw_sphere_and_axes((x, y, z), quat_world, 0.02, (0, 1, 1))

    def _draw_viewer_polyline(self, points, color, env_id=0):
        if points.shape[0] < 2:
            return
        vertices = np.empty((points.shape[0] - 1, 2, 3), dtype=np.float32)
        vertices[:, 0, :] = points[:-1]
        vertices[:, 1, :] = points[1:]
        colors = np.tile(np.array([color], dtype=np.float32), (vertices.shape[0], 1))
        self.gym.add_lines(
            self.viewer, self.envs[env_id], vertices.shape[0], vertices.reshape(-1, 3), colors,
        )

    def _draw_policy_trajectory(self, env_id=0):
        if not self.cfg.arm.trajectory.enabled or self.headless or self.viewer is None:
            return
        points = self.traj_pos_world[env_id].detach().cpu().numpy().astype(np.float32)
        if points.shape[0] < 2:
            return
        stride = max(1, points.shape[0] // 96)
        sampled_points = points[::stride]
        progress = int(self.traj_progress_idx[env_id].item())
        sampled_progress = int(np.clip(progress // stride, 0, sampled_points.shape[0] - 1))

        self._draw_viewer_polyline(sampled_points[: sampled_progress + 1], (1.0, 0.75, 0.0), env_id)
        self._draw_viewer_polyline(sampled_points[sampled_progress:], (0.0, 0.85, 1.0), env_id)

        target = self.traj_pos_world[env_id, self.traj_progress_idx[env_id]]
        target_quat = self.traj_quat_world[env_id, self.traj_progress_idx[env_id]]
        final = self.traj_pos_world[env_id, -1]
        final_quat = self.traj_quat_world[env_id, -1]
        self.draw_sphere_and_axes(
            (target[0].item(), target[1].item(), target[2].item()), target_quat, 0.035, (0.0, 1.0, 1.0), scale=0.12,
        )
        self.draw_sphere_and_axes(
            (final[0].item(), final[1].item(), final[2].item()), final_quat, 0.03, (1.0, 0.0, 1.0), scale=0.1,
        )

        ee_to_target = torch.stack((self.end_effector_state[env_id, :3], target), dim=0).detach().cpu().numpy().astype(np.float32)
        self._draw_viewer_polyline(ee_to_target, (1.0, 0.1, 0.1), env_id)

        lookahead_ids = torch.clamp(
            self.traj_progress_idx[env_id] + self.traj_window_offsets,
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        for waypoint in self.traj_pos_world[env_id, lookahead_ids[::2]]:
            sphere_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(0.2, 0.8, 1.0))
            sphere_pose = gymapi.Transform(
                gymapi.Vec3(waypoint[0].item(), waypoint[1].item(), waypoint[2].item()), r=None,
            )
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

    def _arm_draw_overlay_hook(self):
        self._draw_ee_ori_coord()
        self._draw_command_ori_coord()
        self._draw_policy_trajectory()

    def _policy_command_overlay_panels(self, env_id=0):
        """Return (left_lines, right_lines) for two-panel overlay."""
        def vals(tensor, count):
            return tensor[env_id, : min(count, tensor.shape[1])].detach().cpu().tolist()

        arm_open = global_switch.switch_open
        dog = vals(self.commands_dog, min(11, self.commands_dog.shape[1]))

        # ---- LEFT: commands ----
        left = []
        left.append(f'vx={dog[dog_cmd_idx["x_vel"]]:+.2f}  vy={dog[dog_cmd_idx["y_vel"]]:+.2f}  yaw={dog[dog_cmd_idx["yaw_vel"]]:+.2f}')

        if self.cfg.commands.use_dynamic_gait and self.commands_dog.shape[1] >= 11:
            left.append(f'pitch={dog[dog_cmd_idx["body_pitch"]]:+.2f}  roll={dog[dog_cmd_idx["body_roll"]]:+.2f}  h_cmd={dog[dog_cmd_idx["body_height"]]:+.2f}')
            left.append(f'freq={dog[dog_cmd_idx["gait_frequency"]]:.2f}  swing={dog[dog_cmd_idx["footswing_height"]]:.2f}')
            left.append(f'width={dog[dog_cmd_idx["stance_width"]]:.2f}  len={dog[dog_cmd_idx["stance_length"]]:.2f}  dur={dog[dog_cmd_idx["gait_duration"]]:.2f}')
        elif self.commands_dog.shape[1] >= 6:
            left.append(f'pitch={dog[dog_cmd_idx["body_pitch"]]:+.2f}  roll={dog[dog_cmd_idx["body_roll"]]:+.2f}  h_cmd={dog[dog_cmd_idx["body_height"]]:+.2f}')

        if arm_open:
            if self.cfg.arm.trajectory.enabled:
                user = vals(self.user_vel_cmd, 3)
                extra = vals(self.arm_delta_vel_cmd, 3)
                left.append(f"user vx={user[0]:+.2f} vy={user[1]:+.2f} yaw={user[2]:+.2f}")
                left.append(f"arm dvx={extra[0]:+.2f} dvy={extra[1]:+.2f} dyaw={extra[2]:+.2f}")
            else:
                arm_cmd = vals(self.commands_arm_obs, min(6, self.commands_arm_obs.shape[1]))
                left.append(f"arm l={arm_cmd[0]:+.2f} p={arm_cmd[1]:+.2f} y={arm_cmd[2]:+.2f}")
                if len(arm_cmd) >= 6:
                    left.append(f"    r={arm_cmd[3]:+.2f} p={arm_cmd[4]:+.2f} y={arm_cmd[5]:+.2f}")

        # ---- RIGHT: status ----
        right = []
        base_z = self.root_states[env_id, 2].item()
        right.append(f"z={base_z:.2f}  pitch={self.pitch[env_id].item():+.2f}  roll={self.roll[env_id].item():+.2f}")

        # contact = (self.contact_forces[env_id, self.feet_indices, 2] > 1.0).cpu().tolist()
        # contact_str = "".join("█" if c else "░" for c in contact)
        # right.append(f"feet: {contact_str}  (FL FR RL RR)")

        if arm_open and self.cfg.arm.trajectory.enabled:
            traj_step = int(self.traj_progress_idx[env_id].item())
            t_el = self.traj_elapsed_time[env_id].item()
            t_tgt = self.traj_target_time[env_id].item()
            pos_err = self.traj_final_pos_error[env_id].item()
            rot_err = self.traj_final_rot_error[env_id].item()
            cur_lvl = int(self.traj_curriculum_level[env_id].item())
            max_lvl = self.cfg.arm.trajectory.curriculum_levels - 1
            max_level = max(1, max_lvl)
            difficulty = cur_lvl / max_level
            lo, hi = self.cfg.arm.trajectory.length_range
            cur_len = lo + (hi - lo) * difficulty
            right.append(f"traj {traj_step}/{self.traj_num_waypoints - 1}  t={t_el:.1f}/{t_tgt:.1f}s")
            right.append(f"err pos={pos_err:.3f}  rot={rot_err:.3f}")
            right.append(f"curriculum lv={cur_lvl}/{max_lvl}  len={cur_len:.2f}")

        if arm_open:
            arm_action = vals(
                self.actions[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm],
                self.num_actions_arm,
            )
            right.append("arm: " + " ".join(f"{v:+.2f}" for v in arm_action))

        stage = "hybrid" if arm_open else "stage1"
        right.append(f"[{stage}]")

        return left, right

    @staticmethod
    def _draw_panel(frame, lines, x0, y0, font, font_scale, line_height, alpha=0.55):
        """Draw a semi-transparent panel and text on a RGBA frame."""
        if not lines:
            return
        (tw, th), _ = cv2.getTextSize(
            max(lines, key=len), font, font_scale, 1
        )
        pad = 6
        w = tw + pad * 2
        h = line_height * len(lines) + pad
        x1, y1 = x0 + w, y0 + h

        # alpha blend dark background
        roi = frame[y0:y1, x0:x1]
        bg = np.zeros_like(roi)
        bg[:, :, 3] = 255
        frame[y0:y1, x0:x1] = (roi * (1 - alpha) + bg * alpha).astype(np.uint8)

        for i, text in enumerate(lines):
            y = y0 + pad + i * line_height + line_height - 4
            cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (0, 0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (230, 230, 230, 255), 1, cv2.LINE_AA)

    def _overlay_policy_text(self, frame, env_id=0):
        left_lines, right_lines = self._policy_command_overlay_panels(env_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.32
        line_height = 13
        W = frame.shape[1]

        # left panel: top-left
        self._draw_panel(frame, left_lines, 4, 4, font, font_scale, line_height)

        # right panel: top-right (estimate width from longest line)
        if right_lines:
            (tw, _), _ = cv2.getTextSize(max(right_lines, key=len), font, font_scale, 1)
            x0_right = max(W // 2, W - tw - 20)
            self._draw_panel(frame, right_lines, x0_right, 4, font, font_scale, line_height)

    def _project_world_points_to_camera(self, points_world, env_handle, camera_handle):
        view = np.asarray(self.gym.get_camera_view_matrix(self.sim, env_handle, camera_handle), dtype=np.float32).reshape(4, 4)
        proj = np.asarray(self.gym.get_camera_proj_matrix(self.sim, env_handle, camera_handle), dtype=np.float32).reshape(4, 4)
        points_h = np.concatenate((points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)), axis=1)
        clip = points_h @ view @ proj
        w = clip[:, 3:4]
        valid = np.abs(w[:, 0]) > 1e-5
        ndc = np.zeros((points_world.shape[0], 3), dtype=np.float32)
        ndc[valid] = clip[valid, :3] / w[valid]
        width, height = self.camera_props.width, self.camera_props.height
        pixels = np.empty((points_world.shape[0], 2), dtype=np.int32)
        pixels[:, 0] = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
        pixels[:, 1] = ((1.0 - ndc[:, 1]) * 0.5 * height).astype(np.int32)
        valid &= np.isfinite(ndc).all(axis=1)
        valid &= (pixels[:, 0] >= -width) & (pixels[:, 0] <= 2 * width)
        valid &= (pixels[:, 1] >= -height) & (pixels[:, 1] <= 2 * height)
        return pixels, valid

    def _overlay_policy_trajectory(self, frame, env_id, env_handle, camera_handle):
        if not self.cfg.env.recording_overlay_trajectory or not self.cfg.arm.trajectory.enabled:
            return
        points = self.traj_pos_world[env_id].detach().cpu().numpy().astype(np.float32)
        stride = max(1, points.shape[0] // 96)
        points = points[::stride]
        if points.shape[0] < 2:
            return
        try:
            pixels, valid = self._project_world_points_to_camera(points, env_handle, camera_handle)
        except Exception:
            return
        for i in range(points.shape[0] - 1):
            if valid[i] and valid[i + 1]:
                cv2.line(frame, tuple(pixels[i]), tuple(pixels[i + 1]), (255, 220, 0, 255), 2, cv2.LINE_AA)
        target = self.traj_pos_world[env_id, self.traj_progress_idx[env_id]].detach().cpu().numpy()[None, :].astype(np.float32)
        pixels, valid = self._project_world_points_to_camera(target, env_handle, camera_handle)
        if valid[0]:
            cv2.circle(frame, tuple(pixels[0]), 5, (0, 255, 255, 255), -1, cv2.LINE_AA)

    def _arm_render_overlay_hook(self, frame, env_id, env_handle, camera_handle):
        self._overlay_policy_trajectory(frame, env_id, env_handle, camera_handle)
        if self.cfg.env.recording_overlay_text:
            self._overlay_policy_text(frame, env_id)

    # ============================================================
    # Step bookkeeping that depends on plan_actions
    # ============================================================

    def _map_unit_interval_to_command_range(self, value, limits):
        lo, hi = limits
        half = (hi - lo) / 2.0
        center = (lo + hi) / 2.0
        return torch.clip(center + half * value, lo, hi)

    def _apply_body_attitude_plan(self, scaled_plan):
        self.commands_dog[:, dog_cmd_idx["body_pitch"]] = torch.clip(
            scaled_plan[..., 0],
            self.cfg.commands.limit_body_pitch[0],
            self.cfg.commands.limit_body_pitch[1] / 4 * 3.0,
        )
        self.commands_dog[:, dog_cmd_idx["body_roll"]] = torch.clip(
            scaled_plan[..., 1],
            self.cfg.commands.limit_body_roll[0],
            self.cfg.commands.limit_body_roll[1],
        )

    def _apply_dynamic_gait_plan(self, gait_plan):
        self.commands_dog[:, dog_cmd_idx["gait_frequency"]] = self._map_unit_interval_to_command_range(
            gait_plan[..., 0], self.cfg.commands.limit_gait_frequency
        )
        self.commands_dog[:, dog_cmd_idx["footswing_height"]] = self._map_unit_interval_to_command_range(
            gait_plan[..., 1], self.cfg.commands.limit_footswing_height
        )
        self.commands_dog[:, dog_cmd_idx["stance_width"]] = self._map_unit_interval_to_command_range(
            gait_plan[..., 2], self.cfg.commands.limit_stance_width
        )
        self.commands_dog[:, dog_cmd_idx["stance_length"]] = self._map_unit_interval_to_command_range(
            gait_plan[..., 3], self.cfg.commands.limit_stance_length
        )
        self.commands_dog[:, dog_cmd_idx["gait_duration"]] = self._map_unit_interval_to_command_range(
            gait_plan[..., 4], self.cfg.commands.limit_gait_duration
        )

    def _apply_trajectory_plan(self, obs):
        limits = torch.tensor(
            self.cfg.arm.trajectory.delta_vel_limit,
            dtype=torch.float,
            device=self.device,
        ).view(1, 3)
        delta_vel = torch.clip(obs[..., :3], -1.0, 1.0) * limits
        self.arm_delta_vel_cmd[:] = delta_vel
        self.commands_dog[:, dog_cmd_idx["velocity"]] = self.user_vel_cmd + delta_vel
        self.plan_actions[:, :3] = delta_vel

        if self.cfg.commands.use_dynamic_gait and obs.shape[-1] >= 10:
            gait_obs = obs[..., 3:10]
            scaled_attitude = gait_obs[..., :2] * 0.4
            self._apply_body_attitude_plan(scaled_attitude)
            self._apply_dynamic_gait_plan(gait_obs[..., 2:])
            self.plan_actions[:, 3:10] = torch.cat((scaled_attitude, gait_obs[..., 2:]), dim=-1)

    def _apply_body_plan(self, obs):
        rescaled_obs = obs * 0.4
        self._apply_body_attitude_plan(rescaled_obs)

        if self.cfg.commands.use_dynamic_gait and rescaled_obs.shape[-1] >= 7:
            self._apply_dynamic_gait_plan(obs[..., 2:7])

        if self.cfg.hybrid.plan_vel and not self.cfg.commands.use_dynamic_gait:
            self.commands_dog[:, dog_cmd_idx["x_vel"]] = torch.clip(rescaled_obs[..., 2], -2, 2)  # lin_vel
            self.commands_dog[:, dog_cmd_idx["yaw_vel"]] = torch.clip(rescaled_obs[..., 3], -2, 2)  # ang_vel
        self.plan_actions[:] = rescaled_obs

    def plan(self, obs):
        if self.cfg.arm.trajectory.enabled:
            self._apply_trajectory_plan(obs)
            return

        self._apply_body_plan(obs)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def get_arm_observations(self):

        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        obs_buf = torch.cat(
            (
                (
                    self.dof_pos[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm]
                    - self.default_dof_pos[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm]
                )
                * self.obs_scales.dof_pos,
                # self.dof_vel[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] * self.obs_scales.dof_vel,
                self.actions[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm],
            ),
            dim=-1,
        )

        if self.cfg.arm.trajectory.enabled:
            contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
            remaining_time = torch.clamp(
                self.traj_target_time - self.traj_elapsed_time, min=0.0
            ).unsqueeze(-1)
            obs_buf = torch.cat(
                (
                    obs_buf,
                    self.base_pos[:, 2:3],
                    contact_states,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.commands_dog[:, dog_cmd_idx["velocity"]],
                    (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["gait_params"]]
                    if self.cfg.commands.use_dynamic_gait
                    else torch.empty(self.num_envs, 0, device=self.device),
                    self.get_ee_pose_body_9d(),
                    self.get_ee_twist_body(),
                    self.get_trajectory_window_obs(),
                    remaining_time,
                ),
                dim=-1,
            )
            obs_builder = ObservationBuilder(self, "arm", self.cfg.arm.arm_num_observations)
            obs_builder.add(obs_buf)
            obs_buf = obs_builder.build()
            privileged_obs_buf = self._get_physics_privileged_observations("arm")
            assert privileged_obs_buf.shape[1] == self.cfg.arm.arm_num_privileged_obs, (
                f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
            )
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)
            return obs_buf, privileged_obs_buf

        if self.cfg.hybrid.use_vision:
            env_ids = (
                (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
            self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )
        else:
            idx = self.cfg.arm.arm_num_commands
            obs_buf = torch.cat(
                (
                    obs_buf,
                    self.commands_arm_obs[:, :idx],
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )

        if self.cfg.commands.use_dynamic_gait:
            obs_buf = torch.cat(
                (obs_buf, (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["gait_params"]]), dim=-1
            )

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        privileged_obs_buf = self._get_physics_privileged_observations("arm")

        assert privileged_obs_buf.shape[1] == self.cfg.arm.arm_num_privileged_obs, (
            f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
        )

        # return clipped obs, clipped states (None), rewards, dones and infos
        obs_builder = ObservationBuilder(self, "arm", self.cfg.arm.arm_num_observations)
        obs_builder.add(obs_buf)
        obs_buf = obs_builder.build()
        if privileged_obs_buf is not None:
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)

        return obs_buf, privileged_obs_buf

    def get_dog_observations(self):
        """Computes observations"""
        rpy = quaternion_to_rpy(self.base_quat)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        obs_buf = torch.cat(
            (
                self.projected_gravity,
                (self.dof_pos[:, : self.num_actions_loco] - self.default_dof_pos[:, : self.num_actions_loco])
                * self.obs_scales.dof_pos,
                self.dof_vel[:, : self.num_actions_loco] * self.obs_scales.dof_vel,
                self.actions[:, : self.num_actions_loco],
            ),
            dim=-1,
        )

        if self.cfg.hybrid.use_vision:
            # "obj_obs_pose_in_ee" and "obj_obs_abg_in_ee" are already updated in the "get_arm_observations()"
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :self.cfg.dog.dog_num_commands],
                    (self.obj_obs_pose_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )
        else:
            idx = self.cfg.arm.arm_num_commands
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :self.cfg.dog.dog_num_commands],
                    (self.commands_arm_obs[:, :idx])
                    if global_switch.switch_open
                    else torch.zeros_like(self.commands_arm_obs[:, :idx]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )

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
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            obs_buf = torch.cat((obs_buf, heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            obs_buf = torch.cat(
                (obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(self.num_envs, -1) * 1.0), dim=1
            )

        if self.cfg.arm.trajectory.enabled:
            obs_buf = torch.cat((obs_buf, self.get_ee_pose_body_9d()), dim=-1)

        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        arm_pos = (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice]) * self.obs_scales.dof_pos
        arm_vel = self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel
        obs_buf = torch.cat((obs_buf, arm_pos, arm_vel), dim=-1)

        # add noise if needed
        # if self.add_noise:
        #     obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        privileged_obs_buf = self._get_physics_privileged_observations("dog")

        assert privileged_obs_buf.shape[1] == self.cfg.dog.dog_num_privileged_obs, (
            f"dog num_privileged_obs ({self.cfg.dog.dog_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
        )

        # return clipped obs, clipped states (None), rewards, dones and infos
        obs_builder = ObservationBuilder(self, "dog", self.cfg.dog.dog_num_observations)
        obs_builder.add(obs_buf)
        obs_buf = obs_builder.build()
        if privileged_obs_buf is not None:
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)

        return obs_buf, privileged_obs_buf

    def lpy_to_world_xyz(self):
        # import ipdb; ipdb.set_trace()
        l = self.commands_arm[0, 0]
        p = self.commands_arm[0, 1]
        y = self.commands_arm[0, 2]

        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)

        forward = quat_apply(self.base_quat[0], self.forward_vec[0])
        yaw = torch.atan2(forward[1], forward[0])

        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[0, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[0, 1]
        z_ = torch.mean(z + self.measured_heights) + 0.38
        return x_, y_, z_


class EvaluationWrapper(WBCEnv):
    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        prone=False,
        deploy=False,
        cfg: Cfg = None,
        eval_cfg: Cfg = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
    ):

        super().__init__(
            sim_device,
            headless,
            num_envs=num_envs,
            prone=prone,
            deploy=deploy,
            cfg=cfg,
            eval_cfg=eval_cfg,
            initial_dynamics_dict=initial_dynamics_dict,
            physics_engine=physics_engine,
        )

    def update_arm_commands(self, target_lpy, target_rpy):
        self.commands_arm_obs[:, :3] = target_lpy

        roll = target_rpy[:, 0]
        pitch = target_rpy[:, 1]
        yaw = target_rpy[:, 2]

        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))

        self.obj_quats[:] = quats.reshape(-1, 4)
        assert torch.allclose(
            torch.norm(self.obj_quats[:], dim=1), torch.ones(self.num_envs).to(self.device), atol=1e-5
        ), "quats is not unit vector."

        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        self.visual_rpy[:] = quaternion_to_rpy(self.obj_quats[:]).to(self.device)
        if self.cfg.use_rot6d:
            r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
            self.commands_arm_obs[:, 3:9] = r6d.to(self.device)
        else:
            # use delta angle
            rpy = self.quat_to_angle(self.obj_quats[:]).to(self.device)
            self.commands_arm_obs[:, 3] = rpy[:, 0]
            self.commands_arm_obs[:, 4] = rpy[:, 1]
            self.commands_arm_obs[:, 5] = rpy[:, 2]


class KeyboardWrapper(WBCEnv):
    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_8, "move forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_5, "move backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_4, "move left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_6, "move right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_7, "turn left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_9, "turn right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "arm up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "arm down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "arm forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "arm backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "arm left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "arm right")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "arm pitch down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "arm pitch up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "arm roll left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "arm roll right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "arm yaw left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "arm yaw right")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

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

                # for demo
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

                elif evt.action == "move forward" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["x_vel"]] += 0.1
                elif evt.action == "move backward" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["x_vel"]] -= 0.1
                elif evt.action == "move left" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] += 0.1
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] = torch.clip(self.commands_dog[0, dog_cmd_idx["y_vel"]], -0.5, 0.5)
                elif evt.action == "move right" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] -= 0.1
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] = torch.clip(self.commands_dog[0, dog_cmd_idx["y_vel"]], -0.5, 0.5)
                elif evt.action == "turn left" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["yaw_vel"]] += 0.1
                elif evt.action == "turn right" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["yaw_vel"]] -= 0.1
                elif evt.action == "arm up" and evt.value > 0:
                    self.commands_arm[0, 1] += 0.1
                elif evt.action == "arm down" and evt.value > 0:
                    self.commands_arm[0, 1] -= 0.1
                elif evt.action == "arm forward" and evt.value > 0:
                    self.commands_arm[0, 0] += 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == "arm backward" and evt.value > 0:
                    self.commands_arm[0, 0] -= 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == "arm left" and evt.value > 0:
                    self.commands_arm[0, 2] += 0.1
                elif evt.action == "arm right" and evt.value > 0:
                    self.commands_arm[0, 2] -= 0.1
                elif evt.action == "arm pitch down" and evt.value > 0:
                    self.commands_arm[0, 4] += 0.1
                elif evt.action == "arm pitch up" and evt.value > 0:
                    self.commands_arm[0, 4] -= 0.1
                elif evt.action == "arm roll left" and evt.value > 0:
                    self.commands_arm[0, 3] += 0.1
                elif evt.action == "arm roll right" and evt.value > 0:
                    self.commands_arm[0, 3] -= 0.1
                elif evt.action == "arm yaw left" and evt.value > 0:
                    self.commands_arm[0, 5] += 0.1
                elif evt.action == "arm yaw right" and evt.value > 0:
                    self.commands_arm[0, 5] -= 0.1

                elif evt.action == "reset" and evt.value > 0:
                    self.reset()
                    self.commands_dog[0, dog_cmd_idx["velocity"]] = 0

                elif (
                    evt.action
                    in [
                        "move forward",
                        "move backward",
                        "turn left",
                        "move left",
                        "move right",
                        "turn right",
                        "arm up",
                        "arm down",
                        "arm forward",
                        "arm backward",
                        "arm left",
                        "arm right",
                        "arm pitch down",
                        "arm pitch up",
                        "arm roll left",
                        "arm roll right",
                        "arm yaw left",
                        "arm yaw right",
                    ]
                    and evt.value == 0
                ):
                    print(
                        f'x_vel: {self.commands_dog[0, dog_cmd_idx["x_vel"]]:.2f}, '
                        f'y_vel: {self.commands_dog[0, dog_cmd_idx["y_vel"]]:.2f}, '
                        f'yaw_vel: {self.commands_dog[0, dog_cmd_idx["yaw_vel"]]:.2f}, '
                        f'l: {self.commands_arm[0, 0]:.2f}, '
                        f'p: {self.commands_arm[0, 1]:.2f}, '
                        f'yaw: {self.commands_arm[0, 2]:.2f}, '
                        f'roll: {self.commands_arm[0, 3]:.2f}, '
                        f'pitch: {self.commands_arm[0, 4]:.2f}, '
                        f'yaw: {self.commands_arm[0, 5]:.2f}'
                    )

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

        self.update_arm_commands()

    def update_arm_commands(self):

        self.commands_arm_obs[0:1, 0] = self.commands_arm[0:1, 0]
        self.commands_arm_obs[0:1, 1] = self.commands_arm[0:1, 1]
        self.commands_arm_obs[0:1, 2] = self.commands_arm[0:1, 2]

        roll = self.commands_arm[0:1, 3]
        pitch = self.commands_arm[0:1, 4]
        yaw = self.commands_arm[0:1, 5]

        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        # quats = quat_mul(q3, quat_mul(q2, q1))
        quats = quat_mul(q1, quat_mul(q2, q3))
        # quats = quat_from_euler_xyz(roll, pitch, yaw)
        # print(quats.shape)
        self.obj_quats[0:1] = quats.reshape(-1, 4)

        assert torch.allclose(
            torch.norm(self.obj_quats[0:1], dim=1), torch.ones_like(self.obj_quats[0:1]).to(self.device), atol=1e-5
        ), "quats is not unit vector."

        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        self.visual_rpy[0:1] = quaternion_to_rpy(self.obj_quats[0:1]).to(self.device)
        # self.visual_quats[0:1] = quats.to(self.device)
        rpy = self.quat_to_angle(self.obj_quats[0:1]).to(self.device)
        # self.commands_arm[0:1, 3] = rpy[:, 0]
        # self.commands_arm[0:1, 4] = rpy[:, 1]
        # self.commands_arm[0:1, 5] = rpy[:, 2]

        if self.cfg.use_rot6d:
            r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
            self.commands_arm_obs[0:1, 3:9] = r6d.to(self.device)
        else:
            # use delta angle
            rpy = self.quat_to_angle(self.obj_quats[0:1]).to(self.device)
            self.commands_arm_obs[0:1, 3] = rpy[:, 0]
            self.commands_arm_obs[0:1, 4] = rpy[:, 1]
            self.commands_arm_obs[0:1, 5] = rpy[:, 2]


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env: WBCEnv = env
        cfg: Cfg = self.env.cfg
        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(
            self.env.num_envs, self.num_obs_history, dtype=torch.float, device=self.env.device, requires_grad=False
        )

        self.dog_obs_history = torch.zeros(
            self.env.num_envs,
            cfg.dog.dog_num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.arm_obs_history = torch.zeros(
            self.env.num_envs,
            cfg.arm.arm_num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.arm_fake_actions = torch.zeros(
            self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.env.device, requires_grad=False
        )

    def plan(self, obs):
        return self.env.plan(obs)

    def step(self, action_dog, action_arm):

        if not global_switch.switch_open:
            action_arm = self.arm_fake_actions

        action = torch.concat([action_dog, action_arm], dim=-1)

        rew_dog, rew_arm, done, info = self.env.step(action)

        return rew_dog, rew_arm, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs :], obs), dim=-1)
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def get_dog_observations(self):
        obs, privileged_obs = self.env.get_dog_observations()
        self.dog_obs_history = torch.cat(
            (self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.dog_obs_history}

    def get_dog_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_dog_observations()
        obs[:, 44:50] = pose_in_ee
        self.dog_obs_history = torch.cat(
            (self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.dog_obs_history}

    def get_arm_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_arm_observations()
        obs[:, 12:18] = pose_in_ee
        self.arm_obs_history = torch.cat(
            (self.arm_obs_history[:, self.env.cfg.arm.arm_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.arm_obs_history}

    def get_arm_observations(self):
        obs, privileged_obs = self.env.get_arm_observations()
        self.arm_obs_history = torch.cat(
            (self.arm_obs_history[:, self.env.cfg.arm.arm_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.arm_obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0
        return ret

    def clear_cached(self, env_ids):
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0

    def reset(self):
        ret = super().reset()
        self.obs_history[:, :] = 0
        self.arm_obs_history[:, :] = 0
        self.dog_obs_history[:, :] = 0
        return ret

    def __getattr__(self, name):
        return getattr(self.env, name)


class KeyboardStage1Wrapper(WBCEnv):
    """Keyboard wrapper for stage-1 (dog-only) play.

    Key layout
    ----------
    w / s  — x_vel  +/-
    a / d  — y_vel  +/-
    q / e  — yaw_vel +/-
    j / l  — body_roll +/-
    i / k  — body_pitch +/-
    y / h  — body_height_delta +/-
    r / f  — gait_frequency +/-   (no-op when use_dynamic_gait=False)
    u / o  — stance_width +/-     (no-op when use_dynamic_gait=False)
    SPACE  — reset vel to zero
    """

    _VEL_STEP = 0.1
    _POSE_STEP = 0.05
    _HEIGHT_STEP = 0.05
    _GAIT_FREQ_STEP = 0.5
    _STANCE_STEP = 0.05

    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)

        bindings = [
            (gymapi.KEY_W, "dog_vx_up"),
            (gymapi.KEY_S, "dog_vx_down"),
            (gymapi.KEY_A, "dog_vy_up"),
            (gymapi.KEY_D, "dog_vy_down"),
            (gymapi.KEY_Q, "dog_yaw_up"),
            (gymapi.KEY_E, "dog_yaw_down"),
            (gymapi.KEY_J, "dog_roll_up"),
            (gymapi.KEY_L, "dog_roll_down"),
            (gymapi.KEY_I, "dog_pitch_up"),
            (gymapi.KEY_K, "dog_pitch_down"),
            (gymapi.KEY_Y, "dog_height_up"),
            (gymapi.KEY_H, "dog_height_down"),
            (gymapi.KEY_R, "dog_freq_up"),
            (gymapi.KEY_F, "dog_freq_down"),
            (gymapi.KEY_U, "dog_sw_up"),
            (gymapi.KEY_O, "dog_sw_down"),
            (gymapi.KEY_SPACE, "dog_vel_zero"),
        ]
        for key, action in bindings:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, action)

    def _n_cmd(self):
        return self.commands_dog.shape[1]

    def _add_dog(self, idx, delta, lo, hi):
        if idx >= self._n_cmd():
            return
        val = float(self.commands_dog[0, idx]) + delta
        self.commands_dog[:, idx] = max(lo, min(hi, val))

    def _print_state(self):
        c = self.commands_dog[0]
        n = self._n_cmd()
        parts = [
            f"vx={float(c[0]):+.2f}",
            f"vy={float(c[1]):+.2f}",
            f"yaw={float(c[2]):+.2f}",
        ]
        if n > 5:
            parts += [
                f"pitch={float(c[3]):+.2f}",
                f"roll={float(c[4]):+.2f}",
                f"dh={float(c[5]):+.3f}",
            ]
        if n > 6:
            parts.append(f"freq={float(c[6]):.2f}")
        if n > 8:
            parts.append(f"sw={float(c[8]):.3f}")
        print("  ".join(parts), flush=True)

    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            if self.fixed_cam:
                cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                cam_pos = cam_target + gymapi.Vec3(1, 1, 1)
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            _STAGE1_ACTIONS = {
                "dog_vx_up", "dog_vx_down", "dog_vy_up", "dog_vy_down",
                "dog_yaw_up", "dog_yaw_down", "dog_roll_up", "dog_roll_down",
                "dog_pitch_up", "dog_pitch_down", "dog_height_up", "dog_height_down",
                "dog_freq_up", "dog_freq_down", "dog_sw_up", "dog_sw_down",
                "dog_vel_zero",
            }

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "fixed_cam" and evt.value > 0:
                    self.fixed_cam = not self.fixed_cam

                elif evt.action not in _STAGE1_ACTIONS:
                    continue

                elif evt.value == 0:
                    self._print_state()
                    continue

                # key-down handlers
                elif evt.action == "dog_vx_up":
                    self._add_dog(dog_cmd_idx["x_vel"], self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_vx_down":
                    self._add_dog(dog_cmd_idx["x_vel"], -self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_vy_up":
                    self._add_dog(dog_cmd_idx["y_vel"], self._VEL_STEP, -0.5, 0.5)
                elif evt.action == "dog_vy_down":
                    self._add_dog(dog_cmd_idx["y_vel"], -self._VEL_STEP, -0.5, 0.5)
                elif evt.action == "dog_yaw_up":
                    self._add_dog(dog_cmd_idx["yaw_vel"], self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_yaw_down":
                    self._add_dog(dog_cmd_idx["yaw_vel"], -self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_roll_up":
                    self._add_dog(dog_cmd_idx["body_roll"], self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_roll_down":
                    self._add_dog(dog_cmd_idx["body_roll"], -self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_pitch_up":
                    self._add_dog(dog_cmd_idx["body_pitch"], self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_pitch_down":
                    self._add_dog(dog_cmd_idx["body_pitch"], -self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_height_up":
                    self._add_dog(dog_cmd_idx["body_height"], self._HEIGHT_STEP, -0.3, 0.3)
                elif evt.action == "dog_height_down":
                    self._add_dog(dog_cmd_idx["body_height"], -self._HEIGHT_STEP, -0.3, 0.3)
                elif evt.action == "dog_freq_up":
                    self._add_dog(dog_cmd_idx["gait_frequency"], self._GAIT_FREQ_STEP, 1.0, 4.0)
                elif evt.action == "dog_freq_down":
                    self._add_dog(dog_cmd_idx["gait_frequency"], -self._GAIT_FREQ_STEP, 1.0, 4.0)
                elif evt.action == "dog_sw_up":
                    self._add_dog(dog_cmd_idx["stance_width"], self._STANCE_STEP, 0.2, 0.5)
                elif evt.action == "dog_sw_down":
                    self._add_dog(dog_cmd_idx["stance_width"], -self._STANCE_STEP, 0.2, 0.5)
                elif evt.action == "dog_vel_zero":
                    self.commands_dog[:, dog_cmd_idx["velocity"]] = 0.0
                    self.commands_dog[:, dog_cmd_idx["body_pose"]] = 0.0

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self._draw_viewer_overlays()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self._draw_viewer_overlays()
            self.gym.poll_viewer_events(self.viewer)

        # self.update_arm_commands()

