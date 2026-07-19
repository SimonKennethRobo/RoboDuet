import cv2
import isaacgym

assert isaacgym
import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate, to_torch, torch_rand_float

from go1_gym.envs.config import ConfigNode
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import (
    ee_twist_body_6d,
    get_scale_shift,
    pose_world_to_body_9d,
    quat_conjugate,
    quat_to_angle,
    quat_xyzw_to_rot6d,
)

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .traj_gen.trajectory_geometry import sample_trajectory_commands
from .utils import ObservationBuilder, clip_observation
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
        cfg: ConfigNode = None,
        eval_cfg: ConfigNode = None,
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
            return self._pose_world_to_body_9d(self.end_effector_state[:, :3], self.end_effector_state[:, 3:7])
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

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2))
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
        if self.cfg.wbc.trajectory.user_cmd_mode == "random":
            self.user_vel_cmd[env_ids, 0] = torch_rand_float(
                self.cfg.wbc.trajectory.user_lin_vel_x[0],
                self.cfg.wbc.trajectory.user_lin_vel_x[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
            self.user_vel_cmd[env_ids, 1] = torch_rand_float(
                self.cfg.wbc.trajectory.user_lin_vel_y[0],
                self.cfg.wbc.trajectory.user_lin_vel_y[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
            self.user_vel_cmd[env_ids, 2] = torch_rand_float(
                self.cfg.wbc.trajectory.user_ang_vel_yaw[0],
                self.cfg.wbc.trajectory.user_ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze()
        else:
            self.user_vel_cmd[env_ids] = 0.0

    def _stage2_base_unlock_curriculum_enabled(self):
        return (
            self.cfg.wbc.trajectory.enabled
            and global_switch.switch_open
            and bool(getattr(self.cfg.wbc.trajectory, "stage2_base_unlock_curriculum", False))
        )

    def _stage2_base_unlock_weight(self):
        if not self._stage2_base_unlock_curriculum_enabled():
            return 1.0
        if not self.stage2_base_unlock_started:
            return 0.0
        ramp_iters = max(1, int(getattr(self.cfg.wbc.trajectory, "stage2_base_unlock_ramp_iterations", 1)))
        elapsed = max(0, int(global_switch.count) - int(self.stage2_base_unlock_start_iteration))
        return min(1.0, float(elapsed) / float(ramp_iters))

    def _stage2_base_locked(self):
        return self._stage2_base_unlock_curriculum_enabled() and self._stage2_base_unlock_weight() <= 0.0

    def _stage2_force_point_trajectory(self):
        return (
            self._stage2_base_unlock_curriculum_enabled()
            and bool(getattr(self.cfg.wbc.trajectory, "stage2_base_unlock_force_point_until_unlocked", True))
            and self._stage2_base_unlock_weight() < 1.0
        )

    def _scaled_user_vel_cmd(self):
        return self.user_vel_cmd * self._stage2_base_unlock_weight()


    def _traj_curriculum_params(self, env_ids):
        """Interpolate length, s_curve_amplitude, and orientation range from curriculum level."""
        max_level = max(1, self.cfg.wbc.trajectory.curriculum_levels - 1)
        difficulty = self.traj_curriculum_level[env_ids].float() / max_level  # (n,)

        lo, hi = self.cfg.wbc.trajectory.length_range
        length = lo + (hi - lo) * difficulty

        lo_a, hi_a = self.cfg.wbc.trajectory.s_curve_amplitude_range
        s_amplitude = lo_a + (hi_a - lo_a) * difficulty

        return length, s_amplitude, difficulty

    def _resample_trajectory_commands(self, env_ids):
        self._resample_user_commands(env_ids)
        length, s_amplitude, orientation_scale = self._traj_curriculum_params(env_ids)
        traj_type_override = "point" if self._stage2_force_point_trajectory() else None
        traj_pos, traj_quat, target_time, traj_type = sample_trajectory_commands(
            self.cfg,
            self.end_effector_state,
            self.base_quat,
            env_ids,
            self.traj_num_waypoints,
            self.device,
            length=length,
            s_curve_amplitude=s_amplitude,
            orientation_scale=orientation_scale,
            traj_type_override=traj_type_override,
        )
        self.traj_pos_world[env_ids] = traj_pos
        self.traj_quat_world[env_ids] = traj_quat
        self.traj_target_time[env_ids] = target_time
        self.traj_type[env_ids] = traj_type
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
        self.commands_dog[env_ids, dog_cmd_idx["velocity"]] = self._scaled_user_vel_cmd()[env_ids]
        self._reset_dog_command_smoothing(env_ids)

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
        if self.cfg.wbc.trajectory.enabled:
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
            self.cfg.arm.commands.roll_ee[0],
            self.cfg.arm.commands.roll_ee[1],
            (env_ids.shape[0], 1),
            device=self.device,
        ).squeeze()
        pitch = torch_rand_float(
            self.cfg.arm.commands.pitch_ee[0],
            self.cfg.arm.commands.pitch_ee[1],
            (env_ids.shape[0], 1),
            device=self.device,
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

        if self.cfg.wbc.use_vision:
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
        if not self.cfg.wbc.trajectory.enabled or not global_switch.switch_open:
            return
        self.traj_elapsed_time[:] = self.arm_time_buf.float() * self.dt
        time_progress = self.get_trajectory_time_progress_scalar()
        self.traj_progress_idx[:] = torch.clamp(
            (time_progress * (self.traj_num_waypoints - 1)).long(),
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
            & (self.traj_final_pos_error < self.cfg.wbc.trajectory.completion_pos_threshold)
            & (self.traj_final_rot_error < self.cfg.wbc.trajectory.completion_rot_threshold)
        )
        self.traj_episode_success_buf |= self.traj_complete_buf

    def get_trajectory_window_obs(self):
        waypoint_ids = torch.clamp(
            self.traj_progress_idx[:, None] + self.traj_window_offsets[None, :],
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        return self._trajectory_points_body_9d(waypoint_ids)

    def get_trajectory_time_progress_scalar(self):
        return torch.clamp(self.traj_elapsed_time / torch.clamp(self.traj_target_time, min=self.dt), 0.0, 1.0)

    def get_trajectory_remaining_time_obs(self):
        return torch.clamp(self.traj_target_time - self.traj_elapsed_time, min=0.0).unsqueeze(-1)

    def get_trajectory_completion_time_command(self):
        return self.traj_target_time.unsqueeze(-1)

    def get_trajectory_progress_index_obs(self):
        has_visited = self.traj_visited_mask.any(dim=-1)
        visited_points = torch.where(
            has_visited,
            self.traj_progress_idx.float() + 1.0,
            torch.zeros_like(self.traj_progress_idx, dtype=torch.float),
        )
        progress = torch.clamp(visited_points / max(1, self.traj_num_waypoints), 0.0, 1.0)
        return torch.where(self.traj_type == 3, torch.zeros_like(progress), progress)

    def get_arm_dof_vel_obs(self):
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        return self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel

    def get_arm_policy_action_obs(self):
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        action_parts = [self.actions[:, arm_slice]]
        if self.num_plan_actions > 0:
            action_parts.append(self.plan_actions)
        return torch.cat(action_parts, dim=-1)

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
            self.cfg.wbc.trajectory.pos_error_scale * pos_error + self.cfg.wbc.trajectory.rot_error_scale * rot_error
        )
        error = error * self.traj_visited_mask.float()
        denom = torch.clamp(self.traj_visited_mask.float().sum(dim=-1), min=1.0)
        return torch.sum(error, dim=-1) / denom

    def get_trajectory_current_l2_error(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        target = self._pose_world_to_body_9d(
            self.traj_pos_world[env_ids, self.traj_progress_idx],
            self.traj_quat_world[env_ids, self.traj_progress_idx],
            env_ids,
        )
        ee_pose = self.get_ee_pose_body_9d()
        pos_error = torch.sum(torch.square(ee_pose[:, :3] - target[:, :3]), dim=-1)
        rot_error = torch.sum(torch.square(ee_pose[:, 3:] - target[:, 3:]), dim=-1)
        return self.cfg.wbc.trajectory.pos_error_scale * pos_error + self.cfg.wbc.trajectory.rot_error_scale * rot_error

    def get_trajectory_window_min_l2_error(self):
        waypoint_ids = torch.clamp(
            self.traj_progress_idx[:, None] + self.traj_window_offsets[None, :],
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        flat_env_ids = env_ids[:, None].expand(-1, waypoint_ids.shape[1]).reshape(-1)
        flat_wp_ids = waypoint_ids.reshape(-1)
        targets = self._pose_world_to_body_9d(
            self.traj_pos_world[flat_env_ids, flat_wp_ids],
            self.traj_quat_world[flat_env_ids, flat_wp_ids],
            flat_env_ids,
        ).view(self.num_envs, waypoint_ids.shape[1], 9)
        ee_pose = self.get_ee_pose_body_9d().unsqueeze(1)
        pos_error = torch.sum(torch.square(ee_pose[..., :3] - targets[..., :3]), dim=-1)
        rot_error = torch.sum(torch.square(ee_pose[..., 3:] - targets[..., 3:]), dim=-1)
        error = self.cfg.wbc.trajectory.pos_error_scale * pos_error + self.cfg.wbc.trajectory.rot_error_scale * rot_error
        return torch.min(error, dim=-1).values

    def get_trajectory_tracking_reward(self):
        reward = torch.exp(-self.get_trajectory_error_sum())
        point_mask = self.traj_type == 3
        if torch.any(point_mask):
            reward = reward.clone()
            reward[point_mask] = -self.get_trajectory_window_min_l2_error()[point_mask]
        return reward

    def get_trajectory_current_tracking_reward(self):
        reward = torch.exp(-self.get_trajectory_current_l2_error())
        point_mask = self.traj_type == 3
        if torch.any(point_mask):
            reward = reward.clone()
            reward[point_mask] = -self.get_trajectory_window_min_l2_error()[point_mask]
        return reward

    # ============================================================
    # Arm command helpers (shared by wrappers and external controllers)
    # ============================================================

    def _set_arm_orientation_obs(self, roll, pitch, yaw, env_ids=slice(None)):
        """Convert rpy to quaternion and sync to obj_quats, visual_rpy, commands_arm_obs.

        Called after writing commands_arm position (columns 0-2) and orientation
        (columns 3-5). Handles rot6d / delta-angle encoding internally.
        """
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))

        self.obj_quats[env_ids] = quats.reshape(-1, 4)
        if self.cfg.wbc.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        self.visual_rpy[env_ids] = quaternion_to_rpy(self.obj_quats[env_ids]).to(self.device)
        if self.cfg.use_rot6d:
            r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
            self.commands_arm_obs[env_ids, 3:9] = r6d.to(self.device)
        else:
            rpy = self.quat_to_angle(self.obj_quats[env_ids]).to(self.device)
            self.commands_arm_obs[env_ids, 3] = rpy[:, 0]
            self.commands_arm_obs[env_ids, 4] = rpy[:, 1]
            self.commands_arm_obs[env_ids, 5] = rpy[:, 2]

    def sync_arm_commands_to_obs(self, env_ids=slice(None)):
        """Copy commands_arm position/orientation into obs tensors.

        Reads ``commands_arm`` (written by controller code) and writes
        ``commands_arm_obs``, ``obj_quats``, and ``visual_rpy`` for the
        given env_ids (default: all envs).
        """
        self.commands_arm_obs[env_ids, 0] = self.commands_arm[env_ids, 0]
        self.commands_arm_obs[env_ids, 1] = self.commands_arm[env_ids, 1]
        self.commands_arm_obs[env_ids, 2] = self.commands_arm[env_ids, 2]

        roll = self.commands_arm[env_ids, 3]
        pitch = self.commands_arm[env_ids, 4]
        yaw = self.commands_arm[env_ids, 5]
        self._set_arm_orientation_obs(roll, pitch, yaw, env_ids)

    def format_dog_commands(self) -> str:
        """One-line human-readable summary of the active dog commands."""
        c = self.commands_dog[0]
        n_cmd = self.commands_dog.shape[1]
        parts = [
            f"vx={float(c[0]):+.2f}",
            f"vy={float(c[1]):+.2f}",
            f"wz={float(c[2]):+.2f}",
        ]
        if n_cmd > 5:
            parts += [
                f"pitch={float(c[3]):+.2f}",
                f"roll={float(c[4]):+.2f}",
                f"dh={float(c[5]):+.3f}",
            ]
        if n_cmd > 6:
            parts.append(f"freq={float(c[6]):.2f}")
        if n_cmd > 10:
            parts += [
                f"swing={float(c[7]):.3f}",
                f"sw={float(c[8]):.3f}",
                f"sl={float(c[9]):.3f}",
                f"dur={float(c[10]):.2f}",
            ]
        return "  ".join(parts)

    # ============================================================
    # EE force / arm action curriculum
    # ============================================================

    def resample_force(self, env_ids):
        self.ee_forces[env_ids, self.ee_idx] = torch_rand_float(
            -self.cfg.domain_rand.max_force, self.cfg.domain_rand.max_force, (len(env_ids), 3), device=self.device
        )
        time_range = (self.cfg.commands.T_force_range[1] - self.cfg.commands.T_force_range[0]) / self.dt
        time_interval = torch.randint(
            0,
            int(time_range + 1),
            (len(env_ids),),
            device=self.device,
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
        play_intensity = getattr(self, "stage1_arm_play_intensity", None)
        if play_intensity is not None:
            return min(1.0, max(0.0, float(play_intensity)))
        ramp_iters = max(
            1,
            int(getattr(global_switch, "stage1_arm_ramp_iterations", global_switch.pretrained_to_wbc_start)),
        )
        stage1_iter = getattr(global_switch, "stage1_count", global_switch.count)
        progress = min(1.0, max(0.0, stage1_iter / ramp_iters))
        fixed_fraction = min(1.0, max(0.0, float(self.cfg.env.stage1_arm_fixed_fraction)))
        saturation_fraction = min(
            1.0, max(fixed_fraction, float(getattr(self.cfg.env, "stage1_arm_saturation_fraction", 1.0)))
        )
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
            self.actions[:, arm_slice] = (self.stage1_arm_fixed_dof_pos - arm_default) / self.cfg.control.action_scale
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

    def _resample_stage1_ee_payload(self, env_ids):
        """Sample a per-env EE payload mass for the episode. Scaled by the
        same stage1 curriculum intensity as the rest of the arm disturbance
        (see _get_stage1_arm_curriculum_intensity), so payload weight ramps
        in alongside arm motion/DR rather than jumping in at full strength."""
        stage1_arm_cfg = self.cfg.domain_rand.stage1_arm
        if not (
            self._stage1_arm_curriculum_active()
            and getattr(stage1_arm_cfg, "randomize_ee_payload", False)
        ):
            self.stage1_ee_payload_mass[env_ids] = 0.0
            return
        intensity = self._get_stage1_arm_curriculum_intensity()
        lo, hi = getattr(stage1_arm_cfg, "ee_payload_mass_range", [0.0, 0.0])
        hi = lo + (hi - lo) * intensity
        self.stage1_ee_payload_mass[env_ids] = torch_rand_float(
            lo, hi, (len(env_ids), 1), device=self.device
        ).squeeze(-1)

    def _apply_stage1_ee_payload_force(self):
        """Applies the sampled EE payload as a sustained downward force at
        the EE rigid body's current position, every physics substep (forces
        set via apply_rigid_body_force_at_pos_tensors only last one
        substep). A force at the moving EE reproduces the lever-arm-varying
        base disturbance a real carried payload would cause; IsaacGym only
        allows rigid-body *mass* edits at actor creation, so mass isn't a
        practical way to re-randomize this every episode."""
        stage1_arm_cfg = self.cfg.domain_rand.stage1_arm
        if not (
            self._stage1_arm_curriculum_active()
            and getattr(stage1_arm_cfg, "randomize_ee_payload", False)
        ):
            return
        self.stage1_payload_forces[:, self.ee_idx, 2] = -self.stage1_ee_payload_mass * 9.81
        self.stage1_payload_force_positions[:] = self.rigid_body_state[..., :3].clone().reshape(self.num_envs, -1, 3)
        assert self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.stage1_payload_forces.reshape(-1, 3)),
            gymtorch.unwrap_tensor(self.stage1_payload_force_positions.reshape(-1, 3)),
        ), "Failed to apply stage1 EE payload force."

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

        # Stage-1 simulated EE payload: per-env mass (kg) resampled every
        # episode in _resample_stage1_ee_payload, applied as a sustained
        # downward force at the EE body in _apply_stage1_ee_payload_force.
        self.stage1_ee_payload_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stage1_payload_forces = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.stage1_payload_force_positions = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)

        self.num_plan_actions = self.cfg.arm.num_actions_arm_cd - self.num_actions_arm
        self.last_plan_actions = torch.zeros(
            self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.plan_actions = torch.zeros(
            self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.plan_actions_raw = torch.zeros_like(self.plan_actions)

        self.traj_window_offsets = torch.tensor(
            self.cfg.wbc.trajectory.window_offsets, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.traj_num_waypoints = int(self.cfg.wbc.trajectory.num_waypoints)
        self.traj_pos_world = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_quat_world = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_quat_world[..., 3] = 1.0
        self.traj_progress_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.traj_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.traj_target_time = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_elapsed_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.traj_complete_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.traj_episode_success_buf = torch.zeros_like(self.traj_complete_buf)
        self.traj_final_pos_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_final_rot_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_ee_pose_body_history = torch.zeros(
            self.num_envs, self.traj_num_waypoints, 9, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.traj_target_body_history = torch.zeros_like(self.traj_ee_pose_body_history)
        self.traj_visited_mask = torch.zeros(
            self.num_envs, self.traj_num_waypoints, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.user_vel_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_delta_vel_cmd = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.dog_command_plan_targets = self.commands_dog.clone()
        self.dog_command_plan_smoothed = self.commands_dog.clone()
        self.prev_ee_twist_body = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Per-env arm target position when arm is fixed (intensity=0). Populated at reset
        # with randomized init noise so _keep_arm_fixed holds a varied pose, not always 0.
        arm_default = self.default_dof_pos[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm]
        self.stage1_arm_fixed_dof_pos = arm_default.expand(self.num_envs, -1).clone()

        # Arm rigid-body domain rand buffers are initialized while actor body props
        # are processed in _create_envs(), so per-env link mass/COM randomization is
        # applied before the sim starts and is not rewritten during episode reset.
        if not hasattr(self, "arm_link_mass_scales"):
            arm_body_names = [n for n in self.body_names if "zarx" in n.lower()]
            self.arm_body_indices = [self.body_names.index(n) for n in arm_body_names]
            n_arm_bodies = len(self.arm_body_indices)
            props0 = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
            self.arm_default_link_masses = torch.tensor(
                [props0[i].mass for i in self.arm_body_indices],
                dtype=torch.float,
                device=self.device,
            )
            self.arm_default_link_coms = torch.tensor(
                [[props0[i].com.x, props0[i].com.y, props0[i].com.z] for i in self.arm_body_indices],
                dtype=torch.float,
                device=self.device,
            )
            self.arm_link_mass_scales = torch.ones(self.num_envs, n_arm_bodies, dtype=torch.float, device=self.device)
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
        self.stage2_base_unlock_success_ema = 0.0
        self.stage2_base_unlock_batch_success = 0.0
        self.stage2_base_unlock_started = False
        self.stage2_base_unlock_start_iteration = 0

        # See _dog_obs_layout / get_dog_observations.
        dog_obs_layout = self._dog_obs_layout()
        self.dog_obs_noise_scale_vec = torch.cat(
            [torch.ones(w) * scale for _, w, scale, _ in dog_obs_layout if w > 0]
        ).to(self.device)
        self.dog_obs_droppable_segments = []
        cursor = 0
        for _, w, _, droppable in dog_obs_layout:
            if droppable and w > 0:
                self.dog_obs_droppable_segments.append((cursor, cursor + w))
            cursor += w
        self.dog_last_delivered_obs = torch.zeros(
            self.num_envs, self.cfg.dog.dog_num_observations, dtype=torch.float, device=self.device
        )


    def _arm_pre_step_hook(self):
        self._apply_stage1_arm_curriculum_actions()

    def _arm_decimation_hook(self):
        self.add_continue_force()
        self._apply_stage1_ee_payload_force()

    def _arm_post_sim_hook(self):
        if self.cfg.env.keep_arm_fixed:
            self._keep_arm_fixed()

    def _arm_post_physics_hook(self):
        self.arm_time_buf += 1
        self.force_time_buf += 1
        self.end_effector_state[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]
        self._step_traj_track()
        if self.cfg.wbc.trajectory.enabled and global_switch.switch_open:
            self.prev_ee_twist_body[:] = self.get_ee_twist_body()

    def _arm_check_termination_hook(self):
        if global_switch.switch_open and self.cfg.wbc.rewards.use_terminal_roll:
            reverse_buf1 = torch.logical_and(
                self.roll > self.cfg.wbc.rewards.terminal_body_roll, self.commands_arm[:, 2] > 0.0
            )
            reverse_buf2 = torch.logical_and(
                self.roll < -self.cfg.wbc.rewards.terminal_body_roll, self.commands_arm[:, 2] < 0.0
            )
            self.reverse_buf |= reverse_buf1 | reverse_buf2

        p_align = self.commands_arm[:, 1]
        l_align = self.commands_arm[:, 0]
        self.delta_z = l_align * torch.sin(p_align) + 0.38 - self.base_pos[:, 2]

        if global_switch.switch_open and self.cfg.wbc.rewards.use_terminal_pitch:
            reverse_buf3 = torch.logical_and(
                self.pitch < -self.cfg.wbc.rewards.terminal_body_pitch,
                self.delta_z < -self.cfg.wbc.rewards.headupdown_thres,
            )
            reverse_buf4 = torch.logical_and(
                self.pitch > self.cfg.wbc.rewards.terminal_body_pitch,
                self.delta_z > self.cfg.wbc.rewards.headupdown_thres,
            )
            self.reverse_buf |= reverse_buf3 | reverse_buf4

        if global_switch.switch_open:
            time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.6
            self.reverse_buf = self.reverse_buf & time_exceed_half

    def _arm_reset_hook(self, env_ids):
        if self.cfg.wbc.trajectory.enabled and global_switch.switch_open:
            self._update_traj_curriculum(env_ids)
            # dog command smoothing buffers are reset inside
            # _resample_trajectory_commands; do not call it again here.
        elif not self.cfg.wbc.trajectory.enabled:
            self._resample_arm_commands(env_ids)
        # stage1_arm_target_offset / vel / accel are re-initialised in
        # _arm_post_reset_refresh_hook (after the randomised dof_pos is known).
        self._resample_stage1_ee_payload(env_ids)
        self.dog_last_delivered_obs[env_ids] = 0.0
        self.prev_ee_twist_body[env_ids] = 0.0

    def _update_traj_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return
        max_level = self.cfg.wbc.trajectory.curriculum_levels - 1
        completed = self.traj_episode_success_buf[env_ids]
        self._update_stage2_base_unlock_curriculum(env_ids, completed)
        advance_ids = env_ids[completed]
        if len(advance_ids) > 0:
            self.traj_curriculum_level[advance_ids] = torch.clamp(
                self.traj_curriculum_level[advance_ids] + 1, max=max_level
            )
        self.traj_episode_success_buf[env_ids] = False

    def _update_stage2_base_unlock_curriculum(self, env_ids, completed):
        if not self._stage2_base_unlock_curriculum_enabled():
            self.stage2_base_unlock_batch_success = 1.0
            self.stage2_base_unlock_success_ema = 1.0
            self.stage2_base_unlock_started = True
            return

        point_mask = self.traj_type[env_ids] == 3
        if torch.any(point_mask):
            batch_success = completed[point_mask].float().mean().item()
        else:
            batch_success = completed.float().mean().item()

        alpha = float(getattr(self.cfg.wbc.trajectory, "stage2_base_unlock_success_ema_alpha", 0.05))
        alpha = max(0.0, min(1.0, alpha))
        self.stage2_base_unlock_batch_success = float(batch_success)
        self.stage2_base_unlock_success_ema = (
            (1.0 - alpha) * float(self.stage2_base_unlock_success_ema) + alpha * float(batch_success)
        )

        threshold = float(getattr(self.cfg.wbc.trajectory, "stage2_base_unlock_success_threshold", 0.6))
        if (not self.stage2_base_unlock_started) and self.stage2_base_unlock_success_ema >= threshold:
            self.stage2_base_unlock_started = True
            self.stage2_base_unlock_start_iteration = int(global_switch.count)

    def _ensure_arm_rigid_body_rand_buffers(self, props):
        if hasattr(self, "arm_link_mass_scales"):
            return
        arm_body_names = [n for n in self.body_names if "zarx" in n.lower()]
        self.arm_body_indices = [self.body_names.index(n) for n in arm_body_names]
        n_arm = len(self.arm_body_indices)
        self.arm_default_link_masses = torch.tensor(
            [props[i].mass for i in self.arm_body_indices],
            dtype=torch.float,
            device=self.device,
        )
        self.arm_default_link_coms = torch.tensor(
            [[props[i].com.x, props[i].com.y, props[i].com.z] for i in self.arm_body_indices],
            dtype=torch.float,
            device=self.device,
        )
        self.arm_link_mass_scales = torch.ones(self.num_envs, n_arm, dtype=torch.float, device=self.device)
        self.arm_link_com_offsets = torch.zeros(self.num_envs, n_arm, 3, dtype=torch.float, device=self.device)

    def _sample_arm_rigid_body_props(self, env_id):
        if not self.arm_body_indices:
            return
        arm_dr = self.cfg.domain_rand.stage1_arm if not global_switch.switch_open else self.cfg.domain_rand.stage2_arm
        n_arm = len(self.arm_body_indices)
        if arm_dr.randomize_link_mass:
            lo, hi = arm_dr.link_mass_range
            self.arm_link_mass_scales[env_id] = torch.rand(n_arm, device=self.device) * (hi - lo) + lo
        if arm_dr.randomize_link_com:
            r = arm_dr.link_com_range
            self.arm_link_com_offsets[env_id] = torch.rand(n_arm, 3, device=self.device) * 2 * r - r

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        self._ensure_arm_rigid_body_rand_buffers(props)
        self._sample_arm_rigid_body_props(env_id)

        for k, body_idx in enumerate(self.arm_body_indices):
            props[body_idx].mass = self.arm_default_link_masses[k].item() * self.arm_link_mass_scales[env_id, k].item()
            com = self.arm_default_link_coms[k] + self.arm_link_com_offsets[env_id, k]
            props[body_idx].com = gymapi.Vec3(com[0].item(), com[1].item(), com[2].item())
        return props

    def _arm_post_dof_reset_hook(self, env_ids):
        if len(env_ids) == 0:
            return

        # Add additive noise to arm joint positions before the single reset-time
        # set_dof_state_tensor_indexed() call in _reset_dofs().  Issuing a second
        # DOF write after root reset can desynchronise IsaacGym's articulation cache.
        noise = getattr(self.cfg.env, "stage1_arm_init_dof_pos_noise", 0.0)
        if noise <= 0.0:
            return

        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        self.dof_pos[env_ids, arm_slice] += torch_rand_float(
            -noise,
            noise,
            (len(env_ids), self.num_actions_arm),
            device=self.device,
        )
        self.dof_pos[env_ids] = torch.clamp(
            self.dof_pos[env_ids],
            self.dof_pos_limits[:, 0],
            self.dof_pos_limits[:, 1],
        )

    def _arm_post_reset_refresh_hook(self, env_ids):
        if len(env_ids) == 0:
            return

        # Bookkeeping only.  Do not write DOF/root/rigid-body state here; reset root
        # pose must remain the final sim-state write for GUI and PhysX consistency.
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        self.stage1_arm_fixed_dof_pos[env_ids] = self.dof_pos[env_ids, arm_slice].clone()

        arm_default = self.default_dof_pos[:, arm_slice]  # (1, num_actions_arm)
        self.stage1_arm_target_offset[env_ids] = self.stage1_arm_fixed_dof_pos[env_ids] - arm_default
        self.stage1_arm_target_vel[env_ids] = 0.0
        self.stage1_arm_target_accel[env_ids] = 0.0

        if not self.cfg.wbc.trajectory.enabled:
            return
        # Deferred trajectory resample (FK not valid until next simulate()).
        self.traj_deferred_resample[env_ids] = True

    def _randomize_arm_dof_props(self, env_ids):
        """Override arm DOF slice in Kp/Kd/strength/offset buffers with stage-specific ranges."""
        if len(env_ids) == 0:
            return
        arm_dr = self.cfg.domain_rand.stage1_arm if not global_switch.switch_open else self.cfg.domain_rand.stage2_arm
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
        """Deprecated: arm link mass/COM DR is applied per-env during actor creation."""
        return

    def _arm_resample_commands_train_hook(self, env_ids):
        if self.cfg.wbc.trajectory.enabled and global_switch.switch_open:
            self._resample_user_commands(env_ids)
            target_velocity = self._scaled_user_vel_cmd()[env_ids] + self.arm_delta_vel_cmd[env_ids]
            self.commands_dog[env_ids, dog_cmd_idx["velocity"]] = target_velocity
            self.dog_command_plan_targets[env_ids, dog_cmd_idx["velocity"]] = target_velocity
            self.dog_command_plan_smoothed[env_ids, dog_cmd_idx["velocity"]] = target_velocity

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

        if getattr(self.cfg.env, "priv_observe_stage1_ee_payload_mass", False):
            scale, shift = get_scale_shift(self.cfg.normalization.stage1_ee_payload_mass_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.stage1_ee_payload_mass.unsqueeze(1) - shift) * scale),
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

        if policy == "dog":
            arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
            arm_dof_pos = (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice]) * self.obs_scales.dof_pos
            arm_dof_vel = self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel
            privileged_obs_buf = torch.cat((privileged_obs_buf, arm_dof_pos, arm_dof_vel), dim=1)

        if policy == "arm" and self.cfg.wbc.trajectory.enabled:
            foot_contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, foot_contact_states, self.get_full_trajectory_privileged_obs()), dim=1
            )

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
            traj_ids = (self.force_time_buf % (self.T_force / self.dt).long() == 0).nonzero(as_tuple=False).flatten()
            self.resample_force(traj_ids)

    def _arm_observation_hook(self, obs_buf, roll, pitch, yaw):
        # Vision branch reads from self.obj_pose_in_ee / self.obj_abg_in_ee which need refreshing
        # before compute_observations consumes them.
        if self.cfg.wbc.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        n_cmd_dims = self.cfg.dog.dog_num_commands if self.cfg.commands.use_dynamic_gait else 3

        if self.cfg.wbc.use_vision:
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
                    self.obj_obs_pose_in_ee[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    self.obj_obs_abg_in_ee[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
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
                    self.commands_arm_obs[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.commands_arm_obs[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )

        return obs_buf

    def _arm_observation_traj_hook(self, obs_buf):
        if not self.cfg.wbc.trajectory.enabled:
            return obs_buf
        progress = self.get_trajectory_progress_index_obs().unsqueeze(-1)
        return torch.cat(
            (
                obs_buf,
                self.get_arm_dof_vel_obs(),
                self.base_pos[:, 2:3],
                self.get_ee_pose_body_9d(),
                self.get_ee_twist_body(),
                self.get_trajectory_window_obs(),
                progress,
            ),
            dim=-1,
        )

    def _arm_step_end_hook(self):
        self.last_plan_actions[:] = self.plan_actions[:]
        if self.cfg.env.arm_policy_enabled and self.cfg.wbc.trajectory.enabled:
            self.prev_ee_twist_body[:] = self.get_ee_twist_body()

    def _arm_init_performance_metrics_hook(self):
        for name in ("ee_position_sq_error", "ee_orientation_sq_error", "ee_tracking_samples"):
            self.performance_metric_sums[name] = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

    def _arm_update_performance_metrics_hook(self):
        if not self.cfg.wbc.trajectory.enabled or not global_switch.switch_open:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self.traj_pos_world[env_ids, self.traj_progress_idx]
        target_quat = self.traj_quat_world[env_ids, self.traj_progress_idx]
        actual_pos = self.end_effector_state[:, :3]
        actual_quat = self.end_effector_state[:, 3:7]
        actual_quat = actual_quat / torch.clamp(torch.norm(actual_quat, dim=-1, keepdim=True), min=1e-6)
        target_quat = target_quat / torch.clamp(torch.norm(target_quat, dim=-1, keepdim=True), min=1e-6)
        quat_dot = torch.abs(torch.sum(actual_quat * target_quat, dim=-1))
        orientation_error = 2.0 * torch.acos(torch.clamp(quat_dot, 0.0, 1.0))

        sums = self.performance_metric_sums
        sums["ee_position_sq_error"] += torch.sum(torch.square(actual_pos - target_pos), dim=-1)
        sums["ee_orientation_sq_error"] += torch.square(orientation_error)
        sums["ee_tracking_samples"] += 1.0

    def _arm_log_performance_metrics_hook(self, train_env_ids, episode_steps):
        samples = self.performance_metric_sums["ee_tracking_samples"][train_env_ids]
        valid = samples > 0
        if not torch.any(valid):
            return

        position_rmse = torch.sqrt(
            self.performance_metric_sums["ee_position_sq_error"][train_env_ids]
            / torch.clamp(samples, min=1.0)
        )
        orientation_rmse = torch.sqrt(
            self.performance_metric_sums["ee_orientation_sq_error"][train_env_ids]
            / torch.clamp(samples, min=1.0)
        )
        extras = self.extras["train/episode"]
        extras["perf_ee_position_rmse_m"] = self._mean_valid_metric(position_rmse, valid)
        extras["perf_ee_orientation_rmse_rad"] = self._mean_valid_metric(orientation_rmse, valid)

    def _arm_privileged_obs_hook(self, privileged_obs_buf):
        return self._get_physics_privileged_observations("dog")

    # ------------------------------------------------------------------
    # Viewer / video overlays
    # ------------------------------------------------------------------

    def _draw_ee_ori_coord(self):
        grasper_offset = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).reshape(1, -1)
        grasper_in_world = (
            self.end_effector_state[0, :3] + quat_rotate(self.end_effector_state[0:1, 3:7], grasper_offset)[0]
        )
        x, y, z = grasper_in_world[0], grasper_in_world[1], grasper_in_world[2]
        ee_quat = self.end_effector_state[0, 3:7]
        self.draw_sphere_and_axes((x, y, z), ee_quat, 0.02, (1, 1, 0))

    def _draw_command_ori_coord(self):
        if self.cfg.wbc.trajectory.enabled:
            target = self.traj_pos_world[0, self.traj_progress_idx[0]]
            quat = self.traj_quat_world[0, self.traj_progress_idx[0]]
            self.draw_sphere_and_axes((target[0].item(), target[1].item(), target[2].item()), quat, 0.02, (0, 1, 1))
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
            self.viewer,
            self.envs[env_id],
            vertices.shape[0],
            vertices.reshape(-1, 3),
            colors,
        )

    def _draw_policy_trajectory(self, env_id=0):
        if not self.cfg.wbc.trajectory.enabled or self.headless or self.viewer is None:
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
            (target[0].item(), target[1].item(), target[2].item()),
            target_quat,
            0.035,
            (0.0, 1.0, 1.0),
            scale=0.12,
        )
        self.draw_sphere_and_axes(
            (final[0].item(), final[1].item(), final[2].item()),
            final_quat,
            0.03,
            (1.0, 0.0, 1.0),
            scale=0.1,
        )

        ee_to_target = (
            torch.stack((self.end_effector_state[env_id, :3], target), dim=0).detach().cpu().numpy().astype(np.float32)
        )
        self._draw_viewer_polyline(ee_to_target, (1.0, 0.1, 0.1), env_id)

        lookahead_ids = torch.clamp(
            self.traj_progress_idx[env_id] + self.traj_window_offsets,
            min=0,
            max=self.traj_num_waypoints - 1,
        )
        for waypoint in self.traj_pos_world[env_id, lookahead_ids[::2]]:
            sphere_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(0.2, 0.8, 1.0))
            sphere_pose = gymapi.Transform(
                gymapi.Vec3(waypoint[0].item(), waypoint[1].item(), waypoint[2].item()),
                r=None,
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
        left.append(
            f"vx={dog[dog_cmd_idx['x_vel']]:+.2f}  vy={dog[dog_cmd_idx['y_vel']]:+.2f}  yaw={dog[dog_cmd_idx['yaw_vel']]:+.2f}"
        )

        if self.cfg.commands.use_dynamic_gait and self.commands_dog.shape[1] >= 11:
            left.append(
                f"pitch={dog[dog_cmd_idx['body_pitch']]:+.2f}  roll={dog[dog_cmd_idx['body_roll']]:+.2f}  h_cmd={dog[dog_cmd_idx['body_height']]:+.2f}"
            )
            left.append(
                f"freq={dog[dog_cmd_idx['gait_frequency']]:.2f}  swing={dog[dog_cmd_idx['footswing_height']]:.2f}"
            )
            left.append(
                f"width={dog[dog_cmd_idx['stance_width']]:.2f}  len={dog[dog_cmd_idx['stance_length']]:.2f}  dur={dog[dog_cmd_idx['gait_duration']]:.2f}"
            )
        elif self.commands_dog.shape[1] >= 6:
            left.append(
                f"pitch={dog[dog_cmd_idx['body_pitch']]:+.2f}  roll={dog[dog_cmd_idx['body_roll']]:+.2f}  h_cmd={dog[dog_cmd_idx['body_height']]:+.2f}"
            )

        if arm_open:
            if self.cfg.wbc.trajectory.enabled:
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

        if arm_open and self.cfg.wbc.trajectory.enabled:
            traj_step = int(self.traj_progress_idx[env_id].item())
            t_el = self.traj_elapsed_time[env_id].item()
            t_tgt = self.traj_target_time[env_id].item()
            pos_err = self.traj_final_pos_error[env_id].item()
            rot_err = self.traj_final_rot_error[env_id].item()
            cur_lvl = int(self.traj_curriculum_level[env_id].item())
            max_lvl = self.cfg.wbc.trajectory.curriculum_levels - 1
            max_level = max(1, max_lvl)
            difficulty = cur_lvl / max_level
            lo, hi = self.cfg.wbc.trajectory.length_range
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

        stage = "wbc" if arm_open else "stage1"
        right.append(f"[{stage}]")

        return left, right

    @staticmethod
    def _draw_panel(frame, lines, x0, y0, font, font_scale, line_height, alpha=0.55):
        """Draw a semi-transparent panel and text on a RGBA frame."""
        if not lines:
            return
        (tw, th), _ = cv2.getTextSize(max(lines, key=len), font, font_scale, 1)
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
        view = np.asarray(
            self.gym.get_camera_view_matrix(self.sim, env_handle, camera_handle), dtype=np.float32
        ).reshape(4, 4)
        proj = np.asarray(
            self.gym.get_camera_proj_matrix(self.sim, env_handle, camera_handle), dtype=np.float32
        ).reshape(4, 4)
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
        if not self.cfg.env.recording_overlay_trajectory or not self.cfg.wbc.trajectory.enabled:
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
        target = (
            self.traj_pos_world[env_id, self.traj_progress_idx[env_id]]
            .detach()
            .cpu()
            .numpy()[None, :]
            .astype(np.float32)
        )
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

    def _clip_to_command_limit(self, value, limits):
        return torch.clip(value, limits[0], limits[1])

    def _smooth_dog_command_values(self, indices, values):
        if isinstance(indices, slice):
            indices = list(range(indices.start or 0, indices.stop, indices.step or 1))
        alpha = float(getattr(self.cfg.wbc.trajectory, "dog_command_smoothing_alpha", 1.0))
        alpha = max(0.0, min(1.0, alpha))
        prev = self.dog_command_plan_smoothed[:, indices]
        smoothed = prev + alpha * (values - prev)
        self.dog_command_plan_targets[:, indices] = values
        self.dog_command_plan_smoothed[:, indices] = smoothed
        self.commands_dog[:, indices] = smoothed

    def _set_fixed_dog_command_value(self, index, value):
        self.dog_command_plan_targets[:, index] = value
        self.dog_command_plan_smoothed[:, index] = value
        self.commands_dog[:, index] = value

    def _reset_dog_command_smoothing(self, env_ids):
        if len(env_ids) == 0:
            return

        def _midpoint(limits):
            return 0.5 * (float(limits[0]) + float(limits[1]))

        # Neutralize stale body_pose/gait carried over from previous episode so
        # the first plan() after reset is not EMA-blended with the prior policy
        # output. Velocity is set by the caller (resample) and preserved.
        neutral_assignments = []
        if self.commands_dog.shape[1] > dog_cmd_idx["body_pitch"]:
            neutral_assignments.append((dog_cmd_idx["body_pitch"], 0.0))
        if self.commands_dog.shape[1] > dog_cmd_idx["body_roll"]:
            neutral_assignments.append((dog_cmd_idx["body_roll"], 0.0))
        if self.commands_dog.shape[1] > dog_cmd_idx["body_height"]:
            neutral_assignments.append((dog_cmd_idx["body_height"], 0.0))
        if self.cfg.commands.use_dynamic_gait:
            if self.commands_dog.shape[1] > dog_cmd_idx["gait_frequency"]:
                neutral_assignments.append(
                    (dog_cmd_idx["gait_frequency"], _midpoint(self.cfg.commands.limit_gait_frequency))
                )
            if self.commands_dog.shape[1] > dog_cmd_idx["stance_width"]:
                neutral_assignments.append(
                    (dog_cmd_idx["stance_width"], _midpoint(self.cfg.commands.limit_stance_width))
                )
            if self.commands_dog.shape[1] > dog_cmd_idx["stance_length"]:
                neutral_assignments.append(
                    (dog_cmd_idx["stance_length"], _midpoint(self.cfg.commands.limit_stance_length))
                )
        for idx, value in neutral_assignments:
            self.commands_dog[env_ids, idx] = value

        if self.commands_dog.shape[1] > dog_cmd_idx["footswing_height"]:
            self.commands_dog[env_ids, dog_cmd_idx["footswing_height"]] = 0.06
        if self.commands_dog.shape[1] > dog_cmd_idx["gait_duration"]:
            self.commands_dog[env_ids, dog_cmd_idx["gait_duration"]] = 0.49

        self.dog_command_plan_targets[env_ids] = self.commands_dog[env_ids]
        self.dog_command_plan_smoothed[env_ids] = self.commands_dog[env_ids]

    def _apply_body_pose_plan(self, scaled_plan):
        values = [
            self._clip_to_command_limit(scaled_plan[..., 0], self.cfg.commands.limit_body_pitch),
            self._clip_to_command_limit(scaled_plan[..., 1], self.cfg.commands.limit_body_roll),
        ]
        indices = [dog_cmd_idx["body_pitch"], dog_cmd_idx["body_roll"]]
        if scaled_plan.shape[-1] >= 3:
            values.append(self._clip_to_command_limit(scaled_plan[..., 2], self.cfg.commands.limit_body_height))
            indices.append(dog_cmd_idx["body_height"])
        self._smooth_dog_command_values(indices, torch.stack(values, dim=-1))

    def _apply_body_attitude_plan(self, scaled_plan):
        self._apply_body_pose_plan(scaled_plan[..., :2])

    def _apply_dynamic_gait_plan(self, gait_plan):
        values = torch.stack(
            (
                self._map_unit_interval_to_command_range(gait_plan[..., 0], self.cfg.commands.limit_gait_frequency),
                self._map_unit_interval_to_command_range(gait_plan[..., 1], self.cfg.commands.limit_stance_width),
                self._map_unit_interval_to_command_range(gait_plan[..., 2], self.cfg.commands.limit_stance_length),
            ),
            dim=-1,
        )
        self._smooth_dog_command_values(
            [dog_cmd_idx["gait_frequency"], dog_cmd_idx["stance_width"], dog_cmd_idx["stance_length"]], values
        )
        self._set_fixed_dog_command_value(dog_cmd_idx["footswing_height"], 0.06)
        self._set_fixed_dog_command_value(dog_cmd_idx["gait_duration"], 0.49)

    def _apply_locked_base_plan(self):
        zero = torch.zeros(self.num_envs, device=self.device)
        for idx in (dog_cmd_idx["x_vel"], dog_cmd_idx["y_vel"], dog_cmd_idx["yaw_vel"]):
            self._set_fixed_dog_command_value(idx, zero)

        if self.commands_dog.shape[1] > dog_cmd_idx["body_pitch"]:
            self._set_fixed_dog_command_value(dog_cmd_idx["body_pitch"], zero)
        if self.commands_dog.shape[1] > dog_cmd_idx["body_roll"]:
            self._set_fixed_dog_command_value(dog_cmd_idx["body_roll"], zero)
        if self.commands_dog.shape[1] > dog_cmd_idx["body_height"]:
            self._set_fixed_dog_command_value(dog_cmd_idx["body_height"], zero)

        if self.cfg.commands.use_dynamic_gait:
            def _midpoint(limits):
                return 0.5 * (float(limits[0]) + float(limits[1]))

            self._set_fixed_dog_command_value(
                dog_cmd_idx["gait_frequency"],
                torch.full_like(zero, _midpoint(self.cfg.commands.limit_gait_frequency)),
            )
            self._set_fixed_dog_command_value(
                dog_cmd_idx["stance_width"],
                torch.full_like(zero, _midpoint(self.cfg.commands.limit_stance_width)),
            )
            self._set_fixed_dog_command_value(
                dog_cmd_idx["stance_length"],
                torch.full_like(zero, _midpoint(self.cfg.commands.limit_stance_length)),
            )
            self._set_fixed_dog_command_value(dog_cmd_idx["footswing_height"], torch.full_like(zero, 0.06))
            self._set_fixed_dog_command_value(dog_cmd_idx["gait_duration"], torch.full_like(zero, 0.49))

        self.arm_delta_vel_cmd.zero_()
        self.plan_actions.zero_()

    def _apply_trajectory_plan(self, obs):
        unlock_weight = self._stage2_base_unlock_weight()
        limits = torch.tensor(
            self.cfg.wbc.trajectory.delta_vel_limit,
            dtype=torch.float,
            device=self.device,
        ).view(1, 3)
        self.plan_actions_raw[:, :3] = obs[..., :3]
        if unlock_weight <= 0.0:
            if self.cfg.commands.use_dynamic_gait and obs.shape[-1] >= 9:
                self.plan_actions_raw[:, 3:9] = obs[..., 3:9]
            self._apply_locked_base_plan()
            return

        delta_vel = torch.clip(obs[..., :3], -1.0, 1.0) * limits * unlock_weight
        scaled_user_cmd = self._scaled_user_vel_cmd()
        target_velocity = scaled_user_cmd + delta_vel
        target_velocity = torch.stack(
            (
                self._clip_to_command_limit(target_velocity[..., 0], self.cfg.commands.limit_vel_x),
                self._clip_to_command_limit(target_velocity[..., 1], self.cfg.commands.limit_vel_y),
                self._clip_to_command_limit(target_velocity[..., 2], self.cfg.commands.limit_vel_yaw),
            ),
            dim=-1,
        )
        self.arm_delta_vel_cmd[:] = target_velocity - scaled_user_cmd
        self._smooth_dog_command_values(
            [dog_cmd_idx["x_vel"], dog_cmd_idx["y_vel"], dog_cmd_idx["yaw_vel"]], target_velocity
        )
        self.plan_actions[:, :3] = self.arm_delta_vel_cmd

        if self.cfg.commands.use_dynamic_gait and obs.shape[-1] >= 9:
            self.plan_actions_raw[:, 3:9] = obs[..., 3:9]
            body_pose_plan = obs[..., 3:6] * 0.4 * unlock_weight
            gait_plan = obs[..., 6:9] * unlock_weight
            self._apply_body_pose_plan(body_pose_plan)
            self._apply_dynamic_gait_plan(gait_plan)
            self.plan_actions[:, 3:9] = torch.cat((body_pose_plan, gait_plan), dim=-1)

    def _apply_body_plan(self, obs):
        self.plan_actions_raw[:] = obs[..., : self.num_plan_actions]
        rescaled_obs = obs * 0.4
        if self.cfg.commands.use_dynamic_gait:
            self._apply_body_pose_plan(rescaled_obs[..., :3])
            if obs.shape[-1] >= 6:
                self._apply_dynamic_gait_plan(obs[..., 3:6])
        else:
            self._apply_body_attitude_plan(rescaled_obs)

        if self.cfg.wbc.plan_vel and not self.cfg.commands.use_dynamic_gait:
            self.commands_dog[:, dog_cmd_idx["x_vel"]] = torch.clip(rescaled_obs[..., 2], -2, 2)  # lin_vel
            self.commands_dog[:, dog_cmd_idx["yaw_vel"]] = torch.clip(rescaled_obs[..., 3], -2, 2)  # ang_vel
        self.plan_actions[:] = rescaled_obs

    def plan(self, obs):
        if self.cfg.wbc.trajectory.enabled:
            self._apply_trajectory_plan(obs)
            return

        self._apply_body_plan(obs)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def _arm_dog_state_obs_terms(self):
        """Cross-policy channel gated by cfg.env.arm_observe_dog_state: foot
        contact state and the leg policy's (v_actual - v_cmd) tracking
        residual, so the arm/upper policy has some sense of how well the
        legs are keeping up. Dims must match core.arm_obs_dim_parts'
        dog_contact_states / dog_vel_residual entries."""
        if not getattr(self.cfg.env, "arm_observe_dog_state", False):
            return None, None
        contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        vel_residual = torch.cat(
            (
                self.base_lin_vel[:, :2] - self.commands_dog[:, :2],
                self.base_ang_vel[:, 2:3] - self.commands_dog[:, 2:3],
            ),
            dim=-1,
        )
        return contact_states, vel_residual

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
                self.get_arm_dof_vel_obs(),
                self.get_arm_policy_action_obs(),
            ),
            dim=-1,
        )

        if self.cfg.wbc.trajectory.enabled:
            progress = self.get_trajectory_progress_index_obs().unsqueeze(-1)
            obs_buf = torch.cat(
                (
                    obs_buf,
                    self.base_pos[:, 2:3],
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.commands_dog[:, dog_cmd_idx["velocity"]],
                    (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["body_pose"]]
                    if self.cfg.commands.use_dynamic_gait
                    else torch.empty(self.num_envs, 0, device=self.device),
                    (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["gait_params"]]
                    if self.cfg.commands.use_dynamic_gait
                    else torch.empty(self.num_envs, 0, device=self.device),
                    self.get_trajectory_completion_time_command(),
                    self.get_ee_pose_body_9d(),
                    self.get_ee_twist_body(),
                    self.get_trajectory_window_obs(),
                    progress,
                ),
                dim=-1,
            )
            obs_builder = ObservationBuilder(self, "arm", self.cfg.arm.arm_num_observations)
            obs_builder.add(obs_buf, *self._arm_dog_state_obs_terms())
            obs_buf = obs_builder.build()
            privileged_obs_buf = self._get_physics_privileged_observations("arm")
            assert privileged_obs_buf.shape[1] == self.cfg.arm.arm_num_privileged_obs, (
                f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
            )
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)
            return obs_buf, privileged_obs_buf

        if self.cfg.wbc.use_vision:
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
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["body_pose"]],
                    (self.commands_dog * self.commands_scale_dog)[:, dog_cmd_idx["gait_params"]],
                ),
                dim=-1,
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
        obs_builder.add(obs_buf, *self._arm_dog_state_obs_terms())
        obs_buf = obs_builder.build()
        if privileged_obs_buf is not None:
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)

        return obs_buf, privileged_obs_buf

    def _dog_obs_layout(self):
        """Ordered (name, width, noise_scale, droppable) description of
        get_dog_observations()'s actor-facing segments. Single source of
        truth for both the per-element noise vector and the per-segment
        frame-drop offsets built once in _arm_init_buffers_hook -- avoids
        keeping two hand-mirrored copies in sync. [NOTE] must still be
        updated by hand if get_dog_observations()'s layout changes, same
        caveat as _get_noise_scale_vec.

        droppable=True marks an actual sensor reading (IMU, encoders,
        contacts, ...) that can independently simulate a dropped frame.
        Commanded/internal segments (dog_commands, our own last action,
        gait clock, ...) are never dropped -- we always know those exactly,
        there's no sensor to fail."""
        cfg = self.cfg
        ns = cfg.noise_scales
        level = cfg.noise.noise_level
        s = self.obs_scales

        layout = [
            ("projected_gravity", 3, ns.gravity * level, True),
            ("dog_dof_pos", self.num_actions_loco, ns.dof_pos * level * s.dof_pos, True),
            ("dog_dof_vel", self.num_actions_loco, ns.dof_vel * level * s.dof_vel, True),
            ("dog_actions", self.num_actions_loco, 0.0, False),
        ]
        if cfg.wbc.use_vision:
            layout += [
                ("dog_commands", cfg.dog.dog_num_commands, 0.0, False),
                ("obj_pose_in_ee", 3, 0.0, False),
                ("obj_abg_in_ee", 3, 0.0, False),
            ]
        else:
            layout += [
                ("dog_commands", cfg.dog.dog_num_commands, 0.0, False),
                ("arm_commands", cfg.arm.arm_num_commands, 0.0, False),
            ]
        if cfg.env.observe_two_prev_actions:
            layout.append(("two_prev_actions", self.num_actions_loco, 0.0, False))
        if cfg.env.observe_timing_parameter:
            layout.append(("timing_parameter", 1, 0.0, False))
        if cfg.env.observe_clock_inputs:
            layout.append(("clock_inputs", 4, 0.0, False))

        layout.append(("base_ang_vel", 3, ns.ang_vel * level * s.ang_vel, True))
        lin_vel_scale = ns.lin_vel * level * s.lin_vel if cfg.dog.observe_lin_vel else 0.0
        layout.append(("base_lin_vel", 3, lin_vel_scale, True))
        # Fixed tracking slot. Pose actual and tracking errors are independently
        # zero-filled by their dog-policy switches. Only the measured actual
        # pose gets sensor noise; errors receive no independent noise.
        pose_actual_noise = (
            torch.tensor(
                [
                    ns.gravity * level * s.body_height_cmd,
                    ns.gravity * level * s.body_pitch_cmd,
                    ns.gravity * level * s.body_roll_cmd,
                ]
            )
            if cfg.dog.observe_pose_actual
            else 0.0
        )
        layout.append(("body_pose_actual", 3, pose_actual_noise, False))
        layout.append(("body_pose_error", 3, 0.0, False))
        layout.append(("velocity_error", 3, 0.0, False))

        if cfg.env.observe_yaw:
            layout.append(("heading", 1, 0.0, False))
        if cfg.env.observe_contact_states:
            layout.append(("contact_states", 4, ns.contact_states * level, True))
        if cfg.wbc.trajectory.enabled:
            layout.append(("ee_pose_body", 9, 0.0, True))  # no dedicated noise scale yet

        layout.append(("arm_dof_pos", self.num_actions_arm, ns.dof_pos * level * s.dof_pos, True))
        layout.append(("arm_dof_vel", self.num_actions_arm, ns.dof_vel * level * s.dof_vel, True))
        return layout

    def get_dog_observations(self):
        """Computes observations"""
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

        if self.cfg.wbc.use_vision:
            # "obj_obs_pose_in_ee" and "obj_obs_abg_in_ee" are already updated in the "get_arm_observations()"
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, : self.cfg.dog.dog_num_commands],
                    (self.obj_obs_pose_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                ),
                dim=-1,
            )
        else:
            idx = self.cfg.arm.arm_num_commands
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, : self.cfg.dog.dog_num_commands],
                    (self.commands_arm_obs[:, :idx])  # l,p,y,rot6d
                    if global_switch.switch_open
                    else torch.zeros_like(self.commands_arm_obs[:, :idx]),
                ),
                dim=-1,
            )

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)

        # Fixed width regardless of dog.observe_lin_vel: ang_vel is always
        # real; lin_vel's slot always exists but is zeros when the switch
        # is off, so toggling it never changes dog_num_observations.
        if self.cfg.dog.observe_lin_vel:
            lin_vel_term = (
                self.root_states[: self.num_envs, 7:10] * self.obs_scales.lin_vel
                if self.cfg.commands.global_reference
                else self.base_lin_vel * self.obs_scales.lin_vel
            )
        else:
            lin_vel_term = torch.zeros(self.num_envs, 3, device=self.device)
        obs_buf = torch.cat((obs_buf, self.base_ang_vel * self.obs_scales.ang_vel, lin_vel_term), dim=-1)

        # Fixed 9-wide tracking slot: pose actual is controlled independently
        # from pose/velocity errors. Errors are command - actual.
        if self.cfg.dog.observe_pose_actual or self.cfg.dog.observe_track_error:
            height_actual = self.base_pos[:, 2]
            height_target = float(self.cfg.rewards.base_height_target) + self.commands_dog[:, dog_cmd_idx["body_height"]]
        if self.cfg.dog.observe_pose_actual:
            pose_actual = torch.stack(
                (
                    height_actual * self.obs_scales.body_height_cmd,
                    self.pitch * self.obs_scales.body_pitch_cmd,
                    self.roll * self.obs_scales.body_roll_cmd,
                ),
                dim=-1,
            )
        else:
            pose_actual = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg.dog.observe_track_error:
            pose_error = torch.stack(
                (
                    (height_target - height_actual) * self.obs_scales.body_height_cmd,
                    (self.commands_dog[:, dog_cmd_idx["body_pitch"]] - self.pitch) * self.obs_scales.body_pitch_cmd,
                    (self.commands_dog[:, dog_cmd_idx["body_roll"]] - self.roll) * self.obs_scales.body_roll_cmd,
                ),
                dim=-1,
            )
            velocity_error = torch.cat(
                (
                    (self.commands_dog[:, :2] - self.base_lin_vel[:, :2]) * self.obs_scales.lin_vel,
                    (self.commands_dog[:, 2:3] - self.base_ang_vel[:, 2:3]) * self.obs_scales.ang_vel,
                ),
                dim=-1,
            )
        else:
            pose_error = torch.zeros(self.num_envs, 3, device=self.device)
            velocity_error = torch.zeros(self.num_envs, 3, device=self.device)
        track_obs = torch.cat((pose_actual, pose_error, velocity_error), dim=-1)
        obs_buf = torch.cat((obs_buf, track_obs), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            obs_buf = torch.cat((obs_buf, heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            obs_buf = torch.cat(
                (obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(self.num_envs, -1) * 1.0), dim=1
            )

        if self.cfg.wbc.trajectory.enabled:
            obs_buf = torch.cat((obs_buf, self.get_ee_pose_body_9d()), dim=-1)

        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        arm_pos = (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice]) * self.obs_scales.dof_pos
        arm_vel = self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel
        obs_buf = torch.cat((obs_buf, arm_pos, arm_vel), dim=-1)

        if self.cfg.noise.add_noise:
            obs_buf = obs_buf + (2 * torch.rand_like(obs_buf) - 1) * self.dog_obs_noise_scale_vec

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

        # Simulated dropped sensor frames: each droppable segment (a real
        # sensor reading -- IMU, encoders, contacts, ... -- see
        # _dog_obs_layout) independently, per env, has a chance of
        # re-delivering its own last value instead of this step's fresh
        # one. Commanded/internal segments are never dropped.
        drop_prob = float(getattr(self.cfg.domain_rand, "dog_obs_frame_drop_prob", 0.0))
        if drop_prob > 0.0 and self.dog_obs_droppable_segments:
            dropped = (
                torch.rand(self.num_envs, len(self.dog_obs_droppable_segments), device=self.device) < drop_prob
            )
            for i, (start, end) in enumerate(self.dog_obs_droppable_segments):
                mask = dropped[:, i].unsqueeze(-1)
                obs_buf[:, start:end] = torch.where(mask, self.dog_last_delivered_obs[:, start:end], obs_buf[:, start:end])
        self.dog_last_delivered_obs = obs_buf.clone()

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
