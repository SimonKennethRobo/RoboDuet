import torch
import torch.nn.functional as F
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from go1_gym.envs.roboduet.legged_robot import LeggedRobot

class Rewards:
    def __init__(self, env):
        self.env: LeggedRobot = env

    def load_env(self, env):
        self.env = env

    # arm rewards
    def _reward_ee_pos_tracking(self):
        # DLS-IK MVP task-space position tracking (project-design-v3.md
        # Group T). Exponential form: 1 at zero error, decays smoothly,
        # matching _reward_tracking_lin_vel's convention below.
        pos_err_sq = torch.sum(torch.square(self.env.ee_pos_err), dim=-1)
        return torch.exp(-pos_err_sq / self.env.cfg.rewards.ee_pos_tracking_sigma)

    def _reward_ee_rot_tracking(self):
        rot_err_sq = torch.sum(torch.square(self.env.ee_rot_err_axis_angle), dim=-1)
        return torch.exp(-rot_err_sq / self.env.cfg.rewards.ee_rot_tracking_sigma)

    def _reward_arm_control_limits(self):
        # self.env.actions[:, arm_slice] gets overwritten by the IK combine
        # step (see _apply_stage2_arm_ik_action), so the raw pre-combine
        # policy output is kept separately for this saturation check.
        return torch.sum(torch.square((torch.abs(self.env.arm_residual_raw) - 1.0).clip(min=0.0)), dim=1)

    def _reward_arm_energy(self):
        energy_sum = torch.sum(
            torch.square(self.env.torques[:, self.env.num_actions_loco:]*self.env.dof_vel[:, self.env.num_actions_loco:])
            , dim=1)
        return energy_sum

    def _reward_arm_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., self.env.num_actions_loco:]), dim=1)

    def _reward_arm_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., self.env.num_actions_loco:] / self.env.dt), dim=1)

    def _reward_arm_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., self.env.num_actions_loco:], dim=1)

    def _reward_arm_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - 2 * self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2] + self.env.last_last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, self.env.num_actions_loco:] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_ee_smoothness(self):
        twist = self.env.get_ee_twist_body()
        return torch.sum(torch.square((twist - self.env.prev_ee_twist_body) / self.env.dt), dim=-1)

    # dog rewards
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands_dog[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands_dog[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_response_consistency(self):
        # Penalizes deviation from a first-order reference model of the xy
        # velocity command (self.env.dog_vel_ref, updated every step in
        # LeggedRobot._update_dog_vel_ref). Unlike tracking_lin_vel, which
        # rewards matching the raw (discontinuous) command, this keeps the
        # realised response close to a predictable linear plant regardless
        # of payload/posture disturbance, so an upstream planner can rely on
        # a fixed time constant when computing feedforward base motion.
        return torch.sum(torch.square(self.env.base_lin_vel[:, :2] - self.env.dog_vel_ref), dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_loco_energy(self):

        # print("loco energy: ", -0.00005*torch.sum(torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco]), dim=1)[:20])
        return torch.sum(
            torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco])
            , dim=1)

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., :self.env.num_actions_loco]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., :self.env.num_actions_loco] / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., :self.env.num_actions_loco], dim=1)

    def _count_contacts(self, indices):
        return torch.sum(
            torch.norm(self.env.contact_forces[:, indices, :], dim=-1) > 0.1,
            dim=1,
        ).float()

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return self._count_contacts(self.env.penalised_contact_indices)

    def _reward_arm_contact(self):
        return self._count_contacts(self.env.arm_contact_indices)

    def _reward_goal_pos_l2(self):
        return torch.linalg.vector_norm(self.env.ee_pos_err, dim=-1)

    def _reward_reachability_barrier(self):
        cfg = self.env.cfg.wbc.goal_reaching
        beta = 0.05
        return F.softplus((self.env.goal_rho - float(cfg.rho_hi)) / beta) + F.softplus(
            (float(cfg.rho_lo) - self.env.goal_rho) / beta
        )

    def _reward_manipulability(self):
        return torch.log1p(self.env.goal_manipulability)

    def _reward_joint_limit_barrier(self):
        margin = torch.clamp(0.10 - self.env.goal_joint_limit_distance, min=0.0)
        return torch.sum(torch.square(margin), dim=-1)

    def _reward_arm_ema_motion(self):
        return torch.linalg.vector_norm(self.env.arm_ema_motion, dim=-1)

    def _reward_rho_rate(self):
        return torch.abs(self.env.goal_rho - self.env.goal_rho_prev) / self.env.dt

    def _reward_upper_action_rate(self):
        return torch.sum(torch.square(self.env.arm_policy_actions - self.env.last_arm_policy_actions), dim=-1)

    def _reward_delta_vel_magnitude(self):
        return torch.sum(torch.square(self.env.delta_velocity_cmd), dim=-1)

    def _reward_posture_command_rate(self):
        start = self.env.num_actions_arm + 3
        current = self.env.arm_policy_actions[:, start : start + 3]
        previous = self.env.last_arm_policy_actions[:, start : start + 3]
        return torch.sum(torch.square(current - previous), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = self.env.commands_dog[:, 5] + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - self.env.last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actions_loco] + self.env.last_last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_actions_loco] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        if self.env.cfg.commands.use_dynamic_gait:
            footswing_height = self.env.commands_dog[:, 7:8]  # (num_envs, 1)
        else:
            footswing_height = 0.04
        target_height = footswing_height * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # import ipdb; ipdb.set_trace()
        roll_pitch_commands = self.env.commands_dog[:, 3:5]
        # print(roll_pitch_commands)
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.use_dynamic_gait:
            desired_stance_width = self.env.commands_dog[:, 8]  # (num_envs,)
            desired_stance_length = self.env.commands_dog[:, 9]  # (num_envs,)
        else:
            desired_stance_width = torch.full((self.env.num_envs,), 0.3, device=self.env.device)
            desired_stance_length = torch.full((self.env.num_envs,), 0.45, device=self.env.device)
        desired_ys_nom = torch.stack([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)

        desired_xs_nom = torch.stack([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        if self.env.cfg.commands.use_dynamic_gait:
            frequencies = torch.clamp(self.env.commands_dog[:, 6:7], min=0.1)  # (num_envs, 1)
        else:
            frequencies = 3.
        x_vel_des = self.env.commands_dog[:, 0:1]
        yaw_vel_des = self.env.commands_dog[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length.unsqueeze(1) / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
