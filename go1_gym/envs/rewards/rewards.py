import torch
import torch.nn.functional as F
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from go1_gym.envs.roboduet.legged_robot import LeggedRobot
from go1_gym.response.reward_terms import (
    domain_consistency,
    phase_variance,
    reference_tracking,
    steady_gain,
)

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
        # self.env.actions[:, arm_slice] gets overwritten by the action-mode
        # decode step (see _apply_stage2_arm_action), so the raw pre-decode
        # policy output is kept separately for this saturation check. It means
        # the same thing in every arm.action_mode: keep |a| <= 1, i.e. inside
        # the residual / waypoint / joint-target range the mode was scaled for.
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

    # --- R4: response-consistency terms ---------------------------------
    # Thin wrappers; the maths and its rationale live in
    # go1_gym/response/reward_terms.py, where they are unit-tested against
    # synthetic trajectories.

    def _reward_ref_tracking(self):
        # R4.1, the core term. Positive, so it enters the ji22 composition as a
        # task factor. Note it tracks the reference STATE, not the command: at
        # t=0.1 s into a 0.4 rad step the target is 0.036 rad, so arriving early
        # costs exactly as much as arriving late.
        env = self.env
        return reference_tracking(
            env.response_detrended,
            env.response_ref.xi,
            env.response_sigma,
            env.response_channel_weights,
            gate=env.response_soft_gate,
        )

    def _reward_phase_variance(self):
        # R4.2. Penalises the variance about the phase-conditioned mean, NOT the
        # oscillation amplitude: a large but repeatable ripple is predictable and
        # the arm can cancel it; a small but wandering one leaks straight through
        # to the end effector.
        env = self.env
        return phase_variance(
            env.response_oscillation,
            env.response_delta_hat,
            env.response_channel_weights,
            mask=env.response_phase_variance_mask,
        )

    def _reward_steady_gain(self):
        # R4.3. Once a command has been held long enough, compare against the
        # COMMAND rather than the reference state, pinning the DC gain the MPC
        # will assume to 1.
        env = self.env
        return steady_gain(
            env.response_detrended,
            env.response_ref.gather_commands(env.commands_dog),
            env.response_channel_weights,
            mask=env.response_steady_gain_mask,
        )

    def _reward_domain_consistency(self):
        # R5. The same command in a harder domain must produce the twin's
        # response.  Masked off for the twin itself and for any group whose
        # phase drifted after a fall -- see EnvGrouping.valid.
        env = self.env
        return domain_consistency(
            env.response_detrended,
            env.response_twin_detrended,
            env.response_channel_weights,
            env.grouping.valid,
        )

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

    def _reward_stay_still_in_reach_sector(self):
        """Penalize vx/vy/yaw-rate commands while the goal's ground-plane
        projection sits inside a forward-facing sector (radius + half-angle,
        both hyperparameters) -- the base shouldn't reposition for a goal the
        arm can already reach from where it's standing."""
        cfg = self.env.cfg.wbc.goal_reaching
        goal_xy_body = self.env.arm_target_pos_body[:, :2]
        radius = torch.linalg.vector_norm(goal_xy_body, dim=-1)
        bearing = torch.atan2(goal_xy_body[:, 1], goal_xy_body[:, 0]).abs()
        in_sector = (radius <= float(cfg.stay_sector_radius)) & (bearing <= float(cfg.stay_sector_half_angle))
        vel_cmd_sq = torch.sum(torch.square(self.env.commands_dog[:, :3]), dim=-1)
        return torch.where(in_sector, vel_cmd_sq, torch.zeros_like(vel_cmd_sq))

    # ---- trajectory tracking (target_mode='trajectory', Group T) ----
    # All read env buffers precomputed each step in _arm_post_physics_hook.

    def _reward_traj_progress(self):
        """Reward the EE actually advancing along the path (measured arc-length
        speed, clamped >=0 so backsliding isn't rewarded).

        Capped at the reference speed sdot_ref: running *ahead* of the time law
        earns nothing extra, it only buys a traj_timing deadzone penalty. Without
        the cap this term (whose episode integral is w_p * (s_final - s_init),
        i.e. a pure end-of-path bonus that discounting turns into "get there
        sooner") pulls against traj_timing. The cap keeps the dense
        anti-freeze/exploration signal -- which is the term's real job once
        ee_pos_tracking saturates at large error, and the only progress
        incentive left in the tau -> inf pure-path-following regime."""
        return torch.minimum(self.env.traj_sdot_meas.clamp(min=0.0), self.env.traj_sdot_ref)

    def _reward_traj_lateral_err(self):
        """Penalize distance from the EE to the reference path (d_lat)."""
        return self.env.traj_d_lat

    def _reward_traj_timing(self):
        """Deadzone penalty on being ahead of / behind the reference arc length
        by more than tau (a time-tube of tolerance)."""
        tau = float(self.env.cfg.wbc.goal_reaching.trajectory.timing_tau)
        return F.relu(self.env.traj_timing_err.abs() - tau)

    def _reward_traj_twist_err(self):
        """Penalize EE spatial velocity that deviates from the reference
        velocity (tangent * sdot_ref) -- i.e. wrong speed or off-tangent
        motion."""
        return self.env.traj_twist_err

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        # Despite the name this is the body-height tracking term. Terrain height
        # is subtracted through the shared helper rather than assumed to be 0:
        # on flat ground it returns 0 (identical behaviour), and on rough
        # terrain a world-frame height would violate global invariant 9.
        reference_heights = self.env._terrain_reference_height()
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
        reference_heights = self.env._terrain_reference_height().unsqueeze(-1)
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

    def _orientation_error(self):
        """Squared projected-gravity error against the commanded attitude.

        **Sign convention (decided 2026-09-02): a command of +0.3 means the body
        pitches to +0.3 rad in the standard rpy sense.**  This used to negate the
        commands, which put it in direct conflict with the rest of the codebase:
        the observation hands the policy ``pose_target - pose_measured`` with
        ``pose_target = +command`` and ``pose_measured = +self.pitch``, and R2's
        reference model and R4.1/R4.3 all read ``+command`` too.  Only this term
        disagreed, and it won while it was the only pitch term -- a trained
        stage-1 policy measured a DC gain of **-0.28**: inverted, and weak.

        Left unfixed it would have got worse rather than better: once R4.1 ramps
        in at stage 2 it drives pitch to +command at weight 2.0 while this term
        drove it to -command, so the two would have fought.

        Returns the two axes separately.  ``projected_gravity[:, 0]`` tracks
        pitch and ``[:, 1]`` tracks roll, and they need different weights: R4.1
        tracks a reference for **pitch** (one of the five decision channels) but
        there is no decision channel for **roll** -- R1 freezes roll to zero and
        excludes it from the command space.  So the roll half is the only thing
        keeping the body level and must never be weakened on the assumption that
        R4.1 takes over, because for roll it never does.
        """
        pitch_command = self.env.commands_dog[:, 3]
        roll_command = self.env.commands_dog[:, 4]
        quat_roll = quat_from_angle_axis(roll_command,
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(pitch_command,
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)
        error = torch.square(
            self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]
        )
        return error[:, 0], error[:, 1]      # pitch, roll

    def _reward_orientation_control(self):
        # Roll only. Nothing else in the reward set controls roll.
        return self._orientation_error()[1]

    def _reward_pitch_control(self):
        # Pitch only, so the curriculum can hand this channel over to R4.1 as
        # R4.1 ramps in without also releasing roll.
        return self._orientation_error()[0]

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
