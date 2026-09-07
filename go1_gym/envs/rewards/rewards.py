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

    # ------------------------------------------------------------------
    # Clock-free gait rewards (ported from robot_lab's
    # tasks/manager_based/locomotion/velocity/mdp/rewards.py).
    #
    # The stock gait terms -- tracking_contacts_shaped_force/vel,
    # feet_clearance_cmd_linear, raibert_heuristic -- all score the robot
    # against desired_contact_states / foot_indices, i.e. the absolute phase
    # of _step_contact_targets' clock. That phase resets to 0 on episode
    # reset and is then integrated from the gait_frequency command, so with
    # dog.observe_clock_inputs off the actor is being graded against a target
    # it cannot see. These terms replace that with quantities the robot
    # measures for itself: contact stopwatches, foot positions/velocities and
    # joint angles. Selected by rewards.gait_reward_mode -- see
    # config/core.set_gait_reward_mode.
    #
    # Every term carries robot_lab's upright gate: a tumbling robot has feet
    # nowhere near any sane target, and an ungated gait cost then explodes
    # exactly when the policy most needs the rest of its reward signal. That
    # is the failure mode the stage1_sim2real_abl_4/7/9 runs died of, where
    # raibert's per-step cost reached -0.21 and the ji22 shaping's
    # exp(rew_neg / 0.02) factor collapsed the total reward to ~0.
    # ------------------------------------------------------------------

    def _gait_upright_gate(self):
        """1 while level, fading to 0 as the base tips past ~45 deg."""
        return torch.clamp(-self.env.projected_gravity[:, 2], 0.0, 0.7) / 0.7

    def _gait_moving_gate(self):
        """1 where a nonzero velocity command is being given, else 0.

        Gait shaping must not fight the stand-still behaviour: at zero command
        _step_contact_targets pins every foot to the stance phase, and these
        terms would otherwise keep demanding a trot.
        """
        cmd = torch.norm(self.env.commands_dog[:, :3], dim=1)
        return (cmd > 0.1).float()

    def _gait_diagonal_pairs(self):
        """Foot index pairs that swing together in a trot.

        feet_indices follows the URDF body order FL, FR, RL, RR (the same
        order desired_contact_states is written in), so the diagonals are
        (FL, RR) = (0, 3) and (FR, RL) = (1, 2).
        """
        return (0, 3), (1, 2)

    def _reward_gait_sync(self):
        """Trot timing from contact stopwatches alone -- no clock.

        Positive, in [0, 1]: 1 when both diagonals are perfectly in phase with
        each other and perfectly out of phase with the other diagonal. Being
        bounded and positive matters under rewards.only_positive_rewards_ji22
        _style, where it lands in rew_buf_pos and multiplies the total instead
        of entering the exp(rew_neg / sigma_rew_neg) factor the way an
        unbounded cost like raibert_heuristic does.

        Replaces tracking_contacts_shaped_force + tracking_contacts_shaped_vel.
        """
        cfg = self.env.cfg.rewards
        std = cfg.gait_sync_sigma
        max_err_sq = cfg.gait_sync_max_err ** 2
        air = self.env.feet_air_time
        contact = self.env.feet_contact_time

        def sync(a, b):
            # Two feet that swing together: their air times should match, and
            # so should their contact times.
            se_air = torch.clip(torch.square(air[:, a] - air[:, b]), max=max_err_sq)
            se_con = torch.clip(torch.square(contact[:, a] - contact[:, b]), max=max_err_sq)
            return torch.exp(-(se_air + se_con) / std)

        def async_(a, b):
            # Two feet on opposite diagonals: one's air time should match the
            # other's contact time, in both directions.
            se_0 = torch.clip(torch.square(air[:, a] - contact[:, b]), max=max_err_sq)
            se_1 = torch.clip(torch.square(contact[:, a] - air[:, b]), max=max_err_sq)
            return torch.exp(-(se_0 + se_1) / std)

        (p0_a, p0_b), (p1_a, p1_b) = self._gait_diagonal_pairs()
        in_sync = sync(p0_a, p0_b) * sync(p1_a, p1_b)
        out_of_sync = (
            async_(p0_a, p1_a) * async_(p0_b, p1_b) * async_(p0_a, p1_b) * async_(p1_a, p0_b)
        )
        return in_sync * out_of_sync * self._gait_moving_gate() * self._gait_upright_gate()

    def _reward_feet_air_time_variance(self):
        """Cost: spread of the four legs' completed swing/stance durations.

        This is the direct measure of "one leg is not doing what the other
        three do" -- the asymmetric-gait symptom -- and it needs neither the
        clock nor any velocity estimate. Clipped at 0.5 s so a foot parked in
        stance (e.g. during a stumble) cannot dominate the variance.
        """
        clip_s = self.env.cfg.rewards.gait_air_time_clip
        var = torch.var(torch.clip(self.env.last_air_time, max=clip_s), dim=1) + torch.var(
            torch.clip(self.env.last_contact_time, max=clip_s), dim=1
        )
        return var * self._gait_upright_gate()

    def _reward_joint_mirror(self):
        """Cost: the two trot diagonals should be mirror images of each other.

        robot_lab compares raw joint angles, which only works when every leg
        shares one default pose. Here they do not -- init_state.default_joint
        _angles mirrors the hips (FL/RL +0.1, FR/RR -0.1) and gives the rear
        thighs a different nominal (1.0) from the front (0.8) -- so the
        comparison is made on deviations from each joint's own default, and
        the hip term is a *sum* rather than a difference because a mirrored
        pose has hip deviations of opposite sign on opposite sides.

        Needs no clock, no contact sensor and no velocity estimate: it is a
        pure symmetry prior on the joint angles.
        """
        n = self.env.num_actions_loco
        dev = self.env.dof_pos[:, :n] - self.env.default_dof_pos[:, :n]
        # dof order is FL, FR, RL, RR with (hip, thigh, calf) per leg.
        leg = dev.view(self.env.num_envs, 4, 3)
        (p0_a, p0_b), (p1_a, p1_b) = self._gait_diagonal_pairs()
        reward = 0.0
        for a, b in ((p0_a, p0_b), (p1_a, p1_b)):
            hip = torch.square(leg[:, a, 0] + leg[:, b, 0])
            thigh_calf = torch.sum(torch.square(leg[:, a, 1:] - leg[:, b, 1:]), dim=1)
            reward = reward + hip + thigh_calf
        return reward / 2.0 * self._gait_upright_gate()

    def _reward_feet_stance_width(self):
        """Positive, in [0, 1]: keep the feet at the commanded lateral stance.

        This is raibert_heuristic's lateral half rewritten as a bounded
        exponential. raibert_heuristic is an unbounded sum of squared errors
        that, at scale -10, reached -0.05/step in healthy runs and -0.21/step
        in falling ones -- enough on its own to drive the ji22 shaping's
        multiplicative factor to 1e-5. This form cannot do that: it is
        positive and saturates at 1.

        Uses only the commanded stance width and the feet's own body-frame
        positions, so it is independent of both the clock and the base
        velocity estimate.
        """
        feet_body = self._feet_positions_body_frame()
        if self.env.cfg.commands.use_dynamic_gait:
            stance_width = self.env.commands_dog[:, 8]
        else:
            stance_width = torch.full((self.env.num_envs,), 0.3, device=self.env.device)
        # Body +y is left; feet_indices order FL, FR, RL, RR alternates sides.
        side_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.env.device)
        desired_ys = (stance_width.unsqueeze(1) / 2.0) * side_sign.unsqueeze(0)
        err = torch.sum(torch.square(desired_ys - feet_body[:, :, 1]), dim=1)
        reward = torch.exp(-err / self.env.cfg.rewards.gait_stance_width_sigma)
        return reward * self._gait_upright_gate()

    def _reward_feet_swing_height(self):
        """Cost: swing-foot clearance, without asking the clock who is swinging.

        feet_clearance_cmd_linear weights the height error by
        (1 - desired_contact_states), i.e. by the clock's opinion of which feet
        are in swing. Here the weight is tanh(k * |foot horizontal velocity|),
        which identifies swing feet from their own motion instead. The height
        target is body-relative so it does not need an estimated base height.
        """
        cfg = self.env.cfg.rewards
        feet_body = self._feet_positions_body_frame()
        foot_vel_body = self._feet_velocities_body_frame()
        z_err = torch.square(feet_body[:, :, 2] - cfg.gait_swing_height_target)
        swinging = torch.tanh(cfg.gait_swing_tanh_mult * torch.norm(foot_vel_body[:, :, :2], dim=2))
        reward = torch.sum(z_err * swinging, dim=1)
        return reward * self._gait_moving_gate() * self._gait_upright_gate()

    def _feet_positions_body_frame(self):
        """Foot positions relative to the base, in the base frame."""
        translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        out = torch.zeros_like(translated)
        base_conj = quat_conjugate(self.env.base_quat)
        for i in range(4):
            out[:, i, :] = quat_apply(base_conj, translated[:, i, :])
        return out

    def _feet_velocities_body_frame(self):
        """Foot velocities relative to the base, in the base frame."""
        # foot_velocities is world-frame; subtract the base's own world
        # velocity so a fast-moving robot does not read as four swinging feet.
        translated = self.env.foot_velocities - self.env.root_states[: self.env.num_envs, 7:10].unsqueeze(1)
        out = torch.zeros_like(translated)
        base_conj = quat_conjugate(self.env.base_quat)
        for i in range(4):
            out[:, i, :] = quat_apply(base_conj, translated[:, i, :])
        return out
