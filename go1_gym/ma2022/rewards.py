"""Task-owned reward terms from Miki et al. 2022, Supplement S7 (Ma ref. 10).

The printed command equations have a normalization/sign ambiguity. We use a
unit desired direction and the commanded speed as threshold, including signed
yaw. This gives mirrored rewards for mirrored commands, a plateau at/above
requested speed and a separate lateral-motion term. This interpretation is
explicit, not an assertion of access to the authors' implementation.
"""
import torch


def command_rewards(command, velocity, yaw_command, yaw_velocity):
    speed = command.norm(dim=-1)
    direction = command / speed.clamp_min(1e-8).unsqueeze(-1)
    projected = (direction * velocity).sum(-1)
    linear = torch.exp(-torch.clamp(speed - projected, min=0.).square())
    linear = torch.where(speed > 1e-8, linear, torch.exp(-velocity.square().sum(-1)))
    orthogonal = velocity - projected.unsqueeze(-1) * direction
    lateral = torch.exp(-3. * orthogonal.square().sum(-1))
    yaw = torch.exp(-torch.clamp(yaw_command.abs() - yaw_command.sign()*yaw_velocity, min=0.).square())
    yaw = torch.where(yaw_command.abs() > 1e-8, yaw, torch.exp(-yaw_velocity.square()))
    return linear, yaw, lateral


def swing_lift(phase):
    """Reference [10] S5 cubic Hermite lift, phase in cycles, unit height."""
    phase = phase.remainder(1.)
    t = torch.where(phase <= .25, 4.*phase, 2.-4.*phase).clamp(0., 1.)
    return t.square() * (3. - 2.*t)


def reward_terms(*, command, linear, angular, gravity, dof_pos, dof_vel,
                 previous_dof_vel, target, last_target, previous_target,
                 torque, feet_velocity, feet_contact, leg_collision,
                 foot_heights, phase, knee_indices, knee_limit, dt,
                 curriculum, stability_multiplier, joint_acceleration_coef, terminated):
    lv, av, lateral = command_rewards(command[:, :2], linear[:, :2], command[:, 2], angular[:, 2])
    return dict(
        tracking_lin_vel=lv,
        tracking_ang_vel=av,
        orthogonal_velocity=lateral,
        body_motion=-stability_multiplier*(1.25*linear[:, 2].square() + .4*angular[:, :2].abs().sum(-1)),
        # Ma requests stronger orientation control, but does not give its
        # formula/weight. Squared horizontal gravity remains a labeled choice.
        orientation=-stability_multiplier*curriculum*gravity[:, :2].square().sum(-1),
        foot_clearance=-((foot_heights.amax(-1) < -.2) & (phase < .5)).float().sum(-1),
        collision=-curriculum*leg_collision.any(-1).float(),
        joint_motion=-curriculum*(.01*dof_vel.square() + joint_acceleration_coef*(
            (dof_vel-previous_dof_vel)/dt).square()).sum(-1),
        knee_limit=-torch.clamp(dof_pos[:, knee_indices]-knee_limit, min=0.).square().sum(-1),
        target_smoothness=-curriculum*((target-last_target).square() +
                                      (target-2*last_target+previous_target).square()).sum(-1),
        torques=-curriculum*torque.square().sum(-1),
        foot_slip=-curriculum*(feet_velocity.square().sum(-1)*feet_contact).sum(-1),
        termination=-terminated.float(),
    )
