"""Geometry helpers for task-space trajectory tracking."""

import pytorch3d.transforms as pt3d
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse, torch_rand_float


def quat_xyzw_to_rot6d(quat):
    quat_wxyz = quat[:, [3, 0, 1, 2]]
    return pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quat_wxyz))


def pose_world_to_body_9d(pos_world, quat_world, base_pos, base_quat):
    pos_body = quat_rotate_inverse(base_quat, pos_world - base_pos)
    quat_body = quat_mul(quat_conjugate(base_quat), quat_world)
    return torch.cat((pos_body, quat_xyzw_to_rot6d(quat_body)), dim=-1)


def ee_twist_body_6d(end_effector_state, root_states, base_quat, num_envs):
    ee_lin_vel_world = end_effector_state[:, 7:10]
    ee_ang_vel_world = end_effector_state[:, 10:13]
    base_lin_vel_world = root_states[:num_envs, 7:10]
    base_ang_vel_world = root_states[:num_envs, 10:13]
    rel_lin_vel_body = quat_rotate_inverse(base_quat, ee_lin_vel_world - base_lin_vel_world)
    rel_ang_vel_body = quat_rotate_inverse(base_quat, ee_ang_vel_world - base_ang_vel_world)
    return torch.cat((rel_lin_vel_body, rel_ang_vel_body), dim=-1)


def sample_trajectory_commands(cfg, end_effector_state, base_quat, env_ids, num_waypoints, device):
    n_envs = len(env_ids)
    t = torch.linspace(0.0, 1.0, num_waypoints, device=device).view(1, num_waypoints, 1)

    start_pos = end_effector_state[env_ids, :3]
    start_quat = end_effector_state[env_ids, 3:7]
    start_offset_body = torch_rand_float(
        -cfg.arm.trajectory.start_radius,
        cfg.arm.trajectory.start_radius,
        (n_envs, 3),
        device=device,
    )
    start_pos = start_pos + quat_rotate(base_quat[env_ids], start_offset_body)

    theta = torch_rand_float(-torch.pi, torch.pi, (n_envs, 1), device=device)
    direction_body = torch.cat((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1)
    lateral_body = torch.cat((-torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)), dim=-1)
    direction_world = quat_rotate(base_quat[env_ids], direction_body)
    lateral_world = quat_rotate(base_quat[env_ids], lateral_body)

    line = cfg.arm.trajectory.length * t * direction_world[:, None, :]
    s_shape = (
        cfg.arm.trajectory.s_curve_amplitude
        * torch.sin(2.0 * torch.pi * cfg.arm.trajectory.s_curve_frequency * t)
        * lateral_world[:, None, :]
    )
    circle_phase = 2.0 * torch.pi * cfg.arm.trajectory.circle_turns * t
    circle = cfg.arm.trajectory.circle_radius * (
        torch.sin(circle_phase) * direction_world[:, None, :]
        + (1.0 - torch.cos(circle_phase)) * lateral_world[:, None, :]
    )
    traj_type = torch.randint(0, 3, (n_envs, 1, 1), device=device)
    traj_offset = (
        (traj_type == 0).float() * line
        + (traj_type == 1).float() * (line + s_shape)
        + (traj_type == 2).float() * circle
    )

    time_range = cfg.arm.trajectory.completion_time_range[1] - cfg.arm.trajectory.completion_time_range[0]
    target_time = cfg.arm.trajectory.completion_time_range[0] + torch.rand(n_envs, device=device) * time_range
    return start_pos[:, None, :] + traj_offset, start_quat[:, None, :].expand(-1, num_waypoints, -1), target_time
