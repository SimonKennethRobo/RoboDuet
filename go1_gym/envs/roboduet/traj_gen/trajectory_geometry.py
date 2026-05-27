"""Trajectory sampling helpers for task-space trajectory tracking.

Geometry primitives live in :mod:`go1_gym.utils.math_utils`; this module
keeps only the trajectory-shape sampler which is RoboDuet specific.
"""

import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul, quat_rotate, torch_rand_float

from go1_gym.utils.math_utils import (  # re-export for backward compatibility
    ee_twist_body_6d,
    pose_world_to_body_9d,
    quat_xyzw_to_rot6d,
)

__all__ = [
    "ee_twist_body_6d",
    "pose_world_to_body_9d",
    "quat_xyzw_to_rot6d",
    "sample_trajectory_commands",
]


def sample_trajectory_commands(
    cfg,
    end_effector_state,
    base_quat,
    env_ids,
    num_waypoints,
    device,
    length=None,
    s_curve_amplitude=None,
    orientation_scale=None,
    grasper_local_offset=(0.1, 0.0, 0.0),
):
    """Sample trajectory waypoints starting from the grasper position.

    The grasper (actual tool-tip) is offset from the EE link origin along the
    EE's local x-axis by ``grasper_local_offset``.  All visualisation code
    (``_draw_ee_ori_coord``, ``get_lpy_in_base_coord``) uses this same offset,
    so trajectory start aligns with the displayed EE marker.

    Args:
        length: per-env arc length (n_envs,) tensor, or None to use cfg default range.
        s_curve_amplitude: per-env S-curve amplitude (n_envs,) tensor, or None for cfg default.
        grasper_local_offset: (x, y, z) offset from EE link origin to grasper tip in EE frame.
    """
    n_envs = len(env_ids)
    t = torch.linspace(0.0, 1.0, num_waypoints, device=device).view(1, num_waypoints, 1)

    # Grasper position = EE origin + offset rotated by EE orientation.
    ee_pos = end_effector_state[env_ids, :3]
    ee_quat = end_effector_state[env_ids, 3:7]
    offset = torch.tensor(grasper_local_offset, dtype=torch.float, device=device).expand(n_envs, -1)
    grasper_pos = ee_pos + quat_rotate(ee_quat, offset)

    # Random direction in the horizontal plane (body frame)
    theta = torch_rand_float(-torch.pi, torch.pi, (n_envs, 1), device=device)
    direction_body = torch.cat((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1)
    lateral_body = torch.cat((-torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)), dim=-1)
    direction_world = quat_rotate(base_quat[env_ids], direction_body)
    lateral_world = quat_rotate(base_quat[env_ids], lateral_body)

    # Per-env arc length (curriculum)
    if length is None:
        lo, hi = cfg.arm.trajectory.length_range
        length = lo + (hi - lo) * torch.rand(n_envs, device=device)
    length = length.view(n_envs, 1, 1)

    # Per-env S-curve amplitude (curriculum)
    if s_curve_amplitude is None:
        lo, hi = cfg.arm.trajectory.s_curve_amplitude_range
        s_curve_amplitude = lo + (hi - lo) * torch.rand(n_envs, device=device)
    s_curve_amplitude = s_curve_amplitude.view(n_envs, 1, 1)

    sphere_direction = torch.randn(n_envs, 3, device=device)
    sphere_direction = sphere_direction / torch.clamp(torch.norm(sphere_direction, dim=-1, keepdim=True), min=1e-6)
    min_radius = torch.full_like(length, float(cfg.arm.trajectory.length_range[0]))
    start_radius = min_radius + torch.rand(n_envs, 1, 1, device=device) * torch.clamp(length - min_radius, min=0.0)
    start_offset = start_radius * sphere_direction[:, None, :]
    start_pos = grasper_pos[:, None, :] + start_offset

    # Per-env orientation curriculum: shrink roll/pitch/yaw range around 0 by `orientation_scale`.
    if orientation_scale is None:
        ori_scale = torch.ones(n_envs, device=device)
    else:
        ori_scale = orientation_scale.view(n_envs).to(device)

    def _scaled_uniform(lo, hi):
        u = torch_rand_float(float(lo), float(hi), (n_envs, 1), device=device).squeeze(-1)
        return u * ori_scale

    roll = _scaled_uniform(cfg.arm.commands.roll_ee[0], cfg.arm.commands.roll_ee[1])
    pitch = _scaled_uniform(cfg.arm.commands.pitch_ee[0], cfg.arm.commands.pitch_ee[1])
    yaw = _scaled_uniform(cfg.arm.commands.yaw_ee[0], cfg.arm.commands.yaw_ee[1])
    zero = torch.zeros_like(roll)
    random_local_quat = quat_mul(
        quat_from_euler_xyz(zero, zero, yaw),
        quat_mul(quat_from_euler_xyz(zero, pitch, zero), quat_from_euler_xyz(roll, zero, zero)),
    )
    start_quat = quat_mul(base_quat[env_ids], random_local_quat)

    line = length * t * direction_world[:, None, :]
    point = torch.zeros(n_envs, num_waypoints, 3, device=device)

    s_shape = (
        s_curve_amplitude
        * torch.sin(2.0 * torch.pi * cfg.arm.trajectory.s_curve_frequency * t)
        * lateral_world[:, None, :]
    )
    circle_phase = 2.0 * torch.pi * cfg.arm.trajectory.circle_turns * t
    circle = cfg.arm.trajectory.circle_radius * (
        torch.sin(circle_phase) * direction_world[:, None, :]
        + (1.0 - torch.cos(circle_phase)) * lateral_world[:, None, :]
    )

    # Traj type selection (supports cfg whitelist)
    traj_type_name = getattr(cfg.arm.trajectory, "traj_type", [])
    valid_types = {"line": 0, "s_curve": 1, "circle": 2, "point": 3}

    if isinstance(traj_type_name, str):
        selected = [valid_types.get(traj_type_name)]
    elif isinstance(traj_type_name, (list, tuple)):
        selected = [valid_types.get(name) for name in traj_type_name]
    else:
        selected = []

    selected = [v for v in selected if v is not None]
    if not selected:
        traj_type = torch.randint(0, len(valid_types), (n_envs, 1, 1), device=device)
    else:
        options = torch.tensor(selected, device=device)
        choice = torch.randint(0, len(selected), (n_envs, 1, 1), device=device)
        traj_type = options[choice]

    traj_offset = (
        (traj_type == 0).float() * line
        + (traj_type == 1).float() * (line + s_shape)
        + (traj_type == 2).float() * circle
        + (traj_type == 3).float() * point
    )

    time_range = cfg.arm.trajectory.completion_time_range[1] - cfg.arm.trajectory.completion_time_range[0]
    target_time = cfg.arm.trajectory.completion_time_range[0] + torch.rand(n_envs, device=device) * time_range
    traj_quat = start_quat[:, None, :].expand(-1, num_waypoints, -1).clone()

    return (
        start_pos + traj_offset,
        traj_quat,
        target_time,
        traj_type.view(n_envs),
    )
