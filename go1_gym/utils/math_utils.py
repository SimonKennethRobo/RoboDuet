# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from typing import Tuple

import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym.torch_utils import (
    normalize,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    to_torch,
)
from torch import Tensor


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


def get_scale_shift(range):
    scale = 2. / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.
    return scale, shift


def quat_to_angle(quat):
    """Decompose a quaternion (xyzw) into (alpha, beta, gamma) Euler-like angles."""
    device = quat.device
    n = quat.shape[0]
    y_vector = to_torch([0.0, 1.0, 0.0], device=device).repeat((n, 1))
    z_vector = to_torch([0.0, 0.0, 1.0], device=device).repeat((n, 1))
    x_vector = to_torch([1.0, 0.0, 0.0], device=device).repeat((n, 1))
    roll_vec = quat_apply(quat, y_vector)
    alpha = torch.atan2(roll_vec[:, 2], roll_vec[:, 1])
    pitch_vec = quat_apply(quat, z_vector)
    beta = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2])
    yaw_vec = quat_apply(quat, x_vector)
    gamma = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0])
    return torch.stack([alpha, beta, gamma], dim=-1)


def quat_xyzw_to_rot6d(quat):
    quat_wxyz = quat[:, [3, 0, 1, 2]]
    return pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quat_wxyz))


def pose_world_to_body_9d(pos_world, quat_world, base_pos, base_quat):
    pos_body = quat_rotate_inverse(base_quat, pos_world - base_pos)
    quat_body = quat_mul(quat_conjugate(base_quat), quat_world)
    return torch.cat((pos_body, quat_xyzw_to_rot6d(quat_body)), dim=-1)


def quat_error_axis_angle(quat_target, quat_current):
    """Orientation error between quat_target and quat_current (xyzw quats,
    both expressed in the same frame): the vector part of
    q_target * q_current^-1, sign-corrected for the shortest-path rotation.

    This is the bounded small-angle proxy for the true axis-angle log map
    (magnitude saturates at 1 instead of growing to pi as the true log map
    does), matching IsaacGymEnvs' franka_reach.py orientation_error. Using
    the unbounded true log map here was an earlier bug: for a large initial
    rotation error its magnitude (up to ~pi) dwarfs the position-error rows
    in the DLS dpose vector, so the solve overcorrects rotation at position's
    expense. Continuous and zero exactly at zero error either way."""
    q_err = quat_mul(quat_target, quat_conjugate(quat_current))
    return q_err[:, :3] * torch.sign(q_err[:, 3:4])


def ee_twist_body_6d(end_effector_state, root_states, base_quat, num_envs):
    ee_lin_vel_world = end_effector_state[:, 7:10]
    ee_ang_vel_world = end_effector_state[:, 10:13]
    base_lin_vel_world = root_states[:num_envs, 7:10]
    base_ang_vel_world = root_states[:num_envs, 10:13]
    rel_lin_vel_body = quat_rotate_inverse(base_quat, ee_lin_vel_world - base_lin_vel_world)
    rel_ang_vel_body = quat_rotate_inverse(base_quat, ee_ang_vel_world - base_ang_vel_world)
    return torch.cat((rel_lin_vel_body, rel_ang_vel_body), dim=-1)
