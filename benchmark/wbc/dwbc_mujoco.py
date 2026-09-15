"""Deep Whole-Body Control checkpoint adapter for the common MuJoCo plant."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch


JOINT_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
] + [f"x5_joint{i}" for i in range(1, 7)]
VISUAL_POLICY_ORDER = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17])


class DwbcMujoco:
    controls_arm = True

    def __init__(self, root, checkpoint, scene, variant="dwbc"):
        self.root = Path(root).resolve()
        self.variant = variant
        self.controls_arm = True
        self.arm_position_drive = variant == "visual"
        sys.path.insert(0, str(self.root / ("third_party/rsl_rl" if variant == "visual" else "rsl_rl")))
        from rsl_rl.modules.actor_critic import ActorCritic

        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.n = 18
        joints = np.asarray([self.model.joint(name).id for name in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[joints].astype(int)
        self.vadr = self.model.jnt_dofadr[joints].astype(int)
        self.mapping = []
        for joint in joints:
            ids = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(ids) != 1:
                raise ValueError(f"expected one actuator for joint id {joint}")
            self.mapping.append(int(ids[0]))
        self.default_dof_pos = np.asarray([
            0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
            0.1, 1.0 if variant == "visual" else 0.8, -1.5,
            -0.1, 1.0 if variant == "visual" else 0.8, -1.5,
            0.0, 0.6 if variant == "visual" else 1.5,
            0.5 if variant == "visual" else 1.5, 0.0, 0.0, 0.0,
        ])
        self.scale = np.asarray([0.4, 0.45, 0.45] * 4 + [1.0, 0.6, 0.6, 0.5, 0.5, 0.5])
        limits = self.model.actuator_ctrlrange[self.mapping]
        self.p = {
            "decimation": 4 if variant == "visual" else 8,
            "rl_kp": [35.0] * 12 + [50.0, 50.0, 80.0, 30.0, 20.0, 20.0],
            "rl_kd": [1.0] * 12 + (
                [8.0, 15.0, 15.0, 5.0, 4.0, 2.0]
                if variant == "visual" else [2.0, 3.0, 3.0, 0.5, 0.3, 0.1]
            ),
            "torque_limits": np.maximum(abs(limits[:, 0]), abs(limits[:, 1])).tolist(),
        }
        self.control_dt = 0.005 if variant == "visual" else 0.0025
        self.policy_dt = 0.02
        self.substeps = 1
        prop_width = 71 if variant == "visual" else 76
        priv_width = 18 if variant == "visual" else 24
        self.priv_width = priv_width
        self.actor_critic = ActorCritic(
            prop_width, prop_width, 18, actor_hidden_dims=[128], critic_hidden_dims=[128],
            leg_control_head_hidden_dims=[128, 128], arm_control_head_hidden_dims=[128, 128],
            priv_encoder_dims=[64, 20], num_leg_actions=12, num_arm_actions=6,
            adaptive_arm_gains=False, adaptive_arm_gains_scale=10.0,
            num_priv=priv_width, num_hist=10, num_prop=prop_width, zero_actor_output=False,
            init_std=[[1.0] * 18], output_tanh=False,
        ).eval()
        saved = torch.load(checkpoint, map_location="cpu")
        self.actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
        self.checkpoint_path = Path(checkpoint)
        self.history = np.zeros((10, prop_width), dtype=np.float32)
        self.actions = np.zeros(18)
        self.latest_actions = np.zeros(18)
        self.action_queue = [np.zeros(18) for _ in range(1 if variant == "visual" else 2)]
        self.arm_pos_targets = None
        self.command = [0.0] * 6
        self.gait_indices = 0.0
        self.goal_position = None
        self.goal_quaternion = None
        self.base = self.model.body("base_link").id

    def reset_policy(self):
        self.history.fill(0.0)
        self.actions.fill(0.0)
        self.latest_actions.fill(0.0)
        self.action_queue = [np.zeros(18) for _ in range(1 if self.variant == "visual" else 2)]
        self.arm_pos_targets = self.data.qpos[self.qadr[12:18]].copy()

    def set_reference(self, reference, time_s):
        _arc, self.goal_position, self.goal_quaternion = reference.at(time_s)

    def read_state(self):
        q = self.data.qpos[self.qadr].copy()
        dq = self.data.qvel[self.vadr].copy()
        quat = self.data.qpos[[4, 5, 6, 3]].copy()
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return q, dq, quat, self.data.qvel[3:6].copy(), self.data.qpos[:3].copy(), rotation.T @ self.data.qvel[:3]

    def _contacts(self):
        names = ("FL", "FR", "RL", "RR")
        ids = {self.model.geom(name).id: row for row, name in enumerate(names)}
        result = np.zeros(4)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for geom in (int(contact.geom1), int(contact.geom2)):
                if geom in ids:
                    result[ids[geom]] = 1.0
        return result

    @staticmethod
    def _euler(quat):
        x, y, z, w = quat
        return np.asarray([
            np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
            np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
            np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
        ])

    def forward(self, q, dq, quat, gyro, base_pos, lin_vel):
        del lin_vel
        if self.goal_position is None:
            raise RuntimeError("DWBC target was not set")
        rpy = self._euler(quat)
        yaw = rpy[2]
        c, s = np.cos(yaw), np.sin(yaw)
        relative = self.goal_position - np.asarray([base_pos[0], base_pos[1], 0.53])
        local = np.asarray([c*relative[0] + s*relative[1], -s*relative[0] + c*relative[1], relative[2]])
        radius = np.linalg.norm(local)
        sphere = np.asarray([radius, np.arctan2(local[2], np.linalg.norm(local[:2])), np.arctan2(local[1], local[0])])
        goal_rpy = self._euler(self.goal_quaternion)
        orientation_delta = (goal_rpy - np.asarray([0.0, 0.0, yaw]) + np.pi) % (2*np.pi) - np.pi
        q20 = np.r_[q, 0.0, 0.0]
        dq20 = np.r_[dq, 0.0, 0.0]
        if self.variant == "visual":
            # Visual low-level keeps arm motion in scripted IK and observes
            # only the previous 12 leg actions plus five standing-gait slots.
            arm_base = base_pos + self.data.xmat[self.base].reshape(3, 3) @ np.array([0.05, 0.0, 0.1])
            base_rotation = self.data.xmat[self.base].reshape(3, 3)
            local_goal = base_rotation.T @ (self.goal_position - arm_base)
            relative_rotation = base_rotation.T @ self._quat_matrix(self.goal_quaternion)
            wxyz = np.empty(4)
            mujoco.mju_mat2Quat(wxyz, relative_rotation.reshape(-1))
            local_orientation = wxyz[1:] * (1.0 if wxyz[0] >= 0.0 else -1.0)
            prop = np.r_[rpy[:2], gyro, (q - self.default_dof_pos)[VISUAL_POLICY_ORDER],
                         dq[VISUAL_POLICY_ORDER] * 0.05,
                         self.latest_actions[:12], self._contacts()[[1, 0, 3, 2]], [0.0, 0.0, 0.0],
                         local_goal, local_orientation, np.zeros(5)].astype(np.float32)
        else:
            prop = np.r_[rpy[:2], gyro, q20 - np.r_[self.default_dof_pos, 0.0, 0.0],
                         dq20 * 0.05, self.latest_actions, self._contacts(), [0.0, 0.0, 0.0],
                         sphere, orientation_delta].astype(np.float32)
        if prop.shape != (self.history.shape[1],) or not np.isfinite(prop).all():
            raise ValueError(f"invalid DWBC proprioception {prop.shape}")
        if not np.any(self.history):
            self.history[:] = prop
        obs = np.r_[prop, np.zeros(self.priv_width, np.float32), self.history.reshape(-1)]
        with torch.inference_mode():
            action = self.actor_critic.actor(torch.from_numpy(obs[None]), hist_encoding=True)[0].numpy()
        self.history[:-1] = self.history[1:]
        self.history[-1] = prop
        self.latest_actions = np.clip(action, -100.0, 100.0)
        if self.variant == "visual":
            self.latest_actions[12:] = 0.0
        self.action_queue.append(self.latest_actions.copy())
        executed = self.action_queue.pop(0)
        self.actions = executed[VISUAL_POLICY_ORDER] if self.variant == "visual" else executed
        return self.actions.copy()

    def compute_output(self, actions):
        target = self.default_dof_pos + self.scale * actions
        if self.variant == "visual":
            target[12:18] = self._visual_arm_target()
        return target

    def _visual_arm_target(self):
        """Mirror ManipLoco._control_ik and its persistent, bounded integrator."""
        site = self.model.site("x5_ee").id
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
        jacobian = np.vstack((jacp[:, self.vadr[12:18]], .75 * jacr[:, self.vadr[12:18]]))
        relative = self._quat_matrix(self.goal_quaternion) @ self.data.site_xmat[site].reshape(3, 3).T
        wxyz = np.empty(4)
        mujoco.mju_mat2Quat(wxyz, relative.reshape(-1))
        error = np.r_[self.goal_position - self.data.site_xpos[site],
                      .75 * wxyz[1:] * (1.0 if wxyz[0] >= 0 else -1.0)]
        pinv = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05**2 * np.eye(6), np.eye(6))
        q = self.data.qpos[self.qadr[12:18]]
        delta = .2 * (pinv @ error + .15 * (np.eye(6) - pinv @ jacobian) @
                      (self.default_dof_pos[12:18] - q))
        delta *= min(1.0, .025 / max(np.linalg.norm(delta), 1e-8))
        if self.arm_pos_targets is None:
            self.arm_pos_targets = q.copy()
        joints = [self.model.joint(name).id for name in JOINT_NAMES[12:18]]
        limits = self.model.jnt_range[joints]
        self.arm_pos_targets = np.clip(self.arm_pos_targets + delta, limits[:, 0], limits[:, 1])
        return self.arm_pos_targets.copy()

    @staticmethod
    def _quat_matrix(quat):
        x, y, z, w = quat / np.linalg.norm(quat)
        return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                           [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                           [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
