"""Official UMI-on-Legs actor adapter for the common Go2+X5 MuJoCo plant."""

from pathlib import Path

import mujoco
import numpy as np
import torch


JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
               for part in ("hip", "thigh", "calf")] + [f"x5_joint{i}" for i in range(1, 7)]


def _quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class UmiMujoco:
    controls_arm = True

    def __init__(self, checkpoint, scene):
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
        self.default_dof_pos = np.asarray([0.1, .8, -1.5, -.1, .8, -1.5,
                                           .1, 1., -1.5, -.1, 1., -1.5,
                                           0., .3, .5, 0., 0., 0.])
        limits = self.model.actuator_ctrlrange[self.mapping]
        self.p = {"decimation": 4,
                  "rl_kp": [40.] * 12 + [100., 100., 100., 20., 20., 5.],
                  "rl_kd": [1.] * 12 + [3., 3., 3., 2., 1., .5],
                  "torque_limits": np.maximum(abs(limits[:, 0]), abs(limits[:, 1])).tolist()}
        self.control_dt, self.policy_dt, self.substeps = .005, .02, 1
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(96, 128), torch.nn.ELU(), torch.nn.Linear(128, 64),
            torch.nn.ELU(), torch.nn.Linear(64, 32), torch.nn.ELU(), torch.nn.Linear(32, 18),
        ).eval()
        saved = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
        self.actor.load_state_dict({k[len("actor."):]: v for k, v in saved.items()
                                    if k.startswith("actor.")}, strict=True)
        self.checkpoint_path = Path(checkpoint)
        self.actions = np.zeros(18)
        self.history = np.zeros((1, 1))
        self.command = [0.] * 6
        self.gait_indices = 0.
        self.reference = None
        self.reference_time = 0.
        self.base = self.model.body("base_link").id
        self.ee = self.model.site("x5_ee").id

    def reset_policy(self):
        self.actions.fill(0.)

    def set_reference(self, reference, time_s):
        self.reference, self.reference_time = reference, time_s

    def read_state(self):
        q, dq = self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()
        quat = self.data.qpos[[4, 5, 6, 3]].copy()
        rot = self.data.xmat[self.base].reshape(3, 3)
        return q, dq, quat, rot.T @ self.data.qvel[3:6], self.data.qpos[:3].copy(), rot.T @ self.data.qvel[:3]

    def forward(self, q, dq, quat, gyro, base_pos, lin_vel):
        del base_pos, lin_vel
        if self.reference is None:
            raise RuntimeError("UMI reference was not set")
        base_rot = _quat_matrix(quat)
        state = np.r_[gyro * .25, base_rot.T @ [0., 0., -1.],
                      q - self.default_dof_pos, dq * .05]
        ee_rot = self.data.site_xmat[self.ee].reshape(3, 3)
        ee_pos = self.data.site_xpos[self.ee]
        task = []
        for offset in (.02, .04, .06, 1.0):
            _arc, pos, target_q = self.reference.at(self.reference_time + offset)
            relative_rot = ee_rot.T @ _quat_matrix(target_q)
            # pytorch3d matrix_to_rotation_6d: first two matrix rows.
            task.extend(((ee_rot.T @ (pos - ee_pos)) * 3.).tolist())
            task.extend((relative_rot[:2].reshape(-1) * 1.5).tolist())
        obs = np.r_[state, task, self.actions].astype(np.float32)
        if obs.shape != (96,) or not np.isfinite(obs).all():
            raise ValueError(f"invalid UMI observation {obs.shape}")
        with torch.inference_mode():
            self.actions = self.actor(torch.from_numpy(obs[None]))[0].numpy()
        return self.actions.copy()

    def compute_output(self, actions):
        return self.default_dof_pos + .25 * np.clip(actions, -100., 100.)
