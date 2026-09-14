"""Adapt the deployed Ma2022 recurrent student to the common Go2+X5 plant."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


LEG_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
ARM_NAMES = [f"x5_joint{i}" for i in range(1, 7)]


def _load_deployment_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("ma2022_deployment_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Ma2022 adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ma2022Mujoco:
    """Expose Ma2022 through the small interface used by ``wbc.mujoco``."""

    def __init__(self, deployment_root, policy, env_config, scene, config):
        adapter = _load_deployment_adapter(Path(deployment_root) / "adapter.py")
        self.student = adapter.Student(policy, env_config)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(config.get("physics_dt", 0.0025))
        self.n = 18
        joint_names = LEG_NAMES + ARM_NAMES
        joint_ids = np.asarray([self.model.joint(name).id for name in joint_names])
        self.qpos_adr = self.model.jnt_qposadr[joint_ids].astype(int)
        self.dof_adr = self.model.jnt_dofadr[joint_ids].astype(int)
        mapping = []
        for joint in joint_ids:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError(f"expected one actuator for joint id {joint}")
            mapping.append(int(matches[0]))
        self.mapping = mapping
        self.control_dt = self.student.dt
        self.policy_dt = self.student.dt
        self.substeps = max(1, round(self.policy_dt / self.model.opt.timestep))
        arm_home = np.asarray(config["arm_home"], dtype=np.float64)
        self.default_dof_pos = np.r_[self.student.defaults, arm_home]
        limits = self.model.actuator_ctrlrange[self.mapping]
        self.p = {
            "decimation": 1,
            "rl_kp": [self.student.kp] * 12 + list(config["arm_kp"]),
            "rl_kd": [self.student.kd] * 12 + list(config["arm_kd"]),
            "torque_limits": np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1])).tolist(),
        }
        self.history = np.zeros((1, 1), dtype=np.float64)
        self.actions = np.zeros(18, dtype=np.float64)
        self.command = [0.0] * 6
        self.gait_indices = 0.0
        self.reaction = adapter.ArmReaction(self.model)
        self.adapter_source = Path(deployment_root) / "adapter.py"
        self.policy_path = Path(policy)
        self.env_config_path = Path(env_config)
        self.config_path = Path(config.pop("_config_path"))

    def reset_policy(self):
        self.student.reset()

    def read_state(self):
        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()
        quat_xyzw = self.data.qpos[[4, 5, 6, 3]].copy()
        rotation = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
        lin_body = rotation.T @ self.data.qvel[:3]
        return q, dq, quat_xyzw, self.data.qvel[3:6].copy(), self.data.qpos[:3].copy(), lin_body

    def forward(self, q, dq, quat_xyzw, gyro, base_pos, lin_vel_body):
        del quat_xyzw
        rotation = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
        arm_q = np.tile(q[12:18], (5, 1))
        zeros = np.zeros_like(arm_q)
        prediction = self.reaction.predict(self.data, arm_q, zeros, zeros)
        values, _clipped = self.student.inputs(
            lin_vel_body, gyro, rotation.T @ np.array([0.0, 0.0, -1.0]),
            np.asarray(self.command[:3]), q[:12], dq[:12], base_pos[2], prediction,
        )
        leg_actions, _reset = self.student.infer(values)
        self.actions = np.r_[leg_actions, np.zeros(6)]
        return self.actions.copy()

    def compute_output(self, actions):
        output = self.default_dof_pos.copy()
        output[:12] = self.student.defaults + 0.25 * actions[:12]
        return output


def load_config(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text())
    value["_config_path"] = str(path)
    return value
