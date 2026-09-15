"""Client for WB-LocoMan's torque sidecar on the common MuJoCo plant."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq


JOINT_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
] + [f"x5_joint{i}" for i in range(1, 7)]
PROTOCOL = "common-mujoco-controller-v1"


class WbLocomanMujoco:
    direct_torque = True

    def __init__(self, sidecar_root, python, scene, output_dir):
        self.root = Path(sidecar_root).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = 0.0025
        self.data = mujoco.MjData(self.model)
        self.n = 18
        joint_ids = np.asarray([self.model.joint(name).id for name in JOINT_NAMES])
        self.qpos_adr = self.model.jnt_qposadr[joint_ids].astype(int)
        self.dof_adr = self.model.jnt_dofadr[joint_ids].astype(int)
        self.mapping = []
        for joint in joint_ids:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError(f"expected one actuator for joint id {joint}")
            self.mapping.append(int(matches[0]))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.endpoint = f"tcp://127.0.0.1:{port}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(output_dir) / "sidecar.log"
        self.log_stream = self.log_path.open("w")
        self.process = subprocess.Popen(
            [str(python), str(self.root / "benchmark_sidecar.py"),
             "--endpoint", self.endpoint],
            cwd=self.root, stdout=self.log_stream, stderr=subprocess.STDOUT,
            text=True,
        )
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)
        description = self._wait_describe()
        self.control_dt = float(description["control_dt_s"])
        self.policy_dt = self.control_dt
        self.substeps = max(1, round(self.policy_dt / self.model.opt.timestep))
        limits = self.model.actuator_ctrlrange[self.mapping]
        self.p = {
            "decimation": 1, "rl_kp": [0.0] * 18, "rl_kd": [0.0] * 18,
            "torque_limits": np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1])).tolist(),
        }
        self.default_dof_pos = np.zeros(18)
        self.history = np.zeros((1, 1))
        self.actions = np.zeros(18)
        self.command = [0.0] * 6
        self.gait_indices = 0.0
        self.episode = 0
        self.horizon_offsets = []
        self.reference_payload = None
        self.base = self.model.body("base_link").id
        self.ee = self.model.site("x5_ee").id
        self.foot_geoms = [self.model.geom(name).id for name in ("FL", "FR", "RL", "RR")]

    def _request(self, payload, timeout_ms=30000):
        self.socket.send_json({"protocol": PROTOCOL, **payload})
        if not self.socket.poll(timeout_ms):
            raise TimeoutError(f"WB-LocoMan sidecar timeout during {payload['op']}")
        reply = self.socket.recv_json()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "WB-LocoMan sidecar error"))
        return reply

    def _wait_describe(self):
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.log_path.read_text()[-8000:]
                raise RuntimeError(f"WB-LocoMan sidecar exited {self.process.returncode}: {output}")
            try:
                return self._request({"op": "describe"}, 1000)
            except TimeoutError:
                self.socket.close(linger=0)
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.connect(self.endpoint)
        raise TimeoutError("WB-LocoMan sidecar did not become ready")

    def reset_policy(self):
        self.episode += 1
        reply = self._request({"op": "reset", "episode_id": self.episode}, 90000)
        self.horizon_offsets = reply["horizon_offsets_s"]

    def set_reference(self, reference, time_s):
        times = np.asarray(self.horizon_offsets) + time_s
        poses = [reference.at(float(value)) for value in times]
        self.reference_payload = {
            "offsets_s": self.horizon_offsets,
            "positions_m": [value[1].tolist() for value in poses],
            "quaternions_xyzw": [value[2].tolist() for value in poses],
        }

    def _contacts(self):
        active = [False] * 4
        lookup = {geom: i for i, geom in enumerate(self.foot_geoms)}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if int(contact.geom1) in lookup:
                active[lookup[int(contact.geom1)]] = True
            if int(contact.geom2) in lookup:
                active[lookup[int(contact.geom2)]] = True
        return active

    def read_state(self):
        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()
        quat = self.data.qpos[[4, 5, 6, 3]].copy()
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return q, dq, quat, self.data.qvel[3:6].copy(), self.data.qpos[:3].copy(), rotation.T @ self.data.qvel[:3]

    def forward(self, q, dq, quat, gyro, base_pos, lin_vel):
        if self.reference_payload is None:
            raise RuntimeError("WB-LocoMan reference was not set")
        reply = self._request({
            "op": "step", "episode_id": self.episode, "time_s": float(self.data.time),
            "state": {"base_pos": base_pos.tolist(), "base_quat_xyzw": quat.tolist(),
                      "base_lin_vel_body": lin_vel.tolist(), "base_ang_vel_body": gyro.tolist(),
                      "q": q.tolist(), "dq": dq.tolist(), "contacts": self._contacts(),
                      "ee_pos": self.data.site_xpos[self.ee].tolist(),
                      "ee_quat_xyzw": _site_quaternion(self.model, self.data, self.ee).tolist()},
            "reference": self.reference_payload,
        }, 90000)
        self.actions = np.asarray(reply["torque_nm"], dtype=np.float64)
        self.feedback = {key: np.asarray(value, dtype=np.float64)
                         for key, value in reply["joint_feedback"].items()}
        self.diagnostics = reply["diagnostics"]
        return self.actions.copy()

    def compute_torque(self, q, dq):
        # Native main.py evaluates this feedback law every physics tick, not
        # just when a new MPC solution arrives. All 18 joints are bounded revolutes.
        f = self.feedback
        return f["feedforward_nm"] + f["kp"] * (f["q_rad"] - q) + f["kd"] * (f["dq_rad_s"] - dq)

    def compute_output(self, actions):
        del actions
        return self.feedback["q_rad"].copy()

    def close(self):
        try:
            if self.process.poll() is None:
                self._request({"op": "close"}, 3000)
        except Exception:
            pass
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.socket.close(linger=0)
        self.context.term()
        self.log_stream.close()


def _site_quaternion(model, data, site):
    del model
    wxyz = np.empty(4)
    mujoco.mju_mat2Quat(wxyz, data.site_xmat[site])
    return np.roll(wxyz, -1)
