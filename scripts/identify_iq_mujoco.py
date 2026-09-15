"""Reproducible I_Q command-response identification using the deployed bundle.

No PPO, policy changes, or external processes are needed for data collection.
All samples use MuJoCo state at a policy boundary, then the recorded command
is held over the following 20 ms. Failed episodes are retained in full.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim2sim_mujoco import RlSarMujoco
from rl_sar_obs import quat_to_euler_np

STACK = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
CHANNELS = ["vx", "vy", "wz", "height", "pitch", "roll"]
AMPLITUDES = np.array([.55, .28, .65, .035, .20, .14])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    episodes = []
    # IDs, excitation RNG, initial states, and arm motions are frozen before
    # fitting. Each split has fresh excitation and initial-state seeds.
    for split, seed in [("train", 6101), ("development", 7101), ("test", 8101)]:
        for channel in range(6):
            waves = ["steps", "chirp"] if split == "train" else ["prbs"]
            for wave in waves:
                for walking in ([False, True] if split == "train" else [True]):
                    episodes.append(dict(split=split, channel=channel, waveform=wave,
                        walking=walking, arm="fixed", duration_s=20., seed=seed + len(episodes)))
        velocities = [[.2, 0, 0], [.4, 0, 0], [.6, 0, 0], [-.3, 0, 0],
                      [0, .25, 0], [0, 0, .5]]
        for velocity in velocities:
            for arm in ["fixed", "moving"]:
                episodes.append(dict(split=split, channel=-1, waveform="steady",
                    velocity=velocity, walking=True, arm=arm, duration_s=20., seed=seed + len(episodes)))
    for index, episode in enumerate(episodes):
        episode["id"] = f"{episode['split']}_{index:03d}"
    return dict(schema="iq_mujoco_identification_v1", channels=CHANNELS,
        policy="I_Q", checkpoint_iteration=32498, policy_dt_s=.02, control_dt_s=.005,
        physics_dt_s=.0025, integrator="implicitfast", gait_frequency_hz=2.75,
        warmup_s=3., frame="heading vx/vy at trunk, body wz, arm-mount world z, ZYX pitch/roll",
        command_order="vx,vy,wz,height_offset,pitch,roll",
        nominal_model="six independent first-order gain/tau/bias, no assumed pure delay",
        residual_model="(a0+a1*speed)*sin(harmonic*phase+offset), z/pitch/roll",
        harmonic_candidates=[1, 2, 3, 4], prediction_horizons_s=[.1, .3, .6, 1.0],
        selection="train fits; development chooses residual harmonics; test is final only",
        failure="nonfinite state/action, trunk z<0.15 m, abs roll/pitch>0.85 rad, MuJoCo warning",
        scope="flat-ground native MJCF, fixed gait; moving-arm holdout; not hardware evidence",
        episodes=episodes)


def commands(episode, dt=.02):
    t = np.arange(round(episode["duration_s"] / dt) + 1) * dt
    u = np.zeros((len(t), 6))
    if episode["waveform"] == "steady":
        u[:, :3] = episode["velocity"]
    else:
        c = episode["channel"]
        if episode["walking"]:
            u[:, 0] = .35
        rng = np.random.default_rng(episode["seed"])
        if episode["waveform"] == "steps":
            signal = np.array([0., .5, -.5, 1., -1., .7, -.7, 0.])
            signal = signal[np.minimum((t / 2.5).astype(int), 7)]
        elif episode["waveform"] == "chirp":
            phase = rng.uniform(-np.pi, np.pi)
            signal = np.sin(2*np.pi*(.06*t + .5*(1.4-.06)/20*t*t) + phase)
        else:
            signal = rng.uniform(-1., 1., 20)[np.minimum((t / 1.25).astype(int), 19)]
        u[:, c] = AMPLITUDES[c] * signal
        if c == 0 and episode["walking"]:
            u[:, 0] = .35 + .23 * signal
    return t, u


class IdentificationPlant:
    def __init__(self, robot_dir, scene, seed):
        self.sim = RlSarMujoco(robot_dir, "I_Q", scene, seed=seed)
        s = self.sim
        s.model.opt.timestep = .0025
        s.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        s.substeps = 2
        if s.obs_width != 90 or s.history_length != 30 or abs(s.policy_dt-.02) > 1e-10:
            raise ValueError("I_Q contract must be 90x30, 50 Hz")
        self.base = s.model.body("base_link").id
        self.mount = s.model.body("x5_base_link").id
        self.ee = s.model.site("x5_ee").id
        self.kp, self.kd = np.asarray(s.p["rl_kp"]), np.asarray(s.p["rl_kd"])
        limits = s.model.actuator_ctrlrange[s.mapping]
        self.torque_limits = np.minimum(np.asarray(s.p["torque_limits"]), np.max(np.abs(limits), axis=1))
        self.q_target = s.default_dof_pos.copy()
        self.arm_target = np.array([0., .6, .6, 0., 0., 0.])
        self.arm_position = self.arm_target.copy()
        joint_ids = s.model.actuator_trnid[s.mapping, 0]
        self.qadr = s.model.jnt_qposadr[joint_ids]
        self.vadr = s.model.jnt_dofadr[joint_ids]
        rng = np.random.default_rng(seed)
        s.data.qpos[:2] += rng.uniform(-.005, .005, 2)
        s.data.qpos[self.qadr] = s.default_dof_pos
        s.data.qpos[self.qadr[12:]] = self.arm_target
        mujoco.mj_forward(s.model, s.data)
        for _ in range(50):
            self.step(np.zeros(6), policy=False)
        for _ in range(100):
            self.step(np.zeros(6))

    def snapshot(self):
        s = self.sim
        q, dq, quat, gyro, sensor_pos, sensor_vel = s.read_state()
        rpy = quat_to_euler_np(quat)
        # mj_jacBody is explicitly the body-frame origin. mj_objectVelocity's
        # body reference can be the inertial/COM frame and is not interchangeable.
        jacp, jacr = np.zeros((3, s.model.nv)), np.zeros((3, s.model.nv))
        mujoco.mj_jacBody(s.model, s.data, jacp, jacr, self.base)
        spatial = np.r_[jacr @ s.data.qvel, jacp @ s.data.qvel]
        c, sn = np.cos(rpy[2]), np.sin(rpy[2])
        heading_vel = np.array([c*spatial[3] + sn*spatial[4], -sn*spatial[3] + c*spatial[4]])
        body_vel = s.data.xmat[self.base].reshape(3, 3).T @ spatial[3:]
        y = np.r_[heading_vel, gyro[2], s.data.xpos[self.mount, 2], rpy[1], rpy[0]]
        return dict(y=y, phase=float(s.gait_indices*2*np.pi), q=q, dq=dq,
                    base_position=s.data.xpos[self.base].copy(), quaternion=quat,
                    body_velocity=body_vel, gyro=gyro, ee=s.data.site_xpos[self.ee].copy(),
                    ee_rotation=s.data.site_xmat[self.ee].copy(), actions=s.actions[:12].copy())

    def step(self, command, arm=None, arm_dq=None, policy=True):
        s = self.sim
        s.command = np.asarray(command)[[0, 1, 2, 4, 5, 3]].tolist()
        if arm is not None:
            self.arm_target = np.asarray(arm).copy()
        q, dq, quat, gyro, base_pos, lin_vel = s.read_state()
        if policy:
            action = s.forward(q, dq, quat, gyro, base_pos, lin_vel)
            if not np.isfinite(action).all():
                raise FloatingPointError("nonfinite policy action")
            self.q_target = s.compute_output(action)
        velocity_target = np.zeros(s.n)
        if arm_dq is not None:
            velocity_target[12:] = np.clip(arm_dq, -1.5, 1.5)
        for _ in range(4):
            self.arm_position += np.clip(self.arm_target-self.arm_position, -.0075, .0075)
            self.q_target[12:] = self.arm_position
            for _ in range(s.substeps):
                q_now = s.data.qpos[self.qadr]
                dq_now = s.data.qvel[self.vadr]
                torque = self.kp*(self.q_target-q_now) + self.kd*(velocity_target-dq_now)
                s.data.ctrl[s.mapping] = np.clip(torque, -self.torque_limits, self.torque_limits)
                mujoco.mj_step(s.model, s.data)
        mujoco.mj_forward(s.model, s.data)

    def failure(self):
        s = self.sim
        if not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all():
            return "nonfinite"
        z = s.data.xpos[self.base, 2]
        rpy = quat_to_euler_np(s.read_state()[2])
        if z < .15:
            return "height"
        if np.max(np.abs(rpy[:2])) > .85:
            return "tilt"
        if np.any(s.data.warning.number):
            return "mujoco_warning"
        return None


def collect(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "protocol.json"
    if not spec_path.exists():
        spec = json.loads(Path(args.protocol_file).read_text()) if args.protocol_file else protocol()
        files = [Path(__file__), ROOT/"scripts/sim2sim_mujoco.py", ROOT/"scripts/rl_sar_obs.py",
                 Path(args.robot_dir)/"base.yaml", Path(args.robot_dir)/"I_Q/config.yaml",
                 Path(args.robot_dir)/"I_Q/policy.pt", STACK/"overleaf/3method.tex"]
        files += sorted(Path(args.scene).parent.glob("*.xml"))
        write_json(out/"input_manifest.json", dict(files={str(p): sha(p) for p in files},
            python=sys.version, mujoco=mujoco.__version__, torch=torch.__version__))
        write_json(spec_path, spec)
        (out/"source").mkdir(exist_ok=True)
        for p in files[:3]:
            shutil.copy2(p, out/"source"/p.name)
    spec = json.loads(spec_path.read_text())
    if args.protocol_file and spec != json.loads(Path(args.protocol_file).read_text()):
        raise ValueError("existing collection protocol differs from the requested protocol")
    manifest = json.loads((out/"input_manifest.json").read_text())
    for path, digest in manifest["files"].items():
        if sha(path) != digest:
            raise ValueError(f"collection input changed; use a new output root: {path}")
    episodes = [e for e in spec["episodes"] if args.split == "all" or e["split"] == args.split]
    if args.limit:
        episodes = episodes[:args.limit]
    torch.set_num_threads(1)
    for episode in episodes:
        dest = out/"raw"/episode["id"]
        if dest.with_suffix(".json").exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        plant = IdentificationPlant(args.robot_dir, args.scene, episode["seed"])
        # Independent held-out oscillator starts; the standing observation
        # clock is phase-independent, so its warmup history remains valid.
        plant.sim.gait_indices = episode.get("initial_phase_rad", 0.)/(2*np.pi)
        t, u = commands(episode)
        samples = []
        fail = plant.failure()
        for k in range(len(t)):
            samples.append(plant.snapshot())
            if fail or k == len(t)-1:
                break
            arm_time = t[k]*episode.get("arm_frequency_scale", 1.)
            arm_phase = episode.get("arm_phase_rad", 0.)
            arm = plant.arm_target if episode["arm"] == "fixed" else np.array([
                .15*np.sin(.7*arm_time+arm_phase), .6+.2*np.sin(.9*arm_time+arm_phase),
                .6+.2*np.cos(.6*arm_time+arm_phase), .12*np.sin(.8*arm_time+arm_phase),
                .10*np.cos(.7*arm_time+arm_phase), 0.])
            try:
                plant.step(u[k], arm)
                fail = plant.failure()
            except FloatingPointError as error:
                fail = str(error)
        arrays = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
        arrays.update(t=t[:len(samples)], u=u[:len(samples)])
        np.savez_compressed(dest.with_suffix(".npz"), **arrays)
        report = dict(**episode, success=fail is None, failure=fail, samples=len(samples),
            simulated_seconds=float(t[len(samples)-1]), wall_seconds=time.monotonic()-begin,
            sha256=sha(dest.with_suffix(".npz")), warning_counts=plant.sim.data.warning.number.tolist())
        write_json(dest.with_suffix(".json"), report)
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect"])
    parser.add_argument("--output", default=str(ROOT/"tmp/experiments/20260915_iq_identification"))
    parser.add_argument("--robot-dir", default=str(STACK/"rl_sar/policy/go2_x5"))
    parser.add_argument("--scene", default=str(STACK/"rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"))
    parser.add_argument("--split", choices=["all", "train", "development", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--protocol-file", help="Explicit frozen excitation manifest for independent validation")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
