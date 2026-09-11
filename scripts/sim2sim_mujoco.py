"""Run the exported rl_sar bundle in MuJoCo -- a Python mirror of rl_sar's loop.

This executes exactly what `rl_sim_mujoco go2_x5 scene` does, using the same
policy.pt, base.yaml and config.yaml, but in Python so the pipeline can be
validated before building the C++ (which needs libtorch, MuJoCo headers, GLFW
and TBB). It mirrors, function for function:

    RL_Sim::GetState()        -> read_state()
    RL::ComputeObservation()  -> RlSarObservation.assemble() from
                                 scripts/rl_sar_obs.py -- the same port
                                 verify_rl_sar_obs.py checks against training
    RL::Forward()             -> history buffer + TorchScript forward
    RL::ComputeOutput()       -> compute_output()
    RL_Sim::SetCommand()      -> write_command()
    RLFSMStateGetUp / RLFSMStateRLLocomotion -> the phase machine in run()

If this stands and walks, the exported bundle and the MJCF are sound and the
only thing the C++ build adds is the C++ itself.

Usage::

    python scripts/sim2sim_mujoco.py \
        --robot_dir <rl_sar>/policy/go2_x5 --config_name roboduet_stage1 \
        --scene <rl_sar>/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml \
        --seconds 10 --vx 0.5
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rl_sar_obs import RlSarObservation, quat_rotate_inverse_np, effective_gait_frequency  # noqa: E402


class RlSarMujoco:
    """One rl_sar robot instance driving one MuJoCo model."""

    def __init__(self, robot_dir, config_name, scene, seed=0):
        robot_dir = Path(robot_dir)
        robot = robot_dir.name
        base = yaml.safe_load((robot_dir / "base.yaml").read_text())[robot]
        cfg = yaml.safe_load((robot_dir / config_name / "config.yaml").read_text())[
            f"{robot}/{config_name}"
        ]
        # rl_sar reads base.yaml at construction and config.yaml on entering the
        # RL state, with later keys overwriting earlier ones.
        self.p = {**base, **cfg}
        self.obs_builder = RlSarObservation(self.p)

        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.policy = torch.jit.load(str(robot_dir / config_name / "policy.pt"))
        self.policy.eval()

        self.n = int(self.p["num_of_dofs"])
        self.mapping = list(self.p["joint_mapping"])
        self.default_dof_pos = np.array(self.p["default_dof_pos"], dtype=np.float64)
        self.control_dt = float(self.p["dt"])
        self.policy_dt = self.control_dt * int(self.p["decimation"])
        self.substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        self.history_index = list(self.p["observations_history"])
        self.history_length = max(self.history_index) + 1
        self.obs_width = int(self.p["num_observations"])
        self.history = np.zeros((self.history_length, self.obs_width))

        # RL::obs / RL::Control
        self.actions = np.zeros(self.n)
        self.gait_indices = 0.0
        self.command = [0.0] * 6
        self.output_dof_pos = self.default_dof_pos.copy()

    # ---- RL_Sim::GetState -------------------------------------------------
    def read_state(self):
        s, n = self.data.sensordata, self.n
        q = np.array([s[self.mapping[i]] for i in range(n)])
        dq = np.array([s[self.mapping[i] + n] for i in range(n)])
        quat = np.array(s[3 * n: 3 * n + 4])          # w, x, y, z
        gyro = np.array(s[3 * n + 4: 3 * n + 7])
        base_pos = np.array(s[3 * n + 10: 3 * n + 13])
        lin_vel_world = np.array(s[3 * n + 13: 3 * n + 16])
        # QuatRotateInverse in rl_sar takes w,x,y,z; the numpy helper shared with
        # verify_rl_sar_obs.py takes x,y,z,w, hence the roll.
        quat_xyzw = np.roll(quat, -1)
        lin_vel_body = quat_rotate_inverse_np(quat_xyzw, lin_vel_world)
        return q, dq, quat_xyzw, gyro, base_pos, lin_vel_body

    # ---- RL::ComputeObservation + RL::Forward ------------------------------
    def forward(self, q, dq, quat_xyzw, gyro, base_pos, lin_vel_body):
        state = {
            "quat": quat_xyzw, "dof_pos": q, "dof_vel": dq,
            "actions": self.actions, "ang_vel": gyro, "lin_vel": lin_vel_body,
            "base_height": float(base_pos[2]),
            "cmd_x": self.command[0], "cmd_y": self.command[1], "cmd_yaw": self.command[2],
            "cmd_pitch": self.command[3], "cmd_roll": self.command[4],
            "cmd_height": self.command[5],
            "gait_indices": self.gait_indices,
        }
        # ComputeObservation() advances the gait clock as a side effect, in the
        # same call that reads the commands -- matching training, where
        # _step_contact_targets() and compute_observations() see one command.
        self.gait_indices = np.fmod(
            self.gait_indices + self.policy_dt * effective_gait_frequency(self.p, self.command[:3]), 1.0)
        state["gait_indices"] = self.gait_indices

        obs = self.obs_builder.assemble(state)
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = obs
        # observations_history counts down to 0 (0 = newest), so this lays the
        # frames out oldest-first exactly like RoboDuet's HistoryWrapper.
        flat = np.concatenate([self.history[i] for i in self.history_index])

        with torch.no_grad():
            actions = self.policy(torch.tensor(flat, dtype=torch.float32).unsqueeze(0))
        actions = actions.squeeze(0).numpy().astype(np.float64)
        actions = np.clip(actions,
                          self.p["clip_actions_lower"][:len(actions)],
                          self.p["clip_actions_upper"][:len(actions)])
        # rl_sar zero-pads the policy's actions up to num_of_dofs.
        self.actions = np.concatenate([actions, np.zeros(self.n - len(actions))])
        return self.actions

    # ---- RL::ComputeOutput ------------------------------------------------
    def compute_output(self, actions):
        return actions * np.array(self.p["action_scale"]) + self.default_dof_pos

    # ---- RL_Sim::SetCommand ----------------------------------------------
    def write_command(self, q_des, q, dq, kp, kd):
        tau = kp * (q_des - q) - kd * dq
        for i in range(self.n):
            self.data.ctrl[self.mapping[i]] = tau[i]

    def run(self, seconds, command, getup_seconds=2.0, on_step=None):
        steps = int(seconds / self.control_dt)
        getup_steps = int(getup_seconds / self.control_dt)
        fixed_kp = np.array(self.p["fixed_kp"])
        fixed_kd = np.array(self.p["fixed_kd"])
        rl_kp = np.array(self.p["rl_kp"])
        rl_kd = np.array(self.p["rl_kd"])
        decimation = int(self.p["decimation"])

        for step in range(steps):
            q, dq, quat, gyro, base_pos, lin_vel = self.read_state()

            if step < getup_steps:
                # RLFSMStateGetUp: hold default_dof_pos with the fixed gains.
                self.write_command(self.default_dof_pos, q, dq, fixed_kp, fixed_kd)
            else:
                if step == getup_steps:
                    # RLFSMStateRLLocomotion::Enter zeroes the gait clock.
                    self.gait_indices = 0.0
                    self.command = list(command)
                if (step - getup_steps) % decimation == 0:
                    actions = self.forward(q, dq, quat, gyro, base_pos, lin_vel)
                    self.output_dof_pos = self.compute_output(actions)
                self.write_command(self.output_dof_pos, q, dq, rl_kp, rl_kd)

            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)

            if on_step is not None:
                on_step(step, self)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot_dir", type=str, required=True,
                        help="<rl_sar>/policy/<robot>")
    parser.add_argument("--config_name", type=str, default="roboduet_stage1")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--getup_seconds", type=float, default=2.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--body_pitch", type=float, default=0.0)
    parser.add_argument("--body_roll", type=float, default=0.0)
    parser.add_argument("--body_height", type=float, default=0.0)
    args = parser.parse_args()

    sim = RlSarMujoco(args.robot_dir, args.config_name, args.scene)
    command = [args.vx, args.vy, args.yaw, args.body_pitch, args.body_roll, args.body_height]

    trace = []

    def record(step, s):
        n = s.n
        trace.append((
            float(s.data.sensordata[3 * n + 12]),                          # height
            np.array(s.data.sensordata[3 * n + 13: 3 * n + 16]).copy(),    # world lin vel
            np.array(s.data.sensordata[3 * n: 3 * n + 4]).copy(),          # quat
            float(np.max(np.abs(s.actions))),
            float(s.data.sensordata[3 * n + 6]),                           # body yaw rate
        ))

    sim.run(args.seconds, command, getup_seconds=args.getup_seconds, on_step=record)

    heights = np.array([t[0] for t in trace])
    vels = np.stack([t[1] for t in trace])
    quats = np.stack([t[2] for t in trace])
    peak_action = max(t[3] for t in trace)
    rl_from = int(args.getup_seconds / sim.control_dt)
    last = slice(-int(1.0 / sim.control_dt), None)

    # Upright-ness: the body z axis projected on world z, from the w,x,y,z quat.
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    upright = 1 - 2 * (x * x + y * y)

    print(f"\ncommand: vx={args.vx} vy={args.vy} yaw={args.yaw} "
          f"pitch={args.body_pitch} roll={args.body_roll} height={args.body_height}")
    print(f"  height        mean {heights[rl_from:].mean():.4f} m   "
          f"min {heights[rl_from:].min():.4f}   last {heights[-1]:.4f}")
    print(f"  upright       min {upright[rl_from:].min():.4f}  (1.0 = level, <0 = flipped)")
    yaw_rate = np.array([t[4] for t in trace])
    # Body-frame forward speed: what the vx command actually asks for. World vx
    # alone is misleading as soon as the robot is also turning.
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward = vels[:, 0] * np.cos(yaw) + vels[:, 1] * np.sin(yaw)
    lateral = -vels[:, 0] * np.sin(yaw) + vels[:, 1] * np.cos(yaw)

    print(f"  body vx       mean over last 1 s {forward[last].mean():+.4f} m/s "
          f"(commanded {args.vx:+.2f})")
    print(f"  body vy       mean over last 1 s {lateral[last].mean():+.4f} m/s "
          f"(commanded {args.vy:+.2f})")
    print(f"  yaw rate      mean over last 1 s {yaw_rate[last].mean():+.4f} rad/s "
          f"(commanded {args.yaw:+.2f})")
    print(f"  peak |action| {peak_action:.3f}  (clip at {sim.p['clip_actions_upper'][0]})")

    fell = heights[rl_from:].min() < 0.15 or upright[rl_from:].min() < 0.5
    print(f"\n  {'FELL OVER' if fell else 'stayed up'}")
    return 1 if fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
