"""rl_sar's observation assembly for RoboDuet policies, ported to Python.

This is a line-for-line mirror of the `roboduet/*` branches in rl_sar's
`RL::ComputeObservation()` (library/core/rl_sdk/rl_sdk.cpp). It is deliberately
shared by:

  * scripts/verify_rl_sar_obs.py -- diffs it against the real
    WBCEnv.get_dog_observations() inside IsaacGym
  * scripts/sim2sim_mujoco.py    -- runs it against MuJoCo

so the thing proven correct against training is the same code that drives the
simulation. Keep it free of IsaacGym imports: sim2sim must run without it.
"""

import numpy as np


def quat_rotate_inverse_np(quat_xyzw, vec):
    """IsaacGym's quat_rotate_inverse, matching rl_sar's QuatRotateInverse."""
    x, y, z, w = quat_xyzw
    a = vec * (2.0 * w * w - 1.0)
    b = np.cross(np.array([x, y, z]), vec) * w * 2.0
    c = np.array([x, y, z]) * np.dot(np.array([x, y, z]), vec) * 2.0
    return a - b + c


def quat_to_euler_np(quat_xyzw):
    """[roll, pitch, yaw]; identical formula to rl_sar's QuaternionToEuler."""
    x, y, z, w = quat_xyzw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class RlSarObservation:
    """Python port of rl_sar's RL::ComputeObservation() for the roboduet terms.

    Reads exactly the keys the C++ reads, from the exported config.yaml, so a
    typo in the generated config shows up here rather than on the robot.
    """

    def __init__(self, params):
        self.p = params
        self.num_leg = int(params["num_leg_dofs"])
        self.num_arm = int(params["num_arm_dofs"])
        self.default_dof_pos = np.array(params["default_dof_pos"], dtype=np.float64)
        # Compact bundles omit disabled terms entirely. Legacy bundles keep
        # their trained zero slots and are handled by term() below.
        if int(params.get("dog_observation_layout_version", 1)) == 2:
            switches = {
                "observe_clock_inputs": ("roboduet/clock_inputs",),
                "observe_lin_vel": ("roboduet/base_lin_vel",),
                "observe_pose_actual": ("roboduet/body_pose_actual",),
                "observe_track_error": ("roboduet/body_pose_error", "roboduet/velocity_error"),
            }
            for switch, terms in switches.items():
                for term in terms:
                    if (term in params["observations"]) != bool(params[switch]):
                        raise ValueError(f"{switch} disagrees with observations term {term}")
        width = sum(self.widths()[name] for name in params["observations"])
        if width != int(params["num_observations"]):
            raise ValueError(f"Observation terms have width {width}, expected {params['num_observations']}")


    def widths(self):
        p = self.p
        return {
            "gravity_vec": 3,
            "ang_vel": 3,
            "lin_vel": 3,
            "roboduet/leg_dof_pos": self.num_leg,
            "roboduet/leg_dof_vel": self.num_leg,
            "roboduet/leg_actions": self.num_leg,
            "roboduet/dog_commands": len(p["dog_commands_scale"]),
            "roboduet/arm_commands": int(p["arm_num_commands"]),
            "roboduet/clock_inputs": 4,
            "roboduet/base_lin_vel": 3,
            "roboduet/body_pose_actual": 3,
            "roboduet/body_pose_error": 3,
            "roboduet/velocity_error": 3,
            "roboduet/arm_dof_pos": self.num_arm,
            "roboduet/arm_dof_vel": self.num_arm,
        }

    def term(self, name, s):
        """s: dict of raw quantities rl_sar would have available."""
        p = self.p
        num_leg, num_arm = self.num_leg, self.num_arm

        if name == "gravity_vec":
            return quat_rotate_inverse_np(s["quat"], np.array([0.0, 0.0, -1.0]))
        if name == "ang_vel":
            return s["ang_vel"] * p["ang_vel_scale"]
        if name == "roboduet/leg_dof_pos":
            return (s["dof_pos"][:num_leg] - self.default_dof_pos[:num_leg]) * p["dof_pos_scale"]
        if name == "roboduet/leg_dof_vel":
            return s["dof_vel"][:num_leg] * p["dof_vel_scale"]
        if name == "roboduet/leg_actions":
            return s["actions"][:num_leg]
        if name == "roboduet/arm_dof_pos":
            sl = slice(num_leg, num_leg + num_arm)
            return (s["dof_pos"][sl] - self.default_dof_pos[sl]) * p["dof_pos_scale"]
        if name == "roboduet/arm_dof_vel":
            return s["dof_vel"][num_leg:num_leg + num_arm] * p["dof_vel_scale"]
        if name == "roboduet/dog_commands":
            cmd = np.concatenate([
                [s["cmd_x"], s["cmd_y"], s["cmd_yaw"],
                 s["cmd_pitch"], s["cmd_roll"], s["cmd_height"]],
                np.array(p["dog_commands_extra"], dtype=np.float64),
            ])
            return cmd * np.array(p["dog_commands_scale"], dtype=np.float64)
        if name == "roboduet/arm_commands":
            return np.zeros(int(p["arm_num_commands"]))
        if name == "roboduet/clock_inputs":
            gait_phases = p["gait_phases"]
            phases, offsets, bounds = (
                [gait_phases[key] for key in ("phases", "offsets", "bounds")]
                if isinstance(gait_phases, dict) else gait_phases
            )
            duration = float(p["gait_duration"])
            gait = s["gait_indices"]
            foot = [gait + phases + offsets + bounds, gait + offsets, gait + bounds, gait + phases]
            standing = np.linalg.norm([s["cmd_x"], s["cmd_y"], s["cmd_yaw"]]) < 0.1
            out = np.zeros(4)
            for i in range(4):
                idx = 0.25 if standing else np.fmod(foot[i], 1.0)
                if idx < 0.0:
                    idx += 1.0
                idx = (idx * (0.5 / duration) if idx < duration
                       else 0.5 + (idx - duration) * (0.5 / (1.0 - duration)))
                out[i] = np.sin(2.0 * np.pi * idx)
            return out
        if name in ("lin_vel", "roboduet/base_lin_vel"):
            if name == "roboduet/base_lin_vel" and not p.get("observe_lin_vel", True):
                return np.zeros(3)
            return s["lin_vel"] * p["lin_vel_scale"]
        if name == "roboduet/body_pose_actual":
            if not p.get("observe_pose_actual", True):
                return np.zeros(3)
            euler = quat_to_euler_np(s["quat"])
            return np.array([s["base_height"] * p["body_height_cmd_scale"],
                             euler[1] * p["body_pitch_cmd_scale"],
                             euler[0] * p["body_roll_cmd_scale"]])
        if name == "roboduet/body_pose_error":
            if not p.get("observe_track_error", True):
                return np.zeros(3)
            euler = quat_to_euler_np(s["quat"])
            height_target = float(p["base_height_target"]) + s["cmd_height"]
            return np.array([(height_target - s["base_height"]) * p["body_height_cmd_scale"],
                             (s["cmd_pitch"] - euler[1]) * p["body_pitch_cmd_scale"],
                             (s["cmd_roll"] - euler[0]) * p["body_roll_cmd_scale"]])
        if name == "roboduet/velocity_error":
            if not p.get("observe_track_error", True):
                return np.zeros(3)
            return np.array([(s["cmd_x"] - s["lin_vel"][0]) * p["lin_vel_scale"],
                             (s["cmd_y"] - s["lin_vel"][1]) * p["lin_vel_scale"],
                             (s["cmd_yaw"] - s["ang_vel"][2]) * p["ang_vel_scale"]])
        raise KeyError(f"rl_sar term '{name}' is not implemented in this verifier")

    def assemble(self, s):
        parts = [np.asarray(self.term(name, s), dtype=np.float64) for name in self.p["observations"]]
        return np.clip(np.concatenate(parts), -self.p["clip_obs"], self.p["clip_obs"])
