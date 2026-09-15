"""RoboDuetRaw dual actors: learned arm + learned body-posture plan + dog.

The selected run trained plan_vel=False. The optional base follower supplies
operator-like velocity commands while the original arm/posture actor remains
in control. Native mode retains the original posture-only behavior.
"""
from pathlib import Path
import pickle

import mujoco
import numpy as np
import torch

from sim2sim_mujoco import RlSarMujoco
from rl_sar_obs import quat_to_euler_np
from benchmark.wbc.mujoco import _quat_to_matrix
from benchmark.wbc.omni_waypoint_follower import OmniFollowerConfig, configure_follower, follow_reference


class RoboDuetRawMujoco(RlSarMujoco):
    controls_arm = True
    arm_position_drive = True

    def __init__(self, robot_dir, policy_key, scene, run_root, seed=0,
                 base_mode="follow", target_mode="bounded"):
        super().__init__(robot_dir, policy_key, scene, seed=seed)
        self.run_root = Path(run_root).resolve()
        self.parameters_path = self.run_root / "parameters.pkl"
        with self.parameters_path.open("rb") as stream:
            cfg = pickle.load(stream)["Cfg"]
        if cfg["hybrid"]["plan_vel"] or cfg["hybrid"]["use_vision"] or cfg["arm"]["arm_num_observations"] != 20:
            raise ValueError("this adapter requires the selected 20D non-visual, posture-only Raw run")
        self.arm_model_paths = [self.run_root / "deploy_model" / name for name in
                                ("history_latest_arm.jit", "adaptation_module_latest_arm.jit", "body_latest_arm.jit")]
        self.arm_history_model, self.arm_adaptation, self.arm_body = [
            torch.jit.load(str(path), map_location="cpu").eval() for path in self.arm_model_paths]
        self.arm_history = np.zeros((30, 20), dtype=np.float32)
        self.arm_commands = np.zeros(6)
        self.goal_position = self.goal_quaternion = None
        self.p["action_scale"][12:18] = [float(cfg["control"]["action_scale"])] * 6
        self.arm_action_clip = float(cfg["normalization"]["clip_actions"])
        self.plan_limits = cfg["commands"]
        self.training_config = cfg
        if base_mode not in ("follow", "stand") or target_mode not in ("bounded", "native"):
            raise ValueError("Raw requires follow/stand base mode and bounded/native target mode")
        self.base_mode, self.target_mode = base_mode, target_mode
        self.base = self.model.body("base_link").id
        self.target_projected = False
        configure_follower(self, OmniFollowerConfig(
            max_speed_mps=.50, max_lateral_speed_mps=.45, max_backward_speed_mps=.35,
            max_acceleration_mps2=.70, max_yaw_rate_rps=1.0,
            max_yaw_acceleration_rps2=2.0))

    def reset_policy(self):
        self.history.fill(0.)
        self.arm_history.fill(0.)
        self.actions.fill(0.)
        self.arm_commands.fill(0.)
        self.command = [0.] * 6
        self.gait_indices = 0.
        self.follower.reset()
        self.target_projected = False

    def set_reference(self, reference, time_s):
        _, position, quaternion = reference.at(time_s)
        self.reference_position = np.asarray(position).copy()
        self.reference_quaternion = np.asarray(quaternion).copy()
        self.goal_position = self.reference_position.copy()
        self.goal_quaternion = self.reference_quaternion.copy()
        q, dq, quat, gyro, _, lin_vel = self.read_state()
        trunk = self.data.xpos[self.base].copy()
        if self.base_mode == "follow":
            follow_reference(self, reference, time_s)
        if self.target_mode == "bounded":
            self._bound_target(quat, trunk)

    def _bound_target(self, base_quat, trunk):
        """Project actor inputs into its trained spherical and Euler ranges."""
        yaw = quat_to_euler_np(base_quat)[2]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        center = np.array([trunk[0], trunk[1], .38])
        local = rotation.T @ (self.reference_position - center)
        lpy = np.array([np.linalg.norm(local),
                        np.arctan2(local[2], np.linalg.norm(local[:2])),
                        np.arctan2(local[1], local[0])])
        ranges = self.training_config["arm"]["commands"]
        bounded = np.array([np.clip(value, *ranges[key]) for value, key in zip(lpy, ("l", "p", "y"))])
        length, pitch, heading = bounded
        self.goal_position = center + rotation @ np.array([
            length * np.cos(pitch) * np.cos(heading),
            length * np.cos(pitch) * np.sin(heading), length * np.sin(pitch)])
        local_rotation = rotation.T @ _quat_to_matrix(self.reference_quaternion)
        wxyz = np.empty(4)
        mujoco.mju_mat2Quat(wxyz, local_rotation.reshape(-1))
        rpy = quat_to_euler_np(np.roll(wxyz, -1))
        r, p, y = [float(np.clip(value, float(ranges[key][0]), float(ranges[key][1])))
                   for value, key in zip(rpy, ("roll_ee", "pitch_ee", "yaw_ee"))]
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        bounded_rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                     [-sp, cp*sr, cp*cr]])
        mujoco.mju_mat2Quat(wxyz, (rotation @ bounded_rotation).reshape(-1))
        self.goal_quaternion = np.roll(wxyz, -1)
        self.target_projected = bool(np.linalg.norm(lpy - bounded) > 1e-7 or
                                     np.linalg.norm(local_rotation - bounded_rotation) > 1e-7)

    def forward(self, q, dq, quat, gyro, base_pos, lin_vel):
        if self.goal_position is None:
            raise RuntimeError("Raw arm target was not set")
        rpy = quat_to_euler_np(quat)
        yaw = rpy[2]
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_inverse = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        trunk = self.data.xpos[self.model.body("base_link").id]
        local = yaw_inverse @ (self.goal_position - np.array([trunk[0], trunk[1], .38]))
        rotation = yaw_inverse @ _quat_to_matrix(self.goal_quaternion)
        # Native quat_to_angle uses projected body axes, NOT standard Euler RPY.
        self.arm_commands = np.r_[np.linalg.norm(local),
            np.arctan2(local[2], np.linalg.norm(local[:2])), np.arctan2(local[1], local[0]),
            np.arctan2(rotation[2, 1], rotation[1, 1]),
            np.arctan2(rotation[0, 2], rotation[2, 2]),
            np.arctan2(rotation[1, 0], rotation[0, 0])]
        obs = np.r_[(q[12:18] - self.default_dof_pos[12:18]) * self.p["dof_pos_scale"],
                    self.actions[12:18], self.arm_commands, rpy[:2]]
        obs = np.clip(obs, -self.p["clip_obs"], self.p["clip_obs"]).astype(np.float32)
        self.arm_history[:-1] = self.arm_history[1:]
        self.arm_history[-1] = obs
        history = torch.from_numpy(self.arm_history.reshape(1, -1))
        with torch.inference_mode():
            latent = self.arm_adaptation(history)
            hist = self.arm_history_model(history[:, :-20])
            arm_plan = self.arm_body(torch.cat((torch.from_numpy(obs[None]), latent, hist), dim=-1))[0].numpy()
        if arm_plan.shape != (8,) or not np.isfinite(arm_plan).all():
            raise ValueError("invalid Raw arm/posture actor output")
        pitch_limits = self.plan_limits["limit_body_pitch"]
        self.command[3] = float(np.clip(.4 * arm_plan[6], pitch_limits[0], .75 * pitch_limits[1]))
        self.command[4] = float(np.clip(.4 * arm_plan[7], *self.plan_limits["limit_body_roll"]))
        actions = super().forward(q, dq, quat, gyro, base_pos, lin_vel)
        actions[12:18] = np.clip(arm_plan[:6], -self.arm_action_clip, self.arm_action_clip)
        self.actions = actions
        return actions.copy()
