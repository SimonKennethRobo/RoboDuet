"""RoboDuetRaw dual actors: learned arm + learned body-posture plan + dog.

The selected run trained plan_vel=False. Zero planar command is intentional
for this method, unlike silently replacing its arm actor with generic IK.
"""
from pathlib import Path
import pickle

import mujoco
import numpy as np
import torch

from sim2sim_mujoco import RlSarMujoco
from rl_sar_obs import quat_to_euler_np
from benchmark.wbc.mujoco import _quat_to_matrix


class RoboDuetRawMujoco(RlSarMujoco):
    controls_arm = True
    arm_position_drive = True

    def __init__(self, robot_dir, policy_key, scene, run_root, seed=0):
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

    def reset_policy(self):
        self.history.fill(0.)
        self.arm_history.fill(0.)
        self.actions.fill(0.)
        self.arm_commands.fill(0.)

    def set_reference(self, reference, time_s):
        _, self.goal_position, self.goal_quaternion = reference.at(time_s)

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
